"""Survival-first Mech Royale bot.

The practice field is a coordinated 15-bot squad and placement is the only
score.  This bot therefore values cover, zone position, regeneration, and
disengagement above kills.  It still returns fire while retreating so a single
hunter cannot chase for free.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable

import mechfight as mf


TAU = 2.0 * math.pi


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def wrap_angle(value: float) -> float:
    return (value + math.pi) % TAU - math.pi


def length2(v: tuple[float, float] | list[float]) -> float:
    return math.hypot(v[0], v[1])


def distance2(a: list[float], b: list[float] | tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(x: float, y: float) -> tuple[float, float]:
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-6 else (0.0, 0.0)


def polygon_area(ring: list[float]) -> float:
    total = 0.0
    count = len(ring) // 2
    for i in range(count):
        x1, y1 = ring[2 * i], ring[2 * i + 1]
        x2, y2 = ring[2 * ((i + 1) % count)], ring[2 * ((i + 1) % count) + 1]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def polygon_centroid(ring: list[float]) -> tuple[float, float]:
    # Vertex mean is guaranteed to stay useful for the mostly rectangular map
    # footprints and is stable for the few degenerate scenery polygons.
    xs = ring[0::2]
    ys = ring[1::2]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def point_segment_distance(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(x - ax, y - ay)
    t = clamp(((x - ax) * dx + (y - ay) * dy) / denom, 0.0, 1.0)
    return math.hypot(x - (ax + t * dx), y - (ay + t * dy))


def ring_edge_distance(x: float, y: float, ring: list[float]) -> float:
    count = len(ring) // 2
    return min(
        point_segment_distance(
            x,
            y,
            ring[2 * i],
            ring[2 * i + 1],
            ring[2 * ((i + 1) % count)],
            ring[2 * ((i + 1) % count) + 1],
        )
        for i in range(count)
    )


def inside_surface(x: float, y: float, ring: list[float], holes: list[list[float]]) -> bool:
    return point_in_ring(x, y, ring) and not any(point_in_ring(x, y, hole) for hole in holes)


def surface_clearance(x: float, y: float, ring: list[float], holes: list[list[float]]) -> float:
    if not inside_surface(x, y, ring, holes):
        return -1.0
    return min([ring_edge_distance(x, y, ring), *(ring_edge_distance(x, y, hole) for hole in holes)])


def maximum_clearance_point(ring: list[float], holes: list[list[float]]) -> tuple[float, float, float] | None:
    """Cheap polylabel-style search; run once during the 20-second hello budget."""
    xs, ys = ring[0::2], ring[1::2]
    best: tuple[float, float, float] | None = None
    step = 4.0
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    y = min_y
    while y <= max_y:
        x = min_x
        while x <= max_x:
            clearance = surface_clearance(x, y, ring, holes)
            if best is None or clearance > best[2]:
                best = (x, y, clearance)
            x += step
        y += step
    if best is None or best[2] < 0:
        return None
    # Two local refinements reach sub-meter accuracy without a geometry package.
    for step in (1.0, 0.25):
        bx, by, _ = best
        for iy in range(-5, 6):
            for ix in range(-5, 6):
                x, y = bx + ix * step, by + iy * step
                clearance = surface_clearance(x, y, ring, holes)
                if clearance > best[2]:
                    best = (x, y, clearance)
    return best


def point_in_ring(x: float, y: float, ring: list[float]) -> bool:
    inside = False
    count = len(ring) // 2
    j = count - 1
    for i in range(count):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def segment_intersection_t(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float
) -> float | None:
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-9:
        return None
    qx, qy = cx - ax, cy - ay
    t = (qx * sy - qy * sx) / denom
    u = (qx * ry - qy * rx) / denom
    return t if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


@dataclass(slots=True)
class Roof:
    x: float
    y: float
    top: float
    area: float
    clearance: float
    ring: list[float]
    holes: list[list[float]]


class Bot:
    name = "roof-rat-v2"

    def __init__(self, hello: mf.Hello):
        self.config = hello["config"]
        self.rng = random.Random(int(hello["seed"]))
        self.reach = float(self.config["loot"]["pickup_radius"])
        self.memory_s = float(self.config["vision"]["memory_s"])
        self.prisms: list[dict[str, Any]] = []
        self.roofs = self._load_roofs(hello)
        self.enemy_memory: dict[int, tuple[float, list[float], list[float], float]] = {}
        self.roof_goal: Roof | None = None
        self.roof_goal_until = 0.0
        self.last_target: tuple[float, list[float], list[float]] | None = None
        self.last_hp = float(self.config["mech"]["hp"])
        self.last_damage_t = -999.0
        self.last_damage_bearing: float | None = None
        self.last_damage_bearing_t = -999.0
        self.strafe_sign = -1.0 if hello["slot"] & 1 else 1.0
        self.last_strafe_flip_t = -999.0

    def _load_roofs(self, hello: mf.Hello) -> list[Roof]:
        try:
            path = os.path.join(hello["map"]["dir"], "v2", "world.json")
            with open(path, "r", encoding="utf-8") as handle:
                world = json.load(handle)
        except (OSError, ValueError, KeyError):
            return []

        self.prisms = list(world.get("prisms", []))
        for solid in self.prisms:
            outer = solid.get("ring", [])
            if outer:
                solid["_bbox"] = (min(outer[0::2]), min(outer[1::2]), max(outer[0::2]), max(outer[1::2]))
        roofs: list[Roof] = []
        for prism in self.prisms:
            if prism.get("kind") != "building":
                continue
            ring = prism.get("ring", [])
            holes = prism.get("holes", [])
            if len(ring) < 6:
                continue
            area = polygon_area(ring)
            top = float(prism.get("top", 0.0))
            base = float(prism.get("base", 0.0))
            # Large, jet-accessible roofs are reliable hiding platforms.
            if area < 140.0 or top - base > 54.0 or top > 58.0:
                continue
            site = maximum_clearance_point(ring, holes)
            if site is None:
                continue
            x, y, clearance = site
            if clearance < 3.1:
                continue
            # A lower overlapping prism is not a usable landing surface here.
            covered = any(
                other is not prism
                and float(other.get("base", 0.0)) <= top + 0.2
                and float(other.get("top", 0.0)) > top + 0.8
                and inside_surface(x, y, other.get("ring", []), other.get("holes", []))
                for other in self.prisms
                if len(other.get("ring", [])) >= 6
            )
            if covered:
                continue
            roofs.append(Roof(x, y, top, area, clearance, ring, holes))
        return roofs

    def _ray_blocked(self, a: list[float], b: list[float]) -> bool:
        seg_min_x, seg_max_x = sorted((a[0], b[0]))
        seg_min_y, seg_max_y = sorted((a[1], b[1]))
        for solid in self.prisms:
            bbox = solid.get("_bbox")
            ring = solid.get("ring", [])
            if not bbox or bbox[2] < seg_min_x or bbox[0] > seg_max_x or bbox[3] < seg_min_y or bbox[1] > seg_max_y:
                continue
            cuts = [0.0, 1.0]
            for boundary in [ring, *solid.get("holes", [])]:
                count = len(boundary) // 2
                for i in range(count):
                    t = segment_intersection_t(
                        a[0], a[1], b[0], b[1],
                        boundary[2 * i], boundary[2 * i + 1],
                        boundary[2 * ((i + 1) % count)], boundary[2 * ((i + 1) % count) + 1],
                    )
                    if t is not None:
                        cuts.append(t)
            cuts.sort()
            for lo, hi in zip(cuts, cuts[1:]):
                t = (lo + hi) * 0.5
                x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                if not inside_surface(x, y, ring, solid.get("holes", [])):
                    continue
                z = a[2] + (b[2] - a[2]) * t
                if float(solid.get("base", -1e9)) <= z <= float(solid.get("top", 1e9)):
                    return True
        return False

    @staticmethod
    def _target_height(target: dict[str, Any]) -> tuple[float, float]:
        kind = target.get("kind")
        if kind == "npc":
            return 1.8, 3.5
        if kind == "crate":
            return 1.2, 2.4
        scale = 1.0 + 0.05 * float(target.get("cubes", 0))
        return 4.5 * scale, 9.0 * scale

    def _confirmed_visible(self, me: dict[str, Any], target: dict[str, Any]) -> bool:
        distance = distance2(me["pos"], target["pos"])
        near = float(self.config["vision"]["near_range"]) + float(me["stats"].get("near_range_bonus", 0.0))
        far = float(self.config["vision"]["range"]) + float(me["stats"].get("vision_bonus", 0.0))
        bearing = math.atan2(target["pos"][1] - me["pos"][1], target["pos"][0] - me["pos"][0])
        in_view = distance <= near or (
            distance <= far and abs(wrap_angle(bearing - float(me["aim"][0]))) <= math.radians(float(self.config["vision"]["fov_deg"])) * 0.5
        )
        if not in_view:
            return False
        center, height = self._target_height(target)
        eye = [me["pos"][0], me["pos"][1], me["pos"][2] + float(me["stats"].get("eye_height", 8.0))]
        for z_offset in (center, height):
            point = [target["pos"][0], target["pos"][1], target["pos"][2] + z_offset]
            if not self._ray_blocked(eye, point):
                return True
        return False

    def _inside_roof(self, pos: list[float], roof: Roof, z_slop: float = 2.0) -> bool:
        return abs(pos[2] - roof.top) <= z_slop and inside_surface(pos[0], pos[1], roof.ring, roof.holes)

    def _select_roof(self, obs: mf.Observation, threats: Iterable[list[float]]) -> Roof | None:
        me = obs["self"]["pos"]
        zone = obs["zone"]
        center = zone["next_center"]
        radius = float(zone["next_radius"])
        threat_points = list(threats)
        best: Roof | None = None
        best_score = -1e30
        for roof in self.roofs:
            center_dist = math.hypot(roof.x - center[0], roof.y - center[1])
            # Leave room for the next shrink and for our cylinder.
            if center_dist > max(8.0, radius - 8.0):
                continue
            travel = math.hypot(roof.x - me[0], roof.y - me[1])
            if travel > 260.0:
                continue
            separation = min(
                (math.hypot(roof.x - p[0], roof.y - p[1]) for p in threat_points),
                default=220.0,
            )
            # Prefer broad, low roofs that are far from the squad and already nearby.
            score = 5.0 * min(roof.clearance, 22.0) + 0.8 * separation - 0.48 * travel - 1.4 * max(0.0, roof.top - me[2])
            if score > best_score:
                best_score, best = score, roof
        return best

    @staticmethod
    def _lead_aim(me: dict[str, Any], target: dict[str, Any], weapon: dict[str, Any] | None) -> tuple[float, float]:
        x, y, z = me["pos"]
        tx, ty, tz = target["pos"]
        vx, vy, vz = target.get("vel", [0.0, 0.0, 0.0])
        kind = weapon.get("kind") if weapon else "gun"
        stats = weapon.get("stats", {}) if weapon else {}
        speed = float(stats.get("speed", 252.0))
        horizontal = math.hypot(tx - x, ty - y)
        target_center, _ = Bot._target_height(target)

        if kind == "mortar":
            # Low ballistic arc. Recompute after a first travel-time estimate.
            gravity = float(stats.get("gravity", 22.0))
            t = clamp(horizontal / max(speed * 0.75, 1.0), 0.15, 3.2)
            tx += vx * t
            ty += vy * t
            horizontal = math.hypot(tx - x, ty - y)
            height = (tz + target_center) - (z + float(me["stats"].get("muzzle_height", 7.0)))
            disc = speed**4 - gravity * (gravity * horizontal**2 + 2.0 * height * speed**2)
            pitch = math.atan((speed**2 - math.sqrt(max(0.0, disc))) / max(gravity * horizontal, 1e-6)) if disc >= 0 else 0.55
            return math.atan2(ty - y, tx - x), clamp(pitch, -0.25, 1.05)

        t = clamp(horizontal / max(speed, 1.0), 0.0, 1.0) if kind == "gun" else 0.0
        tx += vx * t
        ty += vy * t
        tz += vz * t
        yaw = math.atan2(ty - y, tx - x)
        pitch = math.atan2((tz + target_center) - (z + float(me["stats"].get("muzzle_height", 7.0))), math.hypot(tx - x, ty - y))
        return yaw, pitch

    def _projectile_evade(self, obs: mf.Observation) -> tuple[float, float]:
        me = obs["self"]
        mx, my = me["pos"][:2]
        ex = ey = 0.0
        for projectile in obs["projectiles"]:
            px, py = projectile["pos"][:2]
            vx, vy = projectile["vel"][:2]
            vv = vx * vx + vy * vy
            if vv < 1.0:
                continue
            t = clamp(((mx - px) * vx + (my - py) * vy) / vv, 0.0, 0.9)
            cx, cy = px + vx * t, py + vy * t
            miss_x, miss_y = mx - cx, my - cy
            miss = math.hypot(miss_x, miss_y)
            if miss < (10.0 if projectile["kind"] == "mortar" else 5.0):
                side = 1.0 if vx * miss_y - vy * miss_x >= 0 else -1.0
                ex += -vy * side / math.sqrt(vv) * (12.0 - miss)
                ey += vx * side / math.sqrt(vv) * (12.0 - miss)
        return ex, ey

    def _upgrade_pick(self, offer: dict[str, Any]) -> int:
        priority = {
            "plating": 122,
            "servos": 118,
            "jets": 114,
            "nanites": 110,
            "armor": 106,
            "sensors": 102,
            "reactor": 92,
            "capacitor": 82,
            "gun_rate": 76,
            "gun_damage": 74,
            "mortar_reload": 73,
            "mortar_splash": 72,
            "laser_damage": 70,
            "lightning_chain": 69,
            "gyros": 64,
        }
        return max(range(len(offer["choices"])), key=lambda i: priority.get(offer["choices"][i]["id"], 50))

    def _take_weapon(self, obs: mf.Observation, danger: float) -> tuple[int | None, int | None]:
        me = obs["self"]
        held = {w["kind"] for w in me["weapons"] if w}
        choices = [
            p for p in obs["pickups"]
            if p["kind"] not in held
            and distance2(me["pos"], p["pos"]) <= self.reach
            and abs(float(me["pos"][2]) - float(p["pos"][2])) <= 4.0
        ]
        if not choices:
            return None, None
        value = {"laser": 4, "mortar": 3, "lightning": 2, "gun": 0}
        pick = max(choices, key=lambda p: value.get(p["kind"], 0))
        if me["weapons"][1] is None:
            return int(pick["id"]), None
        if danger < 0.45 and value.get(pick["kind"], 0) > value.get(me["weapons"][1]["kind"], 0):
            return int(pick["id"]), 1
        return None, None

    def act(self, obs: mf.Observation) -> mf.Action:
        now = float(obs["t"])
        me = obs["self"]
        pos = me["pos"]
        # The server itself retains exact enemy positions for five seconds after
        # LOS breaks. Keep all of them as threats, but fire direct weapons only
        # at actors whose center/top is currently visible through map solids.
        all_threats = [m for m in obs["mechs"] if m["kind"] == "mech"]
        visible = [m for m in all_threats if self._confirmed_visible(me, m)]

        damaged = [e for e in obs["events"] if e["type"] == "damaged"]
        if damaged or me["hp"] < self.last_hp - 0.5:
            self.last_damage_t = now
            bearings = [float(e["bearing"]) for e in damaged if e.get("bearing") is not None]
            if bearings:
                self.last_damage_bearing = math.atan2(
                    sum(math.sin(value) for value in bearings), sum(math.cos(value) for value in bearings)
                )
                self.last_damage_bearing_t = now
            if now - self.last_strafe_flip_t > 0.65:
                self.strafe_sign *= -1.0
                self.last_strafe_flip_t = now
        self.last_hp = float(me["hp"])
        all_threats.sort(key=lambda m: distance2(pos, m["pos"]))
        nearest = all_threats[0] if all_threats else None
        nearest_dist = distance2(pos, nearest["pos"]) if nearest else 1e9

        hp_frac = float(me["hp"]) / max(float(me["max_hp"]), 1.0)
        recent_damage = now - self.last_damage_t < 5.5
        danger = max(0.0, 1.0 - nearest_dist / 170.0)
        danger = max(danger, 0.9 if damaged else 0.0, 0.65 if recent_damage else 0.0)
        danger *= 1.25 if hp_frac < 0.5 else 1.0

        # Refresh a roof destination once per second or when the next circle invalidates it.
        zone = obs["zone"]
        if self.roof_goal is not None:
            safe_r = max(6.0, float(zone["next_radius"]) - 7.0)
            if math.hypot(self.roof_goal.x - zone["next_center"][0], self.roof_goal.y - zone["next_center"][1]) > safe_r:
                self.roof_goal = None
        if self.roof_goal is None or now >= self.roof_goal_until:
            candidate = self._select_roof(obs, [m["pos"] for m in all_threats])
            if candidate is not None:
                self.roof_goal = candidate
            # Avoid oscillating between similarly scored roofs as enemies move.
            self.roof_goal_until = now + 8.0

        on_roof = self.roof_goal is not None and self._inside_roof(pos, self.roof_goal)
        goal_x, goal_y = zone["next_center"]
        objective = "zone"
        if self.roof_goal is not None:
            goal_x, goal_y = self.roof_goal.x, self.roof_goal.y
            objective = "roof"

        # Opportunistic cubes first, then an uncontested landed supply drop.
        safe_items = danger < 0.35 and not recent_damage
        if safe_items and int(me["cubes"]) < 10 and obs["cubes"]:
            cube = min(obs["cubes"], key=lambda c: distance2(pos, c["pos"]))
            threshold = 60.0 if me["cubes"] < 1 else 40.0 if me["cubes"] < 3 else 25.0
            if not on_roof and distance2(pos, cube["pos"]) < threshold:
                goal_x, goal_y = cube["pos"][:2]
                on_roof = False
                objective = "cube"
        if safe_items and hp_frac < 0.45:
            landed = [d for d in obs["drops"] if d["lands_in"] <= 0 and (d["holder"] is None or d["holder"] == me["id"])]
            if landed:
                drop = min(landed, key=lambda d: distance2(pos, d["pos"]))
                contested = any(distance2(drop["pos"], threat["pos"]) < 160.0 for threat in all_threats)
                if not contested and not on_roof and distance2(pos, drop["pos"]) < 65.0:
                    goal_x, goal_y = drop["pos"][:2]
                    on_roof = False
                    objective = "drop"

        # Navigation vector: objective + strong repulsion from the coordinated squad.
        gx, gy = unit(goal_x - pos[0], goal_y - pos[1])
        move_x, move_y = gx, gy
        for threat in all_threats:
            dx, dy = pos[0] - threat["pos"][0], pos[1] - threat["pos"][1]
            dist = max(math.hypot(dx, dy), 1.0)
            weight = clamp((180.0 - dist) / 110.0, 0.0, 1.8)
            move_x += dx / dist * weight
            move_y += dy / dist * weight
            # Tangential movement is harder for guns and lasers to lead.
            if dist < 110.0:
                move_x += -dy / dist * 0.45 * self.strafe_sign
                move_y += dx / dist * 0.45 * self.strafe_sign

        for event in damaged:
            bearing = event.get("bearing")
            if bearing is not None:
                move_x -= math.cos(bearing) * 1.4
                move_y -= math.sin(bearing) * 1.4

        evade_x, evade_y = self._projectile_evade(obs)
        move_x += evade_x
        move_y += evade_y

        # Never let avoidance push us out of the currently safe circle.
        current_dist = math.hypot(pos[0] - zone["center"][0], pos[1] - zone["center"][1])
        edge = float(zone["radius"]) - current_dist
        if not me["in_zone"] or edge < 28.0:
            zx, zy = unit(zone["center"][0] - pos[0], zone["center"][1] - pos[1])
            force = 3.5 if not me["in_zone"] else clamp((28.0 - edge) / 8.0, 0.0, 3.0)
            move_x += zx * force
            move_y += zy * force

        # A quiet, regenerating rooftop mech is harder to discover than a runner.
        settled = on_roof and danger < 0.22 and not recent_damage and distance2(pos, (goal_x, goal_y)) < 12.0
        if settled:
            move = [0.0, 0.0]
        else:
            move = list(unit(move_x, move_y))

        # At the lethal edge, avoidance may never point outward. Project onto a
        # minimum inward component instead of letting many repulsions cancel it.
        inward = unit(zone["center"][0] - pos[0], zone["center"][1] - pos[1])
        inward_dot = move[0] * inward[0] + move[1] * inward[1]
        if (not me["in_zone"] or float(zone["radius"]) <= 25.0 or edge < 8.0) and inward_dot < 0.35:
            tangent_x, tangent_y = -inward[1], inward[0]
            tangent_component = move[0] * tangent_x + move[1] * tangent_y
            tangent_component = clamp(tangent_component, -0.75, 0.75)
            move = list(unit(inward[0] * 0.55 + tangent_x * tangent_component, inward[1] * 0.55 + tangent_y * tangent_component))
        elif danger < 0.10 and not recent_damage and not settled and distance2(pos, (goal_x, goal_y)) < 40.0:
            # Analog movement below 5 m/s is silent.
            quiet = min(0.39, 4.7 / max(float(me["stats"].get("walk_speed", 11.7)), 1.0))
            move = [move[0] * quiet, move[1] * quiet]

        act: mf.Action = {"tick": obs["tick"], "move": move}

        energy_frac = float(me["energy"]) / max(float(me["max_energy"]), 1.0)
        climbing = (
            self.roof_goal is not None
            and objective == "roof"
            and not on_roof
            and (
                inside_surface(pos[0], pos[1], self.roof_goal.ring, self.roof_goal.holes)
                or ring_edge_distance(pos[0], pos[1], self.roof_goal.ring) < 8.0
            )
            and pos[2] < self.roof_goal.top - 0.8
        )
        if climbing:
            vertical = max(0.0, self.roof_goal.top - pos[2])
            energy_needed = vertical * float(self.config["mech"]["jet_drain"]) / max(float(self.config["mech"]["max_rise_speed"]), 1.0) + 12.0
            climb_ready = float(me["energy"]) >= min(float(me["max_energy"]), energy_needed)
            act["jump"] = bool(me["grounded"] and climb_ready)
            act["boost"] = bool(not me["grounded"] and climb_ready and energy_frac > 0.08)
        else:
            should_sprint = (danger > 0.32 or not me["in_zone"] or distance2(pos, (goal_x, goal_y)) > 90.0) and not settled
            act["boost"] = should_sprint and energy_frac > (0.18 if danger > 0.7 else 0.35)
            act["jump"] = bool(me["grounded"] and danger > 0.72 and obs["tick"] % 24 < 2)

        # Pick upgrades immediately; defensive mobility dominates placement value.
        if obs["offer"]:
            act["pick"] = self._upgrade_pick(obs["offer"])

        take, drop = self._take_weapon(obs, danger)
        if take is not None:
            act["take"] = take
            if drop is not None:
                act["drop"] = drop

        # Aim at the nearest visible hunter, or at a crate/NPC while safely farming.
        attacker_ids = [int(event["by"]) for event in damaged if event.get("by") is not None]
        target: dict[str, Any] | None = next((m for m in visible if int(m["id"]) in attacker_ids), None)
        if target is None and visible:
            target = min(visible, key=lambda m: distance2(pos, m["pos"]))
        if target is None and danger < 0.15 and hp_frac > 0.85 and now - self.last_damage_t > 8.0:
            npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
            npcs = [m for m in npcs if distance2(pos, m["pos"]) < 25.0 and self._confirmed_visible(me, m)]
            if npcs:
                target = min(npcs, key=lambda m: distance2(pos, m["pos"]))
            elif obs["crates"]:
                crate = min(obs["crates"], key=lambda c: distance2(pos, c["pos"]))
                if distance2(pos, crate["pos"]) < 18.0:
                    target = {"id": crate["id"], "kind": "crate", "pos": crate["pos"], "vel": [0.0, 0.0, 0.0], "hp": crate["hp"]}

        if target is not None:
            primary = me["weapons"][0]
            aim_yaw, aim_pitch = self._lead_aim(me, target, primary)
            act["aim"] = [aim_yaw, aim_pitch]
            self.last_target = (now, list(target["pos"]), list(target.get("vel", [0.0, 0.0, 0.0])))
            target_dist = distance2(pos, target["pos"])
            target_center, target_height = self._target_height(target)
            target_radius = 1.8 if target.get("kind") == "npc" else 1.4 if target.get("kind") == "crate" else 2.5 * (1.0 + 0.05 * float(target.get("cubes", 0)))
            yaw_error = abs(wrap_angle(aim_yaw - float(me["aim"][0])))
            pitch_error = abs(aim_pitch - float(me["aim"][1]))
            yaw_tol = clamp(math.atan2(target_radius * 0.9, max(target_dist, 1.0)) + 0.008, 0.014, 0.065)
            pitch_tol = clamp(math.atan2(target_height * 0.42, max(target_dist, 1.0)) + 0.008, 0.014, 0.08)
            recovering = hp_frac < 0.55 and nearest_dist > 90.0 and not damaged
            fire0 = bool(
                primary
                and not recovering
                and target_dist <= float(primary["stats"].get("range", 230.0))
                and yaw_error < yaw_tol
                and pitch_error < pitch_tol
            )
            fire1 = False
            secondary = me["weapons"][1]
            if secondary:
                kind = secondary["kind"]
                secondary_yaw, secondary_pitch = self._lead_aim(me, target, secondary)
                secondary_yaw_error = abs(wrap_angle(secondary_yaw - float(me["aim"][0])))
                secondary_pitch_error = abs(secondary_pitch - float(me["aim"][1]))
                if kind == "lightning":
                    fire1 = (
                        target_dist < float(secondary["stats"].get("range", 45.0)) * 0.96
                        and secondary_yaw_error < 0.35
                        and (float(me["energy"]) >= 45.0 or target_dist < 18.0)
                    )
                elif kind == "mortar":
                    mortar_tol = clamp(math.atan2(float(secondary["stats"].get("splash_radius", 7.0)) * 0.7, max(target_dist, 1.0)), 0.025, 0.075)
                    if 18.0 < target_dist < 220.0 and secondary_yaw_error < mortar_tol and secondary_pitch_error < 0.05 and not recovering:
                        act["aim"] = [secondary_yaw, secondary_pitch]
                        fire1 = True
                        fire0 = False
                elif kind == "laser":
                    charge = float(secondary.get("charge", 0.0))
                    needed = float(secondary["stats"].get("charge", 0.75))
                    in_range = target_dist < float(secondary["stats"].get("range", 280.0))
                    aligned = secondary_yaw_error < yaw_tol and secondary_pitch_error < pitch_tol
                    # Charging halves movement speed: never begin a charge while escaping.
                    if charge <= 0.0 and (danger > 0.45 or hp_frac < 0.55 or target_dist < 90.0):
                        fire1 = False
                    elif charge > 0.0 and (danger > 0.72 or recovering):
                        fire1 = False
                    else:
                        fire1 = bool(in_range and (charge < needed - 0.02 or not aligned))
            act["fire"] = [fire0, fire1]
            if primary and primary["kind"] == "gun" and primary.get("ammo", 1) <= 2 and not fire0:
                act["reload"] = True
        else:
            if self.last_damage_bearing is not None and now - self.last_damage_bearing_t < 2.0:
                act["aim"] = [self.last_damage_bearing, 0.0]
            else:
                # Revisit every bearing before the server's five-second memory expires.
                act["aim"] = [wrap_angle(now * 1.1 + self.rng.random() * 0.01), -0.03]
            gun = me["weapons"][0]
            if gun and gun["kind"] == "gun" and gun.get("ammo", 12) < gun["stats"].get("mag", 12) * 0.45:
                act["reload"] = True

        return act


if __name__ == "__main__":
    mf.run(Bot)
