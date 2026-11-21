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


@dataclass(slots=True)
class Roof:
    x: float
    y: float
    top: float
    area: float
    radius: float
    ring: list[float]


class Bot:
    name = "roof-rat-v1"

    def __init__(self, hello: mf.Hello):
        self.config = hello["config"]
        self.rng = random.Random(int(hello["seed"]))
        self.reach = float(self.config["loot"]["pickup_radius"])
        self.memory_s = float(self.config["vision"]["memory_s"])
        self.roofs = self._load_roofs(hello)
        self.enemy_memory: dict[int, tuple[float, list[float], list[float], float]] = {}
        self.roof_goal: Roof | None = None
        self.roof_goal_until = 0.0
        self.last_target: tuple[float, list[float], list[float]] | None = None
        self.last_hp = float(self.config["mech"]["hp"])
        self.last_damage_t = -999.0
        self.strafe_sign = -1.0 if hello["slot"] & 1 else 1.0

    @staticmethod
    def _load_roofs(hello: mf.Hello) -> list[Roof]:
        try:
            path = os.path.join(hello["map"]["dir"], "v2", "world.json")
            with open(path, "r", encoding="utf-8") as handle:
                world = json.load(handle)
        except (OSError, ValueError, KeyError):
            return []

        roofs: list[Roof] = []
        for prism in world.get("prisms", []):
            if prism.get("kind") != "building" or prism.get("holes"):
                continue
            ring = prism.get("ring", [])
            if len(ring) < 6:
                continue
            area = polygon_area(ring)
            top = float(prism.get("top", 0.0))
            base = float(prism.get("base", 0.0))
            # Large, jet-accessible roofs are reliable hiding platforms.
            if area < 140.0 or top - base > 54.0 or top > 58.0:
                continue
            x, y = polygon_centroid(ring)
            radius = max(math.hypot(ring[i] - x, ring[i + 1] - y) for i in range(0, len(ring), 2))
            roofs.append(Roof(x, y, top, area, radius, ring))
        return roofs

    def _remember_enemies(self, obs: mf.Observation) -> list[dict[str, Any]]:
        now = float(obs["t"])
        visible = [m for m in obs["mechs"] if m["kind"] == "mech"]
        for mech in visible:
            self.enemy_memory[int(mech["id"])] = (
                now,
                list(mech["pos"]),
                list(mech["vel"]),
                float(mech.get("hp", 1400.0)),
            )
        stale = [key for key, value in self.enemy_memory.items() if now - value[0] > self.memory_s]
        for key in stale:
            del self.enemy_memory[key]
        return visible

    def _inside_roof(self, pos: list[float], roof: Roof, z_slop: float = 2.0) -> bool:
        return abs(pos[2] - roof.top) <= z_slop and point_in_ring(pos[0], pos[1], roof.ring)

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
            score = 0.08 * min(roof.area, 1800.0) + 0.8 * separation - 0.55 * travel - 1.7 * max(0.0, roof.top - me[2])
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

        if kind == "mortar":
            # Low ballistic arc. Recompute after a first travel-time estimate.
            gravity = float(stats.get("gravity", 22.0))
            t = clamp(horizontal / max(speed * 0.75, 1.0), 0.15, 3.2)
            tx += vx * t
            ty += vy * t
            horizontal = math.hypot(tx - x, ty - y)
            height = (tz + 4.5) - (z + float(me["stats"].get("muzzle_height", 7.0)))
            disc = speed**4 - gravity * (gravity * horizontal**2 + 2.0 * height * speed**2)
            pitch = math.atan((speed**2 - math.sqrt(max(0.0, disc))) / max(gravity * horizontal, 1e-6)) if disc >= 0 else 0.55
            return math.atan2(ty - y, tx - x), clamp(pitch, -0.25, 1.05)

        t = clamp(horizontal / max(speed, 1.0), 0.0, 1.0) if kind == "gun" else 0.0
        tx += vx * t
        ty += vy * t
        tz += vz * t
        yaw = math.atan2(ty - y, tx - x)
        pitch = math.atan2((tz + 4.5) - (z + float(me["stats"].get("muzzle_height", 7.0))), math.hypot(tx - x, ty - y))
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
            "armor": 120,
            "plating": 115,
            "servos": 108,
            "nanites": 104,
            "jets": 100,
            "reactor": 92,
            "sensors": 86,
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
        choices = [p for p in obs["pickups"] if p["kind"] not in held and distance2(me["pos"], p["pos"]) <= self.reach]
        if not choices:
            return None, None
        value = {"mortar": 4, "laser": 3, "lightning": 2, "gun": 0}
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
        visible = self._remember_enemies(obs)

        damaged = [e for e in obs["events"] if e["type"] == "damaged"]
        if damaged or me["hp"] < self.last_hp - 0.5:
            self.last_damage_t = now
            if int(now * 2) & 1:
                self.strafe_sign *= -1.0
        self.last_hp = float(me["hp"])

        remembered: list[dict[str, Any]] = []
        visible_ids = {int(m["id"]) for m in visible}
        for enemy_id, (seen, epos, evel, hp) in self.enemy_memory.items():
            if enemy_id not in visible_ids:
                remembered.append({"id": enemy_id, "kind": "mech", "pos": epos, "vel": evel, "hp": hp, "seen": seen})
        all_threats = visible + remembered
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
        if self.roof_goal is not None:
            goal_x, goal_y = self.roof_goal.x, self.roof_goal.y

        # Opportunistic cubes first, then an uncontested landed supply drop.
        safe_items = danger < 0.35 and not recent_damage
        if safe_items and obs["cubes"]:
            cube = min(obs["cubes"], key=lambda c: distance2(pos, c["pos"]))
            if distance2(pos, cube["pos"]) < 55.0:
                goal_x, goal_y = cube["pos"][:2]
                on_roof = False
        if safe_items and hp_frac < 0.72:
            landed = [d for d in obs["drops"] if d["lands_in"] <= 0 and (d["holder"] is None or d["holder"] == me["id"])]
            if landed:
                drop = min(landed, key=lambda d: distance2(pos, d["pos"]))
                if distance2(pos, drop["pos"]) < 80.0:
                    goal_x, goal_y = drop["pos"][:2]
                    on_roof = False

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

        act: mf.Action = {"tick": obs["tick"], "move": move}

        energy_frac = float(me["energy"]) / max(float(me["max_energy"]), 1.0)
        climbing = (
            self.roof_goal is not None
            and not on_roof
            and distance2(pos, (self.roof_goal.x, self.roof_goal.y)) < self.roof_goal.radius + 8.0
            and pos[2] < self.roof_goal.top - 0.8
        )
        if climbing:
            act["jump"] = bool(me["grounded"])
            act["boost"] = energy_frac > 0.12
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
        target: dict[str, Any] | None = min(visible, key=lambda m: distance2(pos, m["pos"])) if visible else None
        if target is None and danger < 0.3:
            npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
            if npcs:
                target = min(npcs, key=lambda m: distance2(pos, m["pos"]))
            elif obs["crates"]:
                crate = min(obs["crates"], key=lambda c: distance2(pos, c["pos"]))
                target = {"id": crate["id"], "kind": "crate", "pos": crate["pos"], "vel": [0.0, 0.0, 0.0], "hp": crate["hp"]}

        if target is not None:
            primary = me["weapons"][0]
            aim_yaw, aim_pitch = self._lead_aim(me, target, primary)
            act["aim"] = [aim_yaw, aim_pitch]
            self.last_target = (now, list(target["pos"]), list(target.get("vel", [0.0, 0.0, 0.0])))
            aim_error = abs(wrap_angle(aim_yaw - float(me["aim"][0])))
            target_dist = distance2(pos, target["pos"])
            fire0 = bool(primary and target_dist <= float(primary["stats"].get("range", 230.0)) and aim_error < 0.12)
            fire1 = False
            secondary = me["weapons"][1]
            if secondary:
                kind = secondary["kind"]
                secondary_yaw, secondary_pitch = self._lead_aim(me, target, secondary)
                secondary_error = abs(wrap_angle(secondary_yaw - float(me["aim"][0])))
                if kind == "lightning":
                    fire1 = target_dist < float(secondary["stats"].get("range", 45.0)) * 0.96 and secondary_error < 0.42
                elif kind == "mortar":
                    if 18.0 < target_dist < 220.0 and secondary_error < 0.16:
                        act["aim"] = [secondary_yaw, secondary_pitch]
                        fire1 = True
                elif kind == "laser":
                    charge = float(secondary.get("charge", 0.0))
                    needed = float(secondary["stats"].get("charge", 0.75))
                    in_range = target_dist < float(secondary["stats"].get("range", 280.0))
                    # Hold while charging; release at full charge with a lined-up target.
                    fire1 = bool(in_range and (charge < needed * 0.92 or secondary_error > 0.10))
            act["fire"] = [fire0, fire1]
            if primary and primary["kind"] == "gun" and primary.get("ammo", 1) <= 2 and not fire0:
                act["reload"] = True
        else:
            # Sweep the horizon slowly to acquire hunters before they reach sensor range.
            act["aim"] = [wrap_angle(now * 0.48 + self.rng.random() * 0.01), -0.03]
            gun = me["weapons"][0]
            if gun and gun["kind"] == "gun" and gun.get("ammo", 12) < gun["stats"].get("mag", 12) * 0.45:
                act["reload"] = True

        return act


if __name__ == "__main__":
    mf.run(Bot)
