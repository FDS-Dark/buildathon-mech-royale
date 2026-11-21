"""Deterministic policy fuzzing against the official recorded observation schema."""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import random
import time

from bot import Bot


KIT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mechfight-kit"))
with gzip.open(os.path.join(KIT, "check", "stream.jsonl.gz"), "rt", encoding="utf-8") as handle:
    MESSAGES = [json.loads(line) for line in handle]

HELLO = copy.deepcopy(MESSAGES[0])
HELLO["map"]["dir"] = os.path.join(KIT, "arena")
BASE = next(message for message in MESSAGES if message.get("type") == "obs")


def validate(action: dict, tick: int) -> None:
    assert action["tick"] == tick
    assert len(json.dumps(action, separators=(",", ":"))) < 65536
    if "move" in action:
        assert len(action["move"]) == 2
        assert math.hypot(*action["move"]) <= 1.000001
    if "aim" in action:
        assert len(action["aim"]) == 2
        assert all(math.isfinite(value) for value in action["aim"])
    if "fire" in action:
        assert len(action["fire"]) == 2


def main() -> None:
    rng = random.Random(67)
    bot = Bot(HELLO)
    assert len(bot.roofs) >= 50
    assert len(bot.hide_sites) >= 500
    start = time.perf_counter()
    for index in range(3000):
        obs = copy.deepcopy(BASE)
        obs["tick"] = index * 2
        obs["t"] = index / 10
        me = obs["self"]
        me["pos"] = [rng.uniform(-450, 500), rng.uniform(-300, 340), rng.uniform(-4, 55)]
        me["vel"] = [rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-10, 10)]
        me["hp"] = rng.uniform(1, me["max_hp"])
        me["energy"] = rng.uniform(0, me["max_energy"])
        me["grounded"] = rng.random() < 0.72
        me["in_zone"] = rng.random() < 0.85
        obs["mechs"] = []
        for enemy_id in range(rng.randrange(8)):
            obs["mechs"].append(
                {
                    "id": enemy_id + 4,
                    "kind": "mech",
                    "pos": [me["pos"][0] + rng.uniform(-220, 220), me["pos"][1] + rng.uniform(-220, 220), me["pos"][2]],
                    "vel": [rng.uniform(-20, 20), rng.uniform(-20, 20), 0],
                    "aim": [rng.uniform(-math.pi, math.pi), 0],
                    "hp": rng.uniform(1, 1400),
                    "max_hp": 1400,
                    "cubes": rng.randrange(12),
                    "weapons": ["gun"],
                    "charging": False,
                }
            )
        obs["projectiles"] = [
            {
                "id": 100 + p,
                "kind": rng.choice(["gun", "mortar"]),
                "pos": [me["pos"][0] + rng.uniform(-40, 40), me["pos"][1] + rng.uniform(-40, 40), me["pos"][2] + 4],
                "vel": [rng.uniform(-250, 250), rng.uniform(-250, 250), rng.uniform(-30, 30)],
            }
            for p in range(rng.randrange(5))
        ]
        obs["events"] = []
        if rng.random() < 0.2:
            obs["events"].append(
                {"type": "damaged", "amount": 45, "weapon": "gun", "bearing": rng.uniform(-math.pi, math.pi), "by": 4, "rear": rng.random() < 0.5}
            )
        obs["zone"]["next_center"] = [rng.uniform(-100, 150), rng.uniform(-180, 80)]
        obs["zone"]["next_radius"] = rng.choice([380, 240, 140, 70, 25])
        action = bot.act(obs)
        validate(action, obs["tick"])

    elapsed = time.perf_counter() - start

    # Regression: a symmetric enemy wall must never push us outward during the
    # lethal final collapse.
    final_obs = copy.deepcopy(BASE)
    final_obs["tick"] = 8000
    final_obs["t"] = 400.0
    final_obs["self"]["pos"] = [12.0, 0.0, 0.0]
    final_obs["self"]["hp"] = 300.0
    final_obs["self"]["in_zone"] = False
    final_obs["zone"].update(center=[0.0, 0.0], radius=0.0, next_center=[0.0, 0.0], next_radius=0.0, dps=260.0)
    final_obs["mechs"] = []
    for i in range(8):
        angle = -0.8 + i * 1.6 / 7
        final_obs["mechs"].append(
            {"id": i + 4, "kind": "mech", "pos": [12 + 25 * math.cos(angle), 25 * math.sin(angle), 0],
             "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1400, "max_hp": 1400,
             "cubes": 0, "weapons": ["gun"], "charging": False}
        )
    action = bot.act(final_obs)
    assert action["move"][0] < -0.25, action

    # Regression: never start charging a laser while escaping a close hunter.
    laser_obs = copy.deepcopy(BASE)
    laser_obs["tick"] = 8002
    laser_obs["t"] = 400.1
    laser_obs["self"]["weapons"][1] = {
        "kind": "laser", "cooldown": 0, "charge": 0,
        "stats": {"damage": 315, "charge": 0.75, "range": 280},
    }
    laser_obs["mechs"] = [
        {"id": 7, "kind": "mech", "pos": [laser_obs["self"]["pos"][0] + 20, laser_obs["self"]["pos"][1], laser_obs["self"]["pos"][2]],
         "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1400, "max_hp": 1400,
         "cubes": 0, "weapons": ["gun", "lightning"], "charging": False}
    ]
    action = bot.act(laser_obs)
    assert not action.get("fire", [False, False])[1], action

    # Regression: qualification opening movement stays below the 5 m/s sound
    # threshold and does not reveal us by shooting a non-contact opponent.
    stealth_bot = Bot(HELLO)
    stealth_obs = copy.deepcopy(BASE)
    stealth_obs["mechs"] = []
    stealth_action = stealth_bot.act(stealth_obs)
    assert math.hypot(*stealth_action["move"]) <= 0.390001, stealth_action
    assert stealth_bot.rotation_deadline > stealth_obs["t"] + 30.0

    seen_obs = copy.deepcopy(stealth_obs)
    seen_obs["tick"] = 1
    seen_obs["t"] = 0.05
    seen_obs["mechs"] = [
        {"id": 7, "kind": "mech", "pos": [stealth_obs["self"]["pos"][0] + 85, stealth_obs["self"]["pos"][1], stealth_obs["self"]["pos"][2]],
         "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1400, "max_hp": 1400,
         "cubes": 0, "weapons": ["gun"], "charging": False}
    ]
    seen_action = stealth_bot.act(seen_obs)
    assert not any(seen_action.get("fire", [False, False])), seen_action

    # Once centered under the chosen canopy, hold completely still until a
    # contact or zone deadline forces a move.
    settled_obs = copy.deepcopy(BASE)
    settled_obs["tick"] = 2
    settled_obs["t"] = 0.1
    settled_obs["mechs"] = []
    settled_obs["sounds"] = []
    settled_obs["self"]["pos"][:2] = [stealth_bot.hide_goal.x, stealth_bot.hide_goal.y]
    settled_action = stealth_bot.act(settled_obs)
    assert settled_action["move"] == [0.0, 0.0], settled_action

    # Crossing the computed rotation deadline immediately enables full-speed
    # travel; there is no second grace period that can leak hundreds of HP.
    rotate_obs = copy.deepcopy(BASE)
    rotate_obs["tick"] = 1800
    rotate_obs["t"] = stealth_bot.rotation_deadline + 0.1
    rotate_obs["self"]["energy"] = rotate_obs["self"]["max_energy"]
    rotate_obs["mechs"] = []
    rotate_obs["sounds"] = []
    rotate_action = stealth_bot.act(rotate_obs)
    assert rotate_action.get("boost") is True, rotate_action

    # Zone damage is not hostile contact and must not trigger evasive jumping.
    zone_bot = Bot(HELLO)
    zone_obs = copy.deepcopy(BASE)
    zone_obs["tick"] = 3100
    zone_obs["t"] = 155.0
    zone_obs["self"]["hp"] -= 10
    zone_obs["self"]["in_zone"] = False
    zone_obs["events"] = [{"type": "damaged", "amount": 10, "weapon": "zone", "bearing": None, "by": None}]
    zone_action = zone_bot.act(zone_obs)
    assert zone_action.get("jump") is False, zone_action

    print(f"PASS: 3000 adversarial observations + state regressions in {elapsed:.3f}s ({elapsed / 3:.3f} ms/decision)")


if __name__ == "__main__":
    main()
