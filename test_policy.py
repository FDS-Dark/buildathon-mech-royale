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
    print(f"PASS: 3000 adversarial observations in {elapsed:.3f}s ({elapsed / 3:.3f} ms/decision)")


if __name__ == "__main__":
    main()
