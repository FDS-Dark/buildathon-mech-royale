# Ghost Canopy

A survival-first bot for Builders Mech Royale. It uses only the Python standard
library, reads the mounted arena map at startup, and specializes for the
coordinated practice squad: it finds overlapping low tree crowns, moves below
the sound threshold, computes the latest safe zone-rotation deadline, avoids
early fights, and only reveals itself to break contact or contest the endgame.

Local conformance check from the official kit directory:

```sh
python check/check.py -- python -u ../bot/bot.py
docker build -t ghost-canopy ../bot
python check/check.py --map-dir /arena -- docker run --rm -i --network none \
  --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m \
  --user 65534:65534 -v "$PWD/arena:/arena:ro" ghost-canopy
```
