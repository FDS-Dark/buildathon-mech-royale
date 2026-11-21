# Roof Rat

A survival-first bot for Builders Mech Royale. It uses only the Python standard
library, reads the mounted arena map at startup, and prioritizes placement by
seeking accessible roofs, staying ahead of the next zone, disengaging from the
coordinated practice squad, and firing while retreating.

Local conformance check from the official kit directory:

```sh
python check/check.py -- python -u ../bot/bot.py
docker build -t roof-rat ../bot
python check/check.py --map-dir /arena -- docker run --rm -i --network none \
  --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m \
  --user 65534:65534 -v "$PWD/arena:/arena:ro" roof-rat
```
