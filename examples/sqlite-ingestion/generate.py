#!/usr/bin/env python3
"""Generate a deterministic CSV event stream."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import random


CATEGORIES = ("search", "purchase", "login", "view", "download", "share")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.records <= 0:
        parser.error("--records must be positive")
    generator = random.Random((args.seed << 32) ^ args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event_id", "user_id", "category", "amount", "payload"))
        for event_id in range(args.records):
            user_id = generator.randrange(max(100, args.records // 20))
            category = generator.choice(CATEGORIES)
            amount = round(generator.lognormvariate(2.0, 1.0), 2)
            payload = hashlib.blake2s(
                f"{args.seed}:{event_id}".encode(), digest_size=12
            ).hexdigest()
            writer.writerow((event_id, user_id, category, amount, payload))
    temporary.replace(args.output)


if __name__ == "__main__":
    main()

