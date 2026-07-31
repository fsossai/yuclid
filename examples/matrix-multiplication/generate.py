#!/usr/bin/env python3
"""Generate two reproducible square matrices in one native-double file."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import random
import struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")

    # A size-dependent stream means generating sizes in another order does not
    # change any file.
    generator = random.Random((args.seed << 32) ^ args.size)
    values = array(
        "d",
        (generator.uniform(-1.0, 1.0) for _ in range(2 * args.size * args.size)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("=Q", args.size))
        values.tofile(stream)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()

