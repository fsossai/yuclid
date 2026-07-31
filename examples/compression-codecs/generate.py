#!/usr/bin/env python3
"""Generate deterministic corpora with different compressibility profiles."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random


WORDS = (
    "experiment dimension metric sample latency throughput memory cache "
    "request response database compiler vector matrix thread process "
).split()


def derived_seed(seed: int, kind: str, mebibytes: int) -> int:
    digest = hashlib.sha256(f"{seed}:{kind}:{mebibytes}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def text_chunk(generator: random.Random, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        sentence = " ".join(generator.choice(WORDS) for _ in range(14))
        result.extend((sentence + ".\n").encode())
    return bytes(result[:size])


def generate(kind: str, size: int, generator: random.Random) -> bytes:
    if kind == "text":
        return text_chunk(generator, size)
    if kind == "binary":
        return generator.randbytes(size)
    if kind == "mixed":
        result = bytearray()
        block = 64 * 1024
        while len(result) < size:
            remaining = min(block, size - len(result))
            if (len(result) // block) % 2 == 0:
                result.extend(text_chunk(generator, remaining))
            else:
                result.extend(generator.randbytes(remaining))
        return bytes(result)
    raise ValueError(f"unknown kind {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("text", "binary", "mixed"), required=True)
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mib <= 0:
        parser.error("--mib must be positive")
    generator = random.Random(derived_seed(args.seed, args.kind, args.mib))
    payload = generate(args.kind, args.mib * 1024 * 1024, generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()

