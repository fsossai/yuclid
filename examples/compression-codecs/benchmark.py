#!/usr/bin/env python3
"""Time one stdlib compression/decompression round trip."""

from __future__ import annotations

import argparse
import bz2
import gzip
import lzma
from pathlib import Path
import resource
import sys
import time


def peak_rss_kib() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value // 1024 if sys.platform == "darwin" else value


def codec_functions(name: str, level: int):
    if name == "gzip":
        return (
            lambda value: gzip.compress(value, compresslevel=level, mtime=0),
            gzip.decompress,
        )
    if name == "bz2":
        return lambda value: bz2.compress(value, compresslevel=level), bz2.decompress
    if name == "lzma":
        return lambda value: lzma.compress(value, preset=level), lzma.decompress
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--codec", choices=("gzip", "bz2", "lzma"), required=True)
    parser.add_argument("--level", type=int, choices=range(10), required=True)
    args = parser.parse_args()
    original = args.input.read_bytes()
    compress, decompress = codec_functions(args.codec, args.level)

    start = time.perf_counter()
    compressed = compress(original)
    compression_seconds = time.perf_counter() - start
    start = time.perf_counter()
    restored = decompress(compressed)
    decompression_seconds = time.perf_counter() - start

    ratio = len(compressed) / len(original)
    print(
        f"{compression_seconds:.9f} {decompression_seconds:.9f} "
        f"{ratio:.9f} {len(compressed)} {peak_rss_kib()} "
        f"{int(restored == original)}"
    )


if __name__ == "__main__":
    main()
