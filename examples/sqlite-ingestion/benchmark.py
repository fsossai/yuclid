#!/usr/bin/env python3
"""Measure one SQLite loading strategy without third-party packages."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import resource
import sqlite3
import sys
import time


INSERT = "INSERT INTO events VALUES (?, ?, ?, ?, ?)"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader)
        for event_id, user_id, category, amount, payload in reader:
            yield int(event_id), int(user_id), category, float(amount), payload


def peak_rss_kib() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value // 1024 if sys.platform == "darwin" else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--strategy", choices=("autocommit", "transaction", "executemany"), required=True
    )
    parser.add_argument("--journal", choices=("DELETE", "WAL", "OFF"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        args.database, isolation_level=None if args.strategy == "autocommit" else ""
    )
    connection.execute(f"PRAGMA journal_mode={args.journal}")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE events("
        "event_id INTEGER, user_id INTEGER, category TEXT, "
        "amount REAL, payload TEXT)"
    )
    start = time.perf_counter()
    count = 0
    if args.strategy == "autocommit":
        for row in rows(args.input):
            connection.execute(INSERT, row)
            count += 1
    elif args.strategy == "transaction":
        with connection:
            for row in rows(args.input):
                connection.execute(INSERT, row)
                count += 1
    else:
        iterator = iter(rows(args.input))
        with connection:
            while True:
                batch = []
                for _ in range(args.batch):
                    try:
                        batch.append(next(iterator))
                    except StopIteration:
                        break
                if not batch:
                    break
                connection.executemany(INSERT, batch)
                count += len(batch)
    insert_seconds = time.perf_counter() - start

    start = time.perf_counter()
    connection.execute("CREATE INDEX category_amount ON events(category, amount)")
    connection.commit()
    index_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = connection.execute(
        "SELECT category, SUM(amount) FROM events GROUP BY category ORDER BY category"
    ).fetchall()
    query_seconds = time.perf_counter() - start
    connection.close()
    checksum = sum(value for _, value in result)
    print(
        f"{insert_seconds:.9f} {index_seconds:.9f} {query_seconds:.9f} "
        f"{count / insert_seconds:.6f} {args.database.stat().st_size} "
        f"{peak_rss_kib()} {checksum:.6f}"
    )


if __name__ == "__main__":
    main()
