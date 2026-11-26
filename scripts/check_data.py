#!/usr/bin/env python3
"""Validate the benchmark files used by the paper."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED = {
    Path("data/longmemeval/longmemeval_s.json"): "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    Path("data/locomo/locomo_test.json"): "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failed = False
    for path, expected in EXPECTED.items():
        if not path.exists():
            print(f"MISSING {path}")
            failed = True
            continue
        actual = sha256(path)
        ok = actual == expected
        print(f"{'OK' if ok else 'MISMATCH'} {path}")
        failed = failed or not ok
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
