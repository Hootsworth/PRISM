#!/usr/bin/env python3
"""Verify B0 files against experiments/P2/B0_FREEZE_MANIFEST.json."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/P2/B0_FREEZE_MANIFEST.json"
EXPECTED_MANIFEST_SHA256 = "39e7aef55aa4f11d5e3f467abe93d785c9e77681d1e176debd6ca18850f9edde"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> int:
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        print("B0 freeze verification failed: manifest SHA-256 changed")
        return 1
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    failures = []
    for item in records:
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append((item["path"], "MISSING"))
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            failures.append((item["path"], "SHA256_MISMATCH"))
    if failures:
        for path, reason in failures:
            print("FAIL {} {}".format(reason, path))
        print("B0 freeze verification failed: {} issue(s)".format(len(failures)))
        return 1
    print("PASS B0 freeze verified: {} files".format(len(records)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
