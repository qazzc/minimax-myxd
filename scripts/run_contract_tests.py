#!/usr/bin/env python3
"""Run lightweight repository contract tests without requiring pytest."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    vendor_source = root / "ninetoothed" / "vendor" / "src"
    sys.path.insert(0, str(vendor_source))
    test_files = sorted((root / "tests").glob("test_*.py"))
    count = 0
    skipped = 0
    failures = []
    for path in test_files:
        try:
            namespace = runpy.run_path(str(path))
        except ImportError as exc:
            skipped += 1
            print(f"SKIP {path.name}: optional test dependencies missing: {exc}")
            continue
        for name, test in sorted(namespace.items()):
            if not name.startswith("test_") or not callable(test):
                continue
            try:
                test()
            except Exception as exc:
                failures.append((path.name, name, exc))
                print(f"FAIL {path.name}::{name}: {exc}")
            else:
                count += 1
                print(f"PASS {path.name}::{name}")
    if failures:
        print(f"{len(failures)} failed; {count} passed")
        return 1
    print(f"{count} tests passed; {skipped} optional test modules skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
