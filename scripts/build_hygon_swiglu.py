#!/usr/bin/env python3
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFINICORE = ROOT / "InfiniCore"
sys.path[:0] = [
    str(INFINICORE / "src"),
    str(ROOT / "ninetoothed" / "vendor" / "src"),
]

if os.environ.get("HYGON_ARCH") != "gfx936":
    raise SystemExit("NineToothed SwiGLU AOT generation is pinned to gfx936")

expected_target = {
    "NINETOOTHED_BACKEND": "triton",
    "NINETOOTHED_PLATFORM": "hygon-bw1000",
    "NINETOOTHED_COMPUTE_ARCH": "gfx936",
}
for name, expected in expected_target.items():
    actual = os.environ.get(name)
    if actual != expected:
        raise SystemExit(f"{name} must be {expected!r}, got {actual!r}")

import ninetoothed
from infiniop.ops.swiglu.ninetoothed import build as swiglu_build

print(f"NineToothed source: {ninetoothed.__file__}", flush=True)
swiglu_build.build()
print("Generated NineToothed SwiGLU AOT sources for gfx936", flush=True)
