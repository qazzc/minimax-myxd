#!/usr/bin/env python3
import importlib.metadata as md
import os
import pathlib
import sys

import torch
import ninetoothed

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), getattr(p, "gcnArchName", None), round(p.total_memory / 1024**3, 1), "GiB")
for name in ("infinicore", "infinilm", "diffusers", "transformers", "accelerate", "flash-attn", "triton", "torchvision"):
    try:
        print(name + ":", md.version(name))
    except md.PackageNotFoundError:
        print(name + ": MISSING")
print("HYGON_ARCH:", os.environ.get("HYGON_ARCH", ""))
print("INFINI_ROOT:", os.environ.get("INFINI_ROOT", ""))
print("ninetoothed_source:", ninetoothed.__file__)
if os.environ.get("HYGON_ARCH") == "gfx936":
    import triton

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936" or target.warp_size != 64:
        raise SystemExit(f"Unexpected gfx936 Triton target: {target}")
    import ninetoothed.aot as ninetoothed_aot

    header = pathlib.Path(ninetoothed_aot._HEADER_PATH)
    print("ninetoothed_hygon_aot:", target.backend, target.arch, target.warp_size)
    print("ninetoothed_aot_header:", header)
try:
    import torchvision
    print("torchvision_nms:", hasattr(torch.ops.torchvision, "nms"))
except Exception as e:
    print("torchvision_nms: FAIL", repr(e))
try:
    import flash_attn
    print("flash_attn_import: OK")
except Exception as e:
    print("flash_attn_import: FAIL", repr(e))
