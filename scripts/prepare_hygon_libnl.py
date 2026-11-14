#!/usr/bin/env python3
"""Keep Hygon Torch and system RDMA users on one libnl implementation.

The vendor wheel gives its libnl DSOs hashed SONAMEs. A system RDMA provider
can consequently load a second copy whose global symbols interpose with the
first, corrupting the traffic-control registry at ELF shutdown. Private search
aliases resolve the ordinary SONAMEs to the *same files* as the wheel, without
modifying the wheel, changing attention kernels, or disabling RDMA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys


ALIASES = {
    "libnl-3.so.200": "libnl-3-*.so.200*",
    "libnl-route-3.so.200": "libnl-route-3-*.so.200*",
}


def bundled_libnl() -> dict[str, Path]:
    if sys.platform != "linux":
        return {}
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return {}
    libdir = Path(spec.origin).parent.parent / "torch.libs"
    matches = {alias: sorted(libdir.glob(pattern)) for alias, pattern in ALIASES.items()}
    if not any(matches.values()):
        return {}
    if any(len(paths) != 1 for paths in matches.values()):
        raise RuntimeError(f"Expected one complete vendor libnl pair under {libdir}: {matches}")
    return {alias: paths[0].resolve(strict=True) for alias, paths in matches.items()}


def prepare_aliases(cache_dir: Path, libraries: dict[str, Path]) -> Path | None:
    if not libraries:
        return None
    if set(libraries) != set(ALIASES):
        raise ValueError("Both libnl core and route libraries are required")
    targets = {alias: path.resolve(strict=True) for alias, path in libraries.items()}
    identity = "\n".join(f"{alias}={path}" for alias, path in sorted(targets.items()))
    directory = cache_dir / hashlib.sha256(identity.encode()).hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    for alias, target in targets.items():
        link = directory / alias
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Refusing to replace unexpected compatibility entry: {link}")
        temporary = directory / f".{alias}.{os.getpid()}.tmp"
        try:
            temporary.symlink_to(target)
            os.replace(temporary, link)
        finally:
            temporary.unlink(missing_ok=True)
    return directory.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = prepare_aliases(args.cache_dir, bundled_libnl())
    if directory is not None:
        print(directory)


if __name__ == "__main__":
    main()
