#!/usr/bin/env python3
"""Check FL2VA container readability and required video/audio dimensions.

This validates media structure only. It does not establish semantic similarity
against the official reference; use minimax_h3_compare.py for that evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--require-audio", action="store_true")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")

    try:
        import torch
        from torchvision.io import read_video
    except Exception as exc:
        print(f"FAIL: cannot import torchvision video decoder: {exc}", file=sys.stderr)
        return 2

    try:
        frames, audio, info = read_video(str(args.video), pts_unit="sec")
    except Exception as exc:
        print(f"FAIL: cannot decode video: {exc}", file=sys.stderr)
        return 2

    failures = []
    actual = tuple(frames.shape)
    expected = (args.frames, args.height, args.width, 3)
    print(f"decoded_shape: {actual}")
    print(f"video_fps: {info.get('video_fps')}")
    print(f"audio_shape: {tuple(audio.shape)}")

    if actual != expected:
        failures.append(f"expected decoded shape {expected}, got {actual}")
    actual_fps = float(info.get("video_fps", 0.0))
    if abs(actual_fps - args.fps) > 0.01:
        failures.append(f"expected {args.fps} FPS, got {actual_fps}")
    if args.require_audio and audio.numel() == 0:
        failures.append("audio stream is required but missing")
    if frames.numel() == 0 or not torch.isfinite(frames.float()).all():
        failures.append("video has no frames or contains non-finite values")
    if frames.float().std().item() < 1.0:
        failures.append("video appears near-constant/black")

    if failures:
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1
    print("PASS: readable video and required media dimensions")
    print("NOTE: semantic quality and first-frame/prompt adherence require visual review and official reference comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
