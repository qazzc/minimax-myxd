#!/usr/bin/env python3
"""Run the preserved PyTorch reference with memory-bounded PyTorch SDPA."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
INFINILM_ROOT = SUBMISSION_ROOT / "InfiniLM"
GENERATOR = INFINILM_ROOT / "examples" / "minimax_h3.py"


def torch_chunked_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Evaluate exact dense attention in query chunks using ordinary PyTorch ops."""
    if enable_gqa and query.shape[-3] != key.shape[-3]:
        repeats = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(repeats, dim=-3)
        value = value.repeat_interleave(repeats, dim=-3)

    query_length, key_length, head_dim = (
        query.shape[-2],
        key.shape[-2],
        query.shape[-1],
    )
    scale = head_dim**-0.5 if scale is None else scale
    query_fp32, key_fp32, value_fp32 = query.float(), key.float(), value.float()
    outputs = []
    chunk_size = _CHUNK_SIZE

    for start in range(0, query_length, chunk_size):
        end = min(start + chunk_size, query_length)
        scores = torch.matmul(
            query_fp32[..., start:end, :], key_fp32.transpose(-2, -1)
        ) * scale

        if is_causal:
            query_indices = torch.arange(start, end, device=query.device).unsqueeze(-1)
            key_indices = torch.arange(key_length, device=query.device).unsqueeze(0)
            scores.masked_fill_(key_indices > query_indices, float("-inf"))

        if attn_mask is not None:
            mask = attn_mask
            if mask.ndim >= 2 and mask.shape[-2] == query_length:
                mask = mask[..., start:end, :]
            if mask.dtype == torch.bool:
                scores.masked_fill_(~mask, float("-inf"))
            else:
                scores = scores + mask.float()

        probabilities = torch.softmax(scores, dim=-1).nan_to_num_(0.0)
        if dropout_p:
            probabilities = F.dropout(probabilities, p=dropout_p, training=True)
        outputs.append(torch.matmul(probabilities, value_fp32[..., :key_length, :]))

    return torch.cat(outputs, dim=-2).to(query.dtype)


_CHUNK_SIZE = 1024


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--chunk-size", type=int, default=1024)
    options, generator_args = parser.parse_known_args()
    if options.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if "--legacy" not in generator_args:
        parser.error("the PyTorch reference command must include --legacy")
    if not generator_args:
        parser.error("pass arguments for examples/minimax_h3.py")

    global _CHUNK_SIZE
    _CHUNK_SIZE = options.chunk_size
    sys.path.insert(0, str(INFINILM_ROOT / "python"))
    sys.path.insert(0, str(SUBMISSION_ROOT / "InfiniCore" / "python"))
    F.scaled_dot_product_attention = torch_chunked_sdpa
    torch.nn.functional.scaled_dot_product_attention = torch_chunked_sdpa
    sys.argv = [str(GENERATOR), *generator_args]
    runpy.run_path(str(GENERATOR), run_name="__main__")


if __name__ == "__main__":
    main()
