#!/usr/bin/env python3
"""Simple PyTorch matmul benchmark for CUDA WSL installer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torch matmul on CPU/GPU")
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Matrix dimension (size x size)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Which device to run on",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs before timing",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Optional JSON output path for recording results",
    )
    return parser.parse_args()


def ensure_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() == False")
    return torch.device(device)


def run_once(size: int, device: torch.device) -> float:
    mat = torch.randn((size, size), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = mat @ mat
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)

    for _ in range(args.warmup):
        run_once(args.size, device)

    timings = [run_once(args.size, device) for _ in range(args.repeats)]
    avg = sum(timings) / len(timings)

    summary = {
        "device": device.type,
        "size": args.size,
        "repeats": args.repeats,
        "average_seconds": avg,
        "samples": timings,
    }

    print(f"[benchmark] device={device.type} size={args.size} avg={avg:.4f}s")
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
