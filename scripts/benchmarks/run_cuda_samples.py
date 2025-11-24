#!/usr/bin/env python3
"""CUDA Samples benchmark for WSL."""

import subprocess
import sys
import time
from pathlib import Path

def run_cuda_sample(sample_name, args=None):
    """Run a CUDA sample and measure time."""
    cuda_samples_path = "/usr/local/cuda/samples/bin/x86_64/linux/release"
    sample_path = f"{cuda_samples_path}/{sample_name}"
    
    if not Path(sample_path).exists():
        raise FileNotFoundError(f"CUDA sample {sample_name} not found. Run install_cuda.sh with samples.")
    
    cmd = [sample_path]
    if args:
        cmd.extend(args)
    
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - start
    
    if result.returncode != 0:
        raise RuntimeError(f"CUDA sample failed: {result.stderr}")
    
    return duration

def benchmark_device_query():
    """Benchmark deviceQuery sample."""
    return run_cuda_sample("deviceQuery")

def benchmark_matrix_mul():
    """Benchmark matrix multiplication sample."""
    return run_cuda_sample("matrixMul")

def benchmark_nbody():
    """Benchmark N-body simulation sample."""
    return run_cuda_sample("nbody", ["-benchmark", "-numbodies=15360"])

def main():
    """Run CUDA samples benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    
    results = {}
    
    try:
        results["device_query"] = benchmark_device_query()
        print(f"[benchmark] cuda_device_query time={results['device_query']:.3f}s")
    except Exception as e:
        print(f"[warning] deviceQuery failed: {e}")
        results["device_query"] = None
    
    try:
        results["matrix_mul"] = benchmark_matrix_mul()
        print(f"[benchmark] cuda_matrix_mul time={results['matrix_mul']:.3f}s")
    except Exception as e:
        print(f"[warning] matrixMul failed: {e}")
        results["matrix_mul"] = None
    
    try:
        results["nbody"] = benchmark_nbody()
        print(f"[benchmark] cuda_nbody time={results['nbody']:.3f}s")
    except Exception as e:
        print(f"[warning] nbody failed: {e}")
        results["nbody"] = None
    
    if args.result_file:
        import json
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
