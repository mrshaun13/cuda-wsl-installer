#!/usr/bin/env python3
"""CUDA kernel benchmark using Numba CUDA."""

import time
from numba import cuda
import numpy as np

def run_numba_cuda_kernel():
    """Run a simple CUDA kernel using Numba."""
    
    @cuda.jit
    def simple_kernel(arr):
        """Simple CUDA kernel that squares array elements."""
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = arr[i] * arr[i]
    
    # Create test data
    n = 1000000
    arr = np.random.random(n).astype(np.float32)
    
    # Copy to device
    d_arr = cuda.to_device(arr)
    
    # Configure kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Run kernel
    start = time.perf_counter()
    simple_kernel[blocks_per_grid, threads_per_block](d_arr)
    cuda.synchronize()  # Wait for completion
    duration = time.perf_counter() - start
    
    # Copy back result
    result = d_arr.copy_to_host()
    
    return duration

def benchmark_numba_cuda():
    """Benchmark Numba CUDA kernel."""
    return run_numba_cuda_kernel()

def main():
    """Run CUDA kernel benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()
    
    results = {}
    
    try:
        duration = benchmark_numba_cuda()
        print(f"[benchmark] cuda_kernel time={duration:.4f}s")
        results["cuda_kernel"] = duration
    except Exception as e:
        print(f"[warning] CUDA kernel benchmark failed: {e}")
        results["cuda_kernel"] = None
    
    if args.result_file:
        import json
        with open(args.result_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
