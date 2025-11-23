# CPU Baseline Scores

These JSON snapshots were produced while the WSL libcuda shim is broken
([microsoft/WSL#13773](https://github.com/microsoft/WSL/issues/13773)). They
serve as the “before” reference so others can compare CPU-only runs or provide
the missing GPU data if their environment works.

Baseline hardware:

- Windows host CPU: Intel Core i7-7700K (4 cores / 8 threads) exposed to WSL
- Distro: Ubuntu 24.04 on WSL2, kernel 6.6.87.2, NVIDIA driver 581.57

| Benchmark            | Device | Config                    | Avg / Duration |
|----------------------|--------|---------------------------|----------------|
| PyTorch matmul       | CPU    | size=2048, repeats=3      | 0.054 s avg    |
| TensorFlow MNIST CNN | CPU    | epochs=1, batch=256       | 10.25 s total  |

Once the shim issue is resolved we’ll add matching GPU JSON results under
`results/gpu-after/` and let the leaderboard compare “before vs after.”
