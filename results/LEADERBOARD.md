# CUDA WSL Hacker Leaderboard 🕹️

**Scoring: Lower times = BETTER!** (CUDA vs CPU battles, fastest wins!)

| Rank | Handle | Benchmark | Score | Status |
|------|--------|-----------|-------|--------|
| 1 | @ShaunRocks | pytorch_matmul | 0.0300s | ELITE HACKER! |
| 2 | @ProvenGuilty | pytorch_matmul | 0.0540s | PHREAKING IT! |

## System Specs for Top Scores (CPU vs GPU details)
1. **@ShaunRocks** - pytorch_matmul (CPU): CPU: AMD Ryzen 9 5900X 12-Core Processor | GPU: NVIDIA GeForce RTX 4090 | OS: Ubuntu 22.04.3 LTS | CUDA: 12.2 | Driver: 525.60.13

2. **@ProvenGuilty** - pytorch_matmul (CPU): CPU: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 20.04.6 LTS | CUDA: 11.5 | Driver: 470.42.01

## Contribute Your Scores! 🚀

1. Fork this repo
2. Run benchmarks: `python scripts/benchmarks/run_pytorch_matmul.py --device cuda`
3. Your score auto-updates `results/hacker_leaderboard.json`
4. Submit a PR to add your entry!

Benchmarks: PyTorch matmul, TensorFlow CNN, RAPIDS cuDF groupby.
