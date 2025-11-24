#!/usr/bin/env python3
"""Generate a markdown leaderboard for GitHub display."""

import json
import os

def generate_markdown_leaderboard(scores):
    header = """# CUDA WSL Hacker Leaderboard 🕹️

**Scoring: Lower times = BETTER!** (CUDA vs CPU battles, fastest wins!)

| Rank | Handle | Benchmark | Score | Status |
|------|--------|-----------|-------|--------|
"""
    lines = []
    for i, score in enumerate(scores[:20]):  # Top 20 for GitHub
        rank = f"{i+1}"
        handle = score.get('handle', 'Anonymous')
        benchmark = score['benchmark']
        time_score = f"{score['score']:.4f}s" if 'score' in score else 'DNF'
        status = score.get('status', 'UNKNOWN!')
        lines.append(f"| {rank} | {handle} | {benchmark} | {time_score} | {status} |")
    
    footer = """

## System Specs for Top Scores (CPU vs GPU details)
"""
    for i, score in enumerate(scores[:10]):  # Details for top 10
        rank = i+1
        handle = score.get('handle', 'Anonymous')
        benchmark = score['benchmark']
        cpu = score.get('cpu', 'Unknown CPU')
        gpu = score.get('gpu', 'Unknown GPU')
        os_ = score.get('os', 'Unknown OS')
        cuda = score.get('cuda_version', 'Unknown CUDA')
        driver = score.get('driver_version', 'Unknown Driver')
        device_type = 'GPU' if 'cuda' in benchmark.lower() else 'CPU'
        footer += f"{rank}. **{handle}** - {benchmark} ({device_type}): CPU: {cpu} | GPU: {gpu} | OS: {os_} | CUDA: {cuda} | Driver: {driver}\n\n"
    
    footer += """## Contribute Your Scores! 🚀

1. Fork this repo
2. Run benchmarks: `python scripts/benchmarks/run_pytorch_matmul.py --device cuda`
3. Your score auto-updates `results/hacker_leaderboard.json`
4. Submit a PR to add your entry!

Benchmarks: PyTorch matmul, TensorFlow CNN, RAPIDS cuDF groupby.
"""

    return header + "\n".join(lines) + footer

if __name__ == "__main__":
    leaderboard_file = os.path.join(os.path.dirname(__file__), "hacker_leaderboard.json")
    if os.path.exists(leaderboard_file):
        with open(leaderboard_file, 'r') as f:
            scores = json.load(f)
        markdown = generate_markdown_leaderboard(scores)
        with open(os.path.join(os.path.dirname(__file__), "LEADERBOARD.md"), 'w') as f:
            f.write(markdown)
        print("Generated LEADERBOARD.md")
    else:
        print("No leaderboard file found.")
