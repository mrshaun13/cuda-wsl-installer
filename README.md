# CUDA WSL Benchmark Installer

**[View the Community Leaderboard](results/LEADERBOARD.md)** - Run benchmarks, compare scores, and contribute via PRs.

Automated CUDA installation scripts for Windows 11 developers using WSL2 with integrated benchmarking and leaderboards.

## Quick Start (1-Click Install)

From a fresh Ubuntu WSL environment:

```bash
# Clone and run installer
git clone https://github.com/<your-org>/cuda-wsl-installer.git
cd cuda-wsl-installer
./install.sh
```

That's it! The installer will:
- Detect your GPU and compute capability
- Install the correct CUDA version
- Set up a Python virtual environment
- Install PyTorch, TensorFlow, cuDF
- Run benchmarks with GPU/CPU fallback
- Generate a leaderboard

For preview without changes: `./install.sh --dry-run`

## Why this repo?

Upgrading between CUDA major versions on WSL often requires uninstalling and
reinstalling multiple packages, repos, and samples. These scripts encapsulate
known-good flows for two scenarios:

| Hardware profile                          | Compute capability | CUDA track | PyTorch | TensorFlow |
|------------------------------------------|--------------------|------------|---------|------------|
| Pascal / early Turing (e.g., GTX 1080 Ti) | ≤ 7.x              | 11.0       | cu118   | CPU-only   |
| Ampere, Ada, Blackwell (e.g., RTX 5070)   | ≥ 8.x              | 13.0       | cu124   | GPU-enabled|

The script auto-detects the compute capability via `nvidia-smi` and automatically:
- **For legacy GPUs (Pascal/Turing)**: Installs CUDA 11.0 via runfile (Ubuntu 24.04 compatible), pins PyTorch to cu118 wheels, and uses TensorFlow CPU
- **For modern GPUs (Ampere+)**: Installs CUDA 13.0 via apt, uses PyTorch cu124 wheels, and enables TensorFlow GPU

You can also force a track via CLI flags when testing.

## What is CUDA & why run it inside WSL2?

[CUDA](https://developer.nvidia.com/cuda-zone) is NVIDIA’s parallel computing
platform that exposes the GPU for general-purpose workloads. When combined with
WSL2 you can:

1. **Prototype in Linux without leaving Windows** – run the exact CLI tooling,
   package managers, and build chains that your Linux servers use.
2. **Accelerate AI/ML frameworks** – libraries like [PyTorch](https://pytorch.org/),
   [TensorFlow](https://www.tensorflow.org/install/pip#windows_setup), and
   [JAX](https://jax.readthedocs.io/en/latest/) automatically leverage CUDA for
   GPU-backed training/inference.
3. **Speed up data engineering & analytics** – [RAPIDS](https://rapids.ai/)
   provides GPU-accelerated pandas/cuDF, cuML, and cuGraph pipelines.
4. **Enable simulation, rendering, and HPC codes** – e.g., [Blender Cycles](https://www.blender.org/),
   [LAMMPS](https://www.lammps.org/), or in-house CUDA kernels.
5. **Run modern GenAI tooling locally** – Stable Diffusion pipelines, LLM
   fine-tuning, and other CUDA-dependent projects work seamlessly when WSL has
   GPU access.
6. **Scientific computing and HPC** – NumPy with CuPy for GPU arrays, JAX for
   composable transformations, OpenCV with CUDA for computer vision.
7. **3D rendering and visualization** – Blender Cycles, ParaView, VTK with GPU
   acceleration.
8. **Molecular dynamics and simulations** – LAMMPS, GROMACS, NAMD with CUDA
   for faster physics simulations.
9. **Custom GPU kernels** – Numba CUDA for writing custom CUDA code in Python,
   or direct CUDA C++ development.

Installing CUDA in WSL means your Windows laptops/desktops act like Linux CUDA
workstations without dual-booting, while still sharing the same NVIDIA driver
stack maintained on Windows.

## Requirements

* Windows 11 with WSL2 (Ubuntu 22.04/24.04) already configured
* Latest NVIDIA Windows driver with WSL GPU support
* WSL distro must have `sudo` privileges and network access

## Verification

After the script finishes, you can re-run the sample check at any time:

```bash
/usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery
```

A final line of `Result = PASS` confirms that CUDA sees your GPU from WSL.

### Sample output (GTX 1080 Ti on CUDA 11.0 track)

Below is the expected `deviceQuery` output on a Pascal card that routes to the
11.0 toolchain. Use it as a reference to confirm your installation matches:

```
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce GTX 1080 Ti"
  CUDA Driver Version / Runtime Version          13.0 / 11.0
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 11264 MBytes (11811028992 bytes)
  (028) Multiprocessors, (128) CUDA Cores/MP:    3584 CUDA Cores
  GPU Max Clock rate:                            1582 MHz (1.58 GHz)
  Memory Clock rate:                             5505 Mhz
  Memory Bus Width:                              352-bit
  L2 Cache Size:                                 2883584 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        98304 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 13.0, CUDA Runtime Version = 12.5, NumDevs = 1
Result = PASS
```

## Advanced Usage

### Manual Component Installation

If you need to run components separately:

```bash
# Install CUDA only
python3 scripts/cuda_install.py

# Setup environment only
python3 scripts/env_setup.py --venv-path .my-venv --gpu

# Run benchmarks only
python3 scripts/benchmark_runner.py --gpu --venv-python .my-venv/bin/python3
```

### Environment Variables

- `XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false`: Fixes TensorFlow cuDNN issues on older GPUs
- `TF_CPP_MIN_LOG_LEVEL=3`: Suppresses TensorFlow warnings

### Benchmark Details

- **PyTorch MatMul**: 2048x2048 matrix multiplication, 10 runs average
- **TensorFlow CNN**: MNIST CNN training for 1 epoch
- **cuDF GroupBy**: 1M row DataFrame groupby operation

All benchmarks include GPU/CPU fallback and leaderboard integration.

## Running Benchmarks and Leaderboard

To participate in the community CUDA WSL benchmarks and contribute to the gamified leaderboard, follow these steps. The leaderboard tracks performance across different hardware setups for fun comparison and optimization insights.

### Prerequisites
- CUDA installed via `./install.sh`
- Python virtual environment with PyTorch and TensorFlow (created automatically)
- Git configured with your GitHub handle (`git config user.name "YourGitHubUsername"`)
- **System verification:** Run `nvidia-smi` to ensure GPU is detected

### Important Notes
- **GPU/CPU Fallback:** Benchmarks attempt GPU first, but fall back to CPU if CUDA fails
- **Device Detection:** The leaderboard shows the actual device used (GPU or CPU)
- **GPU Compatibility:** Modern libraries may not support older GPUs; CPU fallback ensures functionality

### Running Benchmarks
You can run individual benchmarks or all at once.

**Option 1: Run All Benchmarks (Recommended)**
```bash
source .cuda-wsl-bench-venv/bin/activate
python3 scripts/benchmark_runner.py --gpu
```
This runs PyTorch, TensorFlow, cuDF benchmarks with GPU/CPU fallback and updates leaderboards.

**Option 2: Run Individual Benchmarks**

1. **PyTorch matrix multiplication:**
   ```bash
   source .cuda-wsl-bench-venv/bin/activate
   python3 scripts/benchmarks/run_pytorch_matmul.py --device cuda  # or cpu
   ```

2. **TensorFlow CNN:**
   ```bash
   source .cuda-wsl-bench-venv/bin/activate
   python3 scripts/benchmarks/run_tensorflow_cnn.py --device cuda  # or cpu
   ```

3. **cuDF groupby:**
   ```bash
   source .cuda-wsl-bench-venv/bin/activate
   python3 scripts/benchmarks/run_cudf_groupby.py --device cuda  # or cpu
   ```

Each run captures specs, updates `results/hacker_leaderboard_*.json`, and shows top scores.

### Leaderboard Details
- **Scoring:** Lower times = better
- **Hardware capture:** CPU, GPU, OS, CUDA, driver versions
- **Community sharing:** Submit PRs with results

**View the live leaderboard:** [results/LEADERBOARD.md](results/LEADERBOARD.md)

* **`nvidia-smi` missing:** Install/repair the NVIDIA Windows driver, then restart WSL (`wsl --shutdown`).
* **CUDA installation fails:** Check internet, sudo privileges, remove conflicts with `sudo apt-get remove cuda* nvidia*`.
* **Benchmark failures:** PyTorch works broadly; TensorFlow/cuDF may fail on old GPUs, CPU fallback used.
* **Virtual environment issues:** Delete and recreate: `rm -rf .cuda-wsl-bench-venv && python3 scripts/env_setup.py`.

## Known Issues & Solutions

### WSL CUDA Shim Segfaults (RESOLVED ✅)

**Original Problem**: `libcuda.so.1 --version` exits with code 139 (segfault) on Ubuntu 24.04 with Pascal/Turing GPUs.

**Root Cause**: Not a WSL/driver bug, but a **package compatibility issue**:
- Ubuntu 24.04 removed CUDA 11.x apt packages
- Attempting to install CUDA 12.x/13.x on Pascal GPUs (compute capability ≤ 7) causes shim failures
- Modern TensorFlow/PyTorch versions expect different CUDA versions than what legacy GPUs support

**Solution** (implemented in this installer):
1. **Use CUDA 11.0 runfile installer** for Pascal/Turing GPUs on Ubuntu 24.04
2. **Pin PyTorch to cu118 wheels** for optimal legacy GPU support
3. **Use TensorFlow CPU** (modern versions dropped Pascal support)
4. **Export CUDA paths** before package installation

**Result**: 4/4 benchmarks passing, no segfaults, full GPU acceleration where supported.

**For others experiencing similar issues**: See [microsoft/WSL#13773](https://github.com/microsoft/WSL/issues/13773) for the original report. The workaround implemented in this installer resolves the issue without requiring WSL/driver updates.

## Advanced Usage

### Manual Component Installation

If you need to run components separately:

```bash
# Install CUDA only
python3 scripts/cuda_install.py

# Setup environment only
python3 scripts/env_setup.py --venv-path .my-venv --gpu

# Run benchmarks only
python3 scripts/benchmark_runner.py --gpu --venv-python .my-venv/bin/python3
```

### Environment Variables

- `XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false`: Fixes TensorFlow cuDNN issues on older GPUs
- `TF_CPP_MIN_LOG_LEVEL=3`: Suppresses TensorFlow warnings

### Benchmark Details

- **PyTorch MatMul**: 2048x2048 matrix multiplication, 10 runs average
- **TensorFlow CNN**: MNIST CNN training for 1 epoch
- **cuDF GroupBy**: 1M row DataFrame groupby operation
- **CUDA Samples**: deviceQuery, matrixMul, nbody simulation (works on all GPUs)

All benchmarks include GPU/CPU fallback and leaderboard integration.

## API Documentation

### cuda_install.py

Handles CUDA toolkit installation based on GPU detection.

```python
from scripts.cuda_install import detect_gpu, install_cuda

gpu_available, compute_cap = detect_gpu()
if gpu_available:
    install_cuda()  # Installs appropriate CUDA version
```

### env_setup.py

Manages Python virtual environment and package installation.

```python
from scripts.env_setup import setup_venv, install_packages

venv_path = setup_venv('.venv')
install_packages(use_gpu=True)  # Installs PyTorch, TF, cuDF, etc.
```

### benchmark_runner.py

Orchestrates benchmark execution with error handling.

```python
from scripts.benchmark_runner import run_all_benchmarks

results = run_all_benchmarks(use_gpu=True)
# Returns dict with success status for each benchmark
```

## Legacy GPU Support (Pascal/Turing)

### CUDA 11.0 Installation on Ubuntu 24.04

Ubuntu 24.04 (Noble) does not have `cuda-toolkit-11.0` packages in the apt repository. The installer automatically uses the **CUDA 11.0 runfile installer** for legacy GPUs:

```bash
# Automatically handled by the installer
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sudo sh cuda_11.0.2_450.51.05_linux.run --silent --toolkit --override
```

The 2.9GB download includes a progress bar and takes 2-5 minutes depending on connection speed.

### PyTorch cu118 Wheels

For compute capability ≤ 7 (Pascal/Turing GPUs like GTX 1080 Ti), the installer pins PyTorch to **cu118 wheels**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This ensures full GPU acceleration for PyTorch workloads on legacy hardware.

### TensorFlow CPU-Only on Legacy GPUs

**Why CPU-only?**
- TensorFlow 2.16+ dropped support for compute capability < 8.0
- TensorFlow 2.10 (last GPU-supporting version for Pascal) is no longer available on PyPI
- TensorFlow 2.10 requires Python 3.7-3.10 (incompatible with modern Python 3.12)
- CPU performance is acceptable for benchmarking purposes (~5-6 seconds for standard CNN training)

**What you'll see:**
```
[INFO] TensorFlow GPU build skipped for legacy GPU (compute capability < 8)
TensorFlow: 2.20.0
GPU devices: 0
```

This is **expected behavior** and ensures you get modern, secure TensorFlow without downgrading Python.

### Benchmark Expectations on GTX 1080 Ti

- ✅ **PyTorch MatMul**: Full GPU acceleration (~0.002s GPU vs ~0.5s CPU)
- ✅ **TensorFlow CNN**: CPU-only (~5-6s for 3 epochs)
- ✅ **cuDF GroupBy**: GPU acceleration via RAPIDS
- ✅ **CUDA Samples**: GPU acceleration via Numba CUDA

**Result: 4/4 benchmarks pass successfully**

## Troubleshooting

### Common Issues

**"nvidia-smi not found" or "Driver/library version mismatch"**
- **This is the most common issue** with WSL GPU setup
- **Solution:**
  1. In Windows, update NVIDIA drivers to the latest version (GeForce Experience → Drivers)
  2. For GTX 1080 Ti: Ensure driver version 470.x or newer
  3. Restart WSL completely: `wsl --shutdown` then reopen WSL
  4. If still failing: `wsl --update --rollback` to previous WSL kernel
  5. Verify in Windows Device Manager that GPU shows under "Display adapters"
- The installer will automatically detect this and fall back to CPU-only mode with clear error messages

**CUDA installation fails**
- Check internet connection
- Ensure sudo privileges
- Remove conflicting packages: `sudo apt-get remove cuda* nvidia*`
- The installer handles this gracefully and continues with CPU benchmarks

**Ubuntu 24.04 + Pascal/Turing GPU specific issues**
- **"cuda-toolkit-11.0 package not found"**: Expected behavior - Ubuntu 24.04 doesn't have CUDA 11 apt packages. The installer automatically uses the runfile installer instead.
- **"nvcc --version unavailable"**: The installer now exports CUDA paths before package installation. If you still see this warning, manually run: `export PATH=/usr/local/cuda/bin:$PATH`
- **"Downloading CUDA 11.0 runfile installer..." appears hung**: The 2.9GB download shows a progress bar. On slow connections, this can take 5-10 minutes. Be patient!
- **PyTorch not using GPU**: Verify cu118 wheels were installed: `pip show torch | grep cu118`. If not, reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Benchmark failures**
- PyTorch: Usually works on all CUDA versions if GPU drivers are correct
- TensorFlow: May fail on older GPUs (Pascal sm_61); automatically falls back to CPU
- cuDF: Requires Ampere+ GPUs; falls back to pandas CPU
- CUDA Samples: Uses Numba CUDA - works on all NVIDIA GPUs with proper drivers

**Virtual environment issues**
- Delete and recreate: `rm -rf .cuda-wsl-bench-venv && python3 scripts/env_setup.py`
- Ensure Python 3.8+ is available

### Debug Mode

Run with verbose logging:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -c "import logging; logging.basicConfig(level=logging.DEBUG); import scripts.cuda_install; scripts.cuda_install.main()"
```

### Logs

Check `install.log` for detailed execution logs.

## Development

### Adding New Benchmarks

1. Create script in `scripts/benchmarks/`
2. Add leaderboard integration (see existing scripts)
3. Update `benchmark_runner.py` to include new benchmark
4. Test with both GPU and CPU

### Code Structure

```
├── install.sh              # Main installer script
├── scripts/
│   ├── cuda_install.py     # CUDA detection/installation
│   ├── env_setup.py        # Environment setup
│   └── benchmark_runner.py # Benchmark orchestration
├── scripts/benchmarks/     # Individual benchmark scripts
├── results/                # Leaderboards and outputs
└── tests/                  # Unit tests
```

### Testing

**Why CI?** The GitHub Actions pipeline ensures code quality and prevents regressions by automatically testing every change. It validates that the installer works across different environments and catches issues before they reach users.

**CI Jobs:**
- **test-install**: Validates script functionality and dry-runs
- **test-benchmarks**: Runs benchmark tests on CPU
- **validate-docs**: Checks documentation completeness

**Local Testing:**
Run unit tests:
```bash
python3 -m pytest tests/
```

**CI on GitHub:** Automatically runs on pushes and pull requests. Your GitHub repository will execute these checks for all contributors.

Run CI locally (requires act):
```bash
act -j test-install
act -j test-benchmarks
```

## Performance Expectations

### GTX 1080 Ti (Compute Capability 6.1) - CUDA 11.0 Track

**Configuration:**
- CUDA: 11.0 (runfile installer)
- PyTorch: 2.7.1+cu118 (GPU-enabled)
- TensorFlow: 2.20.0 (CPU-only)
- cuDF: 25.10.00 (GPU-enabled)

**Benchmark Results:**
- **PyTorch MatMul**: ~0.002s (GPU) - 250x faster than CPU (~0.5s)
- **TensorFlow CNN**: ~5.6s (CPU-only, 3 epochs) - Expected behavior for legacy GPUs
- **cuDF GroupBy**: GPU-accelerated via RAPIDS
- **CUDA Samples**: GPU-accelerated via Numba CUDA

**Overall: 4/4 benchmarks passing ✅**

Results vary by hardware. Submit PRs to add your scores!

## License

MIT License - see LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure CI passes
5. Submit a pull request

Contributions welcome.
