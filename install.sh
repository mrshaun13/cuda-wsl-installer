#!/bin/bash

# CUDA WSL Installer - Unified Install Script
# One-click install for CUDA, PyTorch, benchmarks, and leaderboard

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.cuda-wsl-bench-venv"
CUDA_VERSION="12.5"
PYTORCH_CUDA_SUFFIX="cu124"  # For CUDA 12.4+, compatible with 12.5
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--dry-run]"
            echo "  --dry-run    Show what would be done without executing"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    log_info "DRY RUN MODE - No changes will be made"
fi

# Function to run commands (with dry-run support)
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would run: $@"
    else
        log_info "Running: $@"
        "$@"
    fi
}

# Detect GPU and CUDA capability
detect_gpu() {
    log_info "Detecting GPU and CUDA capability..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Ensure NVIDIA drivers are installed."
        USE_GPU=false
        return
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null || echo "Unknown GPU")
    log_info "GPU Info: $GPU_INFO"

    # Extract compute capability (e.g., 6.1 for GTX 1080 Ti)
    COMPUTE_CAP=$(echo "$GPU_INFO" | grep -oP '\d+\.\d+' | head -1 || echo "unknown")

    if [ "$COMPUTE_CAP" = "unknown" ]; then
        log_warning "Could not determine compute capability. Assuming CPU-only."
        USE_GPU=false
    else
        log_info "Compute capability: $COMPUTE_CAP"
        USE_GPU=true
        # Map compute capability to CUDA version
        case $COMPUTE_CAP in
            6.*|7.*|8.*)
                REQUIRED_CUDA="11.0"
                ;;
            *)
                REQUIRED_CUDA="12.0"
                ;;
        esac
        log_info "Required CUDA version for this GPU: $REQUIRED_CUDA"
    fi
}

# Check if CUDA is installed
check_cuda() {
    log_info "Checking CUDA installation..."

    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | sed -n -e 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
        log_info "CUDA version detected: $NVCC_VERSION"
        if [[ "$NVCC_VERSION" == "$REQUIRED_CUDA"* ]]; then
            log_success "CUDA $REQUIRED_CUDA already installed."
            CUDA_INSTALLED=true
        else
            log_warning "CUDA $NVCC_VERSION detected, but need $REQUIRED_CUDA for this GPU. Will install."
            CUDA_INSTALLED=false
        fi
    else
        log_warning "CUDA not detected. Will install $REQUIRED_CUDA."
        CUDA_INSTALLED=false
    fi
}

# Install CUDA
install_cuda() {
    if [ "$CUDA_INSTALLED" = true ]; then
        return
    fi

    log_info "Installing CUDA $REQUIRED_CUDA..."

    # Add NVIDIA repository
    run_cmd wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    run_cmd sudo dpkg -i cuda-keyring_1.1-1_all.deb
    run_cmd sudo apt-get update

    # Install CUDA toolkit
    run_cmd sudo apt-get install -y cuda-toolkit-$REQUIRED_CUDA

    # Add to PATH
    export PATH=/usr/local/cuda-$REQUIRED_CUDA/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-$REQUIRED_CUDA/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    log_success "CUDA $REQUIRED_CUDA installed."
}

# Setup Python virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."

    if [ ! -d "$VENV_DIR" ]; then
        run_cmd python3 -m venv "$VENV_DIR"
    else
        log_info "Virtual environment already exists."
    fi

    # Activate venv (skip in dry-run)
    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
    fi

    log_success "Virtual environment ready."
}

# Install Python packages
install_packages() {
    log_info "Installing Python packages..."

    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Install PyTorch with CUDA
    if [ "$USE_GPU" = true ]; then
        run_cmd pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA_SUFFIX
    else
        run_cmd pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install other packages
    run_cmd pip install tensorflow[and-cuda] pandas cudf-cu12 cupy-cuda12x loguru

    # Setup logging
    if [ "$DRY_RUN" != true ]; then
        python3 -c "
import loguru
from loguru import logger
logger.add('install.log', rotation='10 MB', level='INFO')
logger.info('Install script started')
"
    fi

    log_success "Python packages installed."
}

# Run benchmarks
run_benchmarks() {
    log_info "Running benchmarks..."

    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Run PyTorch benchmark
    DEVICE_PYTORCH=${USE_GPU:+cuda:0}
    DEVICE_PYTORCH=${DEVICE_PYTORCH:-cpu}
    run_cmd python3 scripts/benchmarks/run_pytorch_matmul.py --device $DEVICE_PYTORCH

    # Run TensorFlow benchmark
    DEVICE_TF=${USE_GPU:+GPU}
    DEVICE_TF=${DEVICE_TF:-CPU}
    run_cmd python3 scripts/benchmarks/run_tensorflow_cnn.py --device $DEVICE_TF

    # Run cuDF benchmark (only if GPU)
    if [ "$USE_GPU" = true ]; then
        run_cmd python3 scripts/benchmarks/run_cudf_groupby.py --device gpu
    else
        log_info "Skipping cuDF benchmark (CPU-only mode)."
    fi

    log_success "Benchmarks completed."
}

# Generate leaderboard
generate_leaderboard() {
    log_info "Generating leaderboard..."

    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
    fi

    run_cmd python3 results/generate_leaderboard_md.py

    log_success "Leaderboard generated."
}

# Health check
health_check() {
    log_info "Running health checks..."

    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Check Python
    if ! run_cmd python3 -c "import sys; print(f'Python: {sys.version}')"; then
        log_error "Python check failed"
        return 1
    fi

    # Check PyTorch
    if ! run_cmd python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  {i}: {torch.cuda.get_device_name(i)}')
"; then
        log_error "PyTorch check failed"
        return 1
    fi

    # Check TensorFlow
    if ! run_cmd python3 -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
gpu_devices = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpu_devices)}')
for device in gpu_devices:
    print(f'  {device}')
"; then
        log_error "TensorFlow check failed"
        return 1
    fi

    if [ "$USE_GPU" = true ]; then
        # Check cuDF
        if ! run_cmd python3 -c "
import cudf
print(f'cuDF: {cudf.__version__}')
# Test basic functionality
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print('cuDF basic test passed')
"; then
            log_error "cuDF check failed"
            return 1
        fi
    fi

    # Test CUDA kernel execution (if GPU)
    if [ "$USE_GPU" = true ]; then
        if ! run_cmd python3 -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print('CUDA kernel test passed')
else:
    print('CUDA not available, skipping kernel test')
"; then
            log_warning "CUDA kernel test failed - falling back to CPU"
            USE_GPU=false
        fi
    fi

    log_success "Health checks passed."
}

# Main execution
main() {
    log_info "Starting CUDA WSL Installer v1.0"

    # Detect GPU (non-fatal)
    if ! detect_gpu; then
        log_warning "GPU detection failed, falling back to CPU-only mode"
        USE_GPU=false
    fi

    # Check CUDA (non-fatal if GPU not detected)
    if ! check_cuda; then
        if [ "$USE_GPU" = true ]; then
            log_warning "CUDA check failed, will attempt install"
        fi
    fi

    # Install CUDA (fatal if GPU detected but install fails)
    if ! install_cuda; then
        if [ "$USE_GPU" = true ]; then
            log_error "CUDA installation failed. Falling back to CPU-only mode."
            USE_GPU=false
        fi
    fi

    # Setup venv (fatal)
    if ! setup_venv; then
        log_error "Virtual environment setup failed. Exiting."
        exit 1
    fi

    # Install packages (fatal)
    if ! install_packages; then
        log_error "Package installation failed. Exiting."
        exit 1
    fi

    # Health check (fatal)
    if ! health_check; then
        log_error "Health checks failed. Exiting."
        exit 1
    fi

    # Run benchmarks (non-fatal, with fallback)
    if ! run_benchmarks; then
        log_warning "Some benchmarks failed, but continuing..."
    fi

    # Generate leaderboard (non-fatal)
    if ! generate_leaderboard; then
        log_warning "Leaderboard generation failed, but install is complete"
    fi

    log_success "Installation complete! Leaderboard available at: results/LEADERBOARD.md"
    log_info "To rerun benchmarks: source $VENV_DIR/bin/activate && python3 run_all_benchmarks.py"
}

# Run main
main "$@"
