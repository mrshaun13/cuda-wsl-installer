#!/usr/bin/env python3
"""Unit tests for CUDA WSL installer components."""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestCudaInstall(unittest.TestCase):
    """Test CUDA installation functions."""
    
    @patch('subprocess.run')
    def test_detect_gpu_success(self, mock_run):
        """Test GPU detection success."""
        from scripts.cuda_install import detect_gpu
        
        # Mock successful nvidia-smi
        mock_run.return_value = MagicMock(
            stdout="NVIDIA GeForce GTX 1080 Ti, 6.1\n",
            returncode=0
        )
        
        gpu_available, compute_cap = detect_gpu()
        self.assertTrue(gpu_available)
        self.assertEqual(compute_cap, "6.1")
        
    @patch('subprocess.run')
    def test_detect_gpu_no_gpu(self, mock_run):
        """Test GPU detection failure."""
        from scripts.cuda_install import detect_gpu
        
        # Mock failed nvidia-smi
        mock_run.side_effect = Exception("nvidia-smi not found")
        
        gpu_available, compute_cap = detect_gpu()
        self.assertFalse(gpu_available)
        self.assertIsNone(compute_cap)

class TestEnvSetup(unittest.TestCase):
    """Test environment setup functions."""
    
    @patch('venv.create')
    @patch('os.path.exists')
    def test_setup_venv_new(self, mock_exists, mock_create):
        """Test creating new venv."""
        mock_exists.return_value = False
        
        from scripts.env_setup import setup_venv
        
        result = setup_venv('/tmp/test_venv')
        self.assertEqual(result, '/tmp/test_venv')
        mock_create.assert_called_once()
        
    @patch('os.path.exists')
    def test_setup_venv_existing(self, mock_exists):
        """Test existing venv."""
        mock_exists.return_value = True
        
        from scripts.env_setup import setup_venv
        
        result = setup_venv('/tmp/test_venv')
        self.assertEqual(result, '/tmp/test_venv')

class TestBenchmarkRunner(unittest.TestCase):
    """Test benchmark runner functions."""
    
    @patch('subprocess.run')
    def test_run_pytorch_benchmark_success(self, mock_run):
        """Test PyTorch benchmark success."""
        from scripts.benchmark_runner import run_pytorch_benchmark
        
        mock_run.return_value = MagicMock(returncode=0)
        
        result = run_pytorch_benchmark('cpu')
        self.assertTrue(result)
        
    @patch('subprocess.run')
    def test_run_pytorch_benchmark_failure(self, mock_run):
        """Test PyTorch benchmark failure."""
        from scripts.benchmark_runner import run_pytorch_benchmark
        
        mock_run.return_value = MagicMock(returncode=1)
        
        result = run_pytorch_benchmark('cpu')
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
