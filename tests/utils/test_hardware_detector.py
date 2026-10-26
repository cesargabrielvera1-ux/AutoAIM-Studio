"""Tests for hardware_detector module."""

import pytest
import numpy as np

from app.utils.hardware_detector import (
    HardwareDetector, HardwareInfo, get_hardware_detector, setup_hardware_optimizations
)


class TestHardwareInfo:
    """Test suite for HardwareInfo dataclass."""

    def test_creation(self):
        """Test creating HardwareInfo."""
        info = HardwareInfo(
            cpu_count=4,
            cpu_name='Test CPU',
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            has_cuda=False,
            cuda_version=None,
            gpu_count=0,
            gpu_names=[],
            gpu_memory_gb=[],
            recommended_device='cpu',
            recommended_n_jobs=3
        )
        assert info.cpu_count == 4
        assert info.has_cuda is False
        assert info.recommended_device == 'cpu'


class TestHardwareDetector:
    """Test suite for HardwareDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = HardwareDetector()
        assert detector is not None

    def test_info_property(self):
        """Test info property."""
        detector = HardwareDetector()
        info = detector.info
        assert isinstance(info, HardwareInfo)

    def test_info_caching(self):
        """Test that info is cached."""
        detector = HardwareDetector()
        info1 = detector.info
        info2 = detector.info
        assert info1 is info2  # Same object (cached)

    def test_cpu_detection(self):
        """Test CPU detection."""
        detector = HardwareDetector()
        assert detector.info.cpu_count > 0
        assert detector.info.cpu_name != 'Unknown CPU'

    def test_memory_detection(self):
        """Test memory detection."""
        detector = HardwareDetector()
        assert detector.info.total_memory_gb > 0
        assert detector.info.available_memory_gb > 0
        assert detector.info.available_memory_gb <= detector.info.total_memory_gb

    def test_gpu_detection(self):
        """Test GPU detection."""
        detector = HardwareDetector()
        # Either has CUDA or doesn't - both are valid
        assert isinstance(detector.info.has_cuda, bool)
        if detector.info.has_cuda:
            assert detector.info.gpu_count > 0
            assert len(detector.info.gpu_names) == detector.info.gpu_count

    def test_recommended_device(self):
        """Test recommended device."""
        detector = HardwareDetector()
        if detector.info.has_cuda:
            assert detector.info.recommended_device == 'cuda'
        else:
            assert detector.info.recommended_device == 'cpu'

    def test_recommended_n_jobs(self):
        """Test recommended n_jobs."""
        detector = HardwareDetector()
        assert detector.info.recommended_n_jobs >= 1
        assert detector.info.recommended_n_jobs <= detector.info.cpu_count

    def test_get_torch_device(self):
        """Test getting torch device."""
        detector = HardwareDetector()
        device = detector.get_torch_device()
        
        if detector.info.has_cuda:
            import torch
            assert isinstance(device, torch.device)
            assert str(device) == 'cuda'
        else:
            import torch
            assert isinstance(device, torch.device)
            assert str(device) == 'cpu'

    def test_optimize_numpy(self):
        """Test NumPy optimization."""
        detector = HardwareDetector()
        # Should not raise
        detector.optimize_numpy()

    def test_optimize_sklearn(self):
        """Test sklearn optimization."""
        detector = HardwareDetector()
        # Should not raise
        detector.optimize_sklearn()
        assert detector._sklearn_configured is True

    def test_optimize_sklearn_with_n_jobs(self):
        """Test sklearn optimization with specific n_jobs."""
        detector = HardwareDetector()
        detector.optimize_sklearn(n_jobs=2)
        assert detector._sklearn_configured is True

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        detector = HardwareDetector()
        
        batch_size = detector.get_optimal_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= 512

    def test_batch_size_with_sample_size(self):
        """Test batch size with sample size."""
        detector = HardwareDetector()
        
        small_batch = detector.get_optimal_batch_size(sample_size=500)
        large_batch = detector.get_optimal_batch_size(sample_size=200000)
        
        assert small_batch <= large_batch

    def test_get_memory_limit(self):
        """Test memory limit calculation."""
        detector = HardwareDetector()
        limit = detector.get_memory_limit_gb()
        
        assert isinstance(limit, float)
        assert limit > 0
        assert limit <= 8.0  # Should be capped at 8GB

    def test_format_info(self):
        """Test info formatting."""
        detector = HardwareDetector()
        info_str = detector.format_info()
        
        assert isinstance(info_str, str)
        assert 'HARDWARE INFORMATION' in info_str
        assert detector.info.cpu_name in info_str

    def test_format_info_contains_cuda(self):
        """Test that format_info contains CUDA info when available."""
        detector = HardwareDetector()
        info_str = detector.format_info()
        
        assert 'CUDA Available' in info_str
        if detector.info.has_cuda:
            assert 'GPU Count' in info_str


class TestGetHardwareDetector:
    """Test suite for get_hardware_detector."""

    def test_singleton(self):
        """Test that get_hardware_detector returns singleton."""
        detector1 = get_hardware_detector()
        detector2 = get_hardware_detector()
        assert detector1 is detector2

    def test_returns_detector(self):
        """Test that function returns HardwareDetector."""
        detector = get_hardware_detector()
        assert isinstance(detector, HardwareDetector)


class TestSetupHardwareOptimizations:
    """Test suite for setup_hardware_optimizations."""

    def test_setup(self):
        """Test hardware optimizations setup."""
        info = setup_hardware_optimizations()
        assert isinstance(info, HardwareInfo)
