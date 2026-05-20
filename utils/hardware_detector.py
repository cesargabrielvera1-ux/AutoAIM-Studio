"""Hardware detection and optimization utilities."""

import os
import sys
import warnings
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class HardwareInfo:
    """Hardware information container."""
    cpu_count: int
    cpu_name: str
    total_memory_gb: float
    available_memory_gb: float
    has_cuda: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpu_names: list
    gpu_memory_gb: list
    recommended_device: str
    recommended_n_jobs: int


class HardwareDetector:
    """Detect and manage hardware capabilities."""
    
    def __init__(self):
        """Initialize hardware detector."""
        self._info: Optional[HardwareInfo] = None
        self._torch_available = self._check_torch()
        self._sklearn_configured = False
    
    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    @property
    def info(self) -> HardwareInfo:
        """Get hardware information."""
        if self._info is None:
            self._info = self._detect_hardware()
        return self._info
    
    def _detect_hardware(self) -> HardwareInfo:
        """Detect hardware capabilities."""
        import psutil
        
        # CPU information
        cpu_count = os.cpu_count() or 1
        cpu_name = self._get_cpu_name()
        
        # Memory information
        mem = psutil.virtual_memory()
        total_memory_gb = mem.total / (1024 ** 3)
        available_memory_gb = mem.available / (1024 ** 3)
        
        # GPU information
        has_cuda, cuda_version, gpu_count, gpu_names, gpu_memory_gb = self._get_gpu_info()
        
        # Determine recommended device
        recommended_device = "cuda" if has_cuda else "cpu"
        
        # Determine recommended n_jobs (leave one core free for system)
        recommended_n_jobs = max(1, cpu_count - 1)
        
        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_name=cpu_name,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            has_cuda=has_cuda,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_gb=gpu_memory_gb,
            recommended_device=recommended_device,
            recommended_n_jobs=recommended_n_jobs
        )
    
    def _get_cpu_name(self) -> str:
        """Get CPU name."""
        try:
            if sys.platform == "win32":
                import platform
                return platform.processor()
            elif sys.platform == "darwin":
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()
            else:  # Linux
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
        except Exception:
            pass
        return "Unknown CPU"
    
    def _get_gpu_info(self) -> Tuple[bool, Optional[str], int, list, list]:
        """Get GPU information."""
        has_cuda = False
        cuda_version = None
        gpu_count = 0
        gpu_names = []
        gpu_memory_gb = []
        
        if self._torch_available:
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                
                if has_cuda:
                    cuda_version = torch.version.cuda
                    gpu_count = torch.cuda.device_count()
                    
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_names.append(gpu_name)
                        
                        # Get GPU memory
                        try:
                            props = torch.cuda.get_device_properties(i)
                            mem_gb = props.total_memory / (1024 ** 3)
                            gpu_memory_gb.append(mem_gb)
                        except Exception:
                            gpu_memory_gb.append(0.0)
            except Exception:
                pass
        
        # Fallback to pycuda or nvidia-ml-py if torch not available
        if not has_cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                has_cuda = True
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    gpu_names.append(gpu_name)
                    
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_gb = mem_info.total / (1024 ** 3)
                    gpu_memory_gb.append(mem_gb)
                    
                pynvml.nvmlShutdown()
            except Exception:
                pass
        
        return has_cuda, cuda_version, gpu_count, gpu_names, gpu_memory_gb
    
    def get_torch_device(self) -> Any:
        """Get optimal PyTorch device."""
        if self._torch_available:
            import torch
            if self.info.has_cuda:
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return None
    
    def optimize_numpy(self) -> None:
        """Optimize NumPy for current hardware."""
        # Set number of threads for NumPy/BLAS
        n_threads = self.info.recommended_n_jobs
        
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    
    def optimize_sklearn(self, n_jobs: Optional[int] = None) -> None:
        """Configure scikit-learn for optimal performance.
        
        Args:
            n_jobs: Number of parallel jobs. If None, uses recommended value.
        """
        if n_jobs is None:
            n_jobs = self.info.recommended_n_jobs
        
        # Set environment variable
        os.environ['SKLEARN_PARALLEL'] = str(n_jobs)
        
        self._sklearn_configured = True
    
    def get_optimal_batch_size(self, sample_size: int = None) -> int:
        """Get optimal batch size based on hardware.
        
        Args:
            sample_size: Number of samples in dataset
            
        Returns:
            Optimal batch size
        """
        if self.info.has_cuda:
            # For GPU, use powers of 2
            base_batch = 64
            
            # Adjust based on GPU memory
            if self.info.gpu_memory_gb:
                avg_mem = sum(self.info.gpu_memory_gb) / len(self.info.gpu_memory_gb)
                if avg_mem > 16:
                    base_batch = 256
                elif avg_mem > 8:
                    base_batch = 128
        else:
            # For CPU, smaller batches
            base_batch = 32
            
            # Adjust based on available memory
            if self.info.available_memory_gb > 8:
                base_batch = 64
        
        # Adjust based on sample size
        if sample_size is not None:
            if sample_size < 1000:
                base_batch = min(base_batch, 32)
            elif sample_size > 100000:
                base_batch = min(base_batch * 2, 512)
        
        return base_batch
    
    def get_memory_limit_gb(self) -> float:
        """Get recommended memory limit for operations."""
        # Use 80% of available memory or 8GB, whichever is smaller
        return min(self.info.available_memory_gb * 0.8, 8.0)
    
    def format_info(self) -> str:
        """Format hardware information as string."""
        info = self.info
        lines = [
            "=" * 50,
            "HARDWARE INFORMATION",
            "=" * 50,
            f"CPU: {info.cpu_name}",
            f"CPU Cores: {info.cpu_count}",
            f"Total Memory: {info.total_memory_gb:.2f} GB",
            f"Available Memory: {info.available_memory_gb:.2f} GB",
            f"CUDA Available: {'Yes' if info.has_cuda else 'No'}",
        ]
        
        if info.has_cuda:
            lines.extend([
                f"CUDA Version: {info.cuda_version}",
                f"GPU Count: {info.gpu_count}",
            ])
            for i, (name, mem) in enumerate(zip(info.gpu_names, info.gpu_memory_gb)):
                lines.append(f"  GPU {i}: {name} ({mem:.2f} GB)")
        
        lines.extend([
            f"Recommended Device: {info.recommended_device}",
            f"Recommended n_jobs: {info.recommended_n_jobs}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


# Global instance
_hardware_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Get global hardware detector instance."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector


def setup_hardware_optimizations() -> HardwareInfo:
    """Setup all hardware optimizations.
    
    Returns:
        Hardware information
    """
    detector = get_hardware_detector()
    detector.optimize_numpy()
    detector.optimize_sklearn()
    
    return detector.info
