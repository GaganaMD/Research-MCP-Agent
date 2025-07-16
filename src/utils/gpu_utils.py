# src/utils/gpu_utils.py
import torch
import psutil
from typing import Dict, Any
# import nvidia_ml_py as nvml
import pynvml

class GPUManager:
    """Manage GPU resources and optimization"""
    
    def __init__(self):
        if torch.cuda.is_available():
            pynvml.pynvml()
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = 0
            
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
            
        info = {"gpu_available": True, "devices": []}
        
        for i in range(self.device_count):
            handle = pynvml.pynvml(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            info["devices"].append({
                "id": i,
                "name": name,
                "memory_total": mem_info.total,
                "memory_used": mem_info.used,
                "memory_free": mem_info.free
            })
            
        return info
    
    def optimize_memory(self):
        """Optimize GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def get_optimal_batch_size(self, model_size: str) -> int:
        """Get optimal batch size based on available memory"""
        if not torch.cuda.is_available():
            return 1
            
        gpu_info = self.get_gpu_info()
        memory_gb = gpu_info["devices"][0]["memory_total"] / (1024**3)
        
        # Rough estimates based on model size
        if model_size == "7b":
            return max(1, int(memory_gb / 16))
        elif model_size == "13b":
            return max(1, int(memory_gb / 24))
        elif model_size == "30b":
            return 1
        else:
            return 2