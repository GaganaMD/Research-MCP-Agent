# src/utils/monitoring.py
import logging
import time
from functools import wraps
from typing import Any, Callable
import psutil
import torch
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_memory: float
    response_time: float
    token_count: int

class PerformanceMonitor:
    """Monitor system and model performance"""
    
    def __init__(self):
        self.metrics = []
        self.logger = logging.getLogger(__name__)
    
    def log_performance(self, func: Callable) -> Callable:
        """Decorator to log function performance"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().percent
            
            if torch.cuda.is_available():
                start_gpu = torch.cuda.memory_allocated() / 1024**3
            else:
                start_gpu = 0
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_memory=start_gpu,
                response_time=response_time,
                token_count=len(str(result).split()) if result else 0
            )
            
            self.metrics.append(metrics)
            self.logger.info(f"Function {func.__name__} took {response_time:.2f}s")
            
            return result
        return wrapper
    
    def get_average_metrics(self, last_n: int = 100) -> dict:
        """Get average metrics for last n calls"""
        recent_metrics = self.metrics[-last_n:]
        
        if not recent_metrics:
            return {}
        
        return {
            "avg_response_time": sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_gpu_memory": sum(m.gpu_memory for m in recent_metrics) / len(recent_metrics),
            "total_requests": len(recent_metrics)
        }