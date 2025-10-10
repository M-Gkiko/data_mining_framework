import time
from contextlib import contextmanager
from typing import Dict, List, Optional


class Timer:
    """Simple timer for measuring execution times."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.measurements: List[float] = []
    
    def start(self) -> None:
        """Start timing."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        elapsed = time.perf_counter() - self.start_time
        self.measurements.append(elapsed)
        self.start_time = None
        return elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.measurements:
            return {}
        
        return {
            'count': len(self.measurements),
            'min': min(self.measurements),
            'max': max(self.measurements),
            'mean': sum(self.measurements) / len(self.measurements),
            'total': sum(self.measurements)
        }
    
    def reset(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()
        self.start_time = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@contextmanager
def time_it(name: str = "Operation"):
    """Context manager for timing operations."""
    timer = Timer(name)
    timer.start()
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        print(f"{name}: {elapsed:.4f}s")


def benchmark(func, *args, iterations: int = 10, **kwargs) -> Dict[str, float]:
    """Benchmark a function multiple times and return statistics."""
    timer = Timer(f"benchmark_{func.__name__}")
    
    for _ in range(iterations):
        timer.start()
        func(*args, **kwargs)
        timer.stop()
    
    return timer.get_stats()
