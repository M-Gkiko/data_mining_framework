"""

This package provides tools for comparing different clustering,
dimensionality reduction, and quality measurement algorithms.
"""

from .core import (
    BenchmarkConfig,
    BenchmarkResult,
    SimpleBenchmark,
    build_benchmark_pipeline
)

from .registry import (
    create_algorithm,
    create_adapter,
    create_distance_measure,
    get_available_algorithms
)

from .utils import (
    load_benchmark_config,
    export_benchmark_results,
    validate_benchmark_config,
    print_benchmark_summary
)


__all__ = [
    # Core classes
    'BenchmarkConfig',
    'BenchmarkResult', 
    'SimpleBenchmark',
    'build_benchmark_pipeline',
    # Registry functions
    'create_algorithm',
    'create_adapter',
    'create_distance_measure',
    'get_available_algorithms',
    # Utility functions
    'load_benchmark_config',
    'export_benchmark_results',
    'validate_benchmark_config',
    'print_benchmark_summary',
]
