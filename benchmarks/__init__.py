"""
Simplified benchmark package for systematic algorithm comparison.

This package provides streamlined tools for comparing different clustering,
dimensionality reduction, and quality measurement algorithms.
"""

from .core import (
    BenchmarkConfig,
    BenchmarkResult,
    SimpleBenchmark,
    create_clustering_algorithm,
    create_quality_measure,
    create_algorithm,
    build_benchmark_pipeline
)

from .utils import (
    load_benchmark_config,
    export_benchmark_results,
    validate_benchmark_config,
    print_benchmark_summary
)

# Backward compatibility aliases
BenchmarkConfiguration = BenchmarkConfig
PipelineBenchmark = SimpleBenchmark
BenchmarkConfigLoader = type('BenchmarkConfigLoader', (), {
    'load_config': staticmethod(load_benchmark_config)
})

__all__ = [
    # New simplified API
    'BenchmarkConfig',
    'BenchmarkResult', 
    'SimpleBenchmark',
    'load_benchmark_config',
    'export_benchmark_results',
    'validate_benchmark_config',
    'print_benchmark_summary',
    'create_clustering_algorithm',
    'create_quality_measure',
    'create_algorithm',
    'build_benchmark_pipeline',
    
    # Backward compatibility
    'BenchmarkConfiguration',
    'PipelineBenchmark',
    'BenchmarkConfigLoader'
]
