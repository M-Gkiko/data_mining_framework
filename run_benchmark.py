#!/usr/bin/env python3
"""
Benchmark runner script for clustering algorithms.

This script loads configuration from YAML files and runs automated benchmarks
comparing different clustering algorithms and quality measures.
"""

import sys
import argparse
from pathlib import Path
import traceback

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmarks import (
    BenchmarkConfigLoader,
    AlgorithmRegistry,
    PipelineBenchmark,
    BenchmarkResultExporter
)
from implementations.datasets import CSVDataset
from implementations.clustering.hierarchical import HierarchicalClustering
from implementations.clustering.dbscan import DBSCANClustering
from implementations.clustering.quality.calinski_harabasz import CalinskiHarabaszIndex
from implementations.clustering.quality.davies_bouldin import DaviesBouldinIndex
from implementations.distance.manhattan import ManhattanDistance
from implementations.distance.euclidean import EuclideanDistance


def setup_algorithm_registry() -> AlgorithmRegistry:
    """Set up the algorithm registry with available algorithms."""
    registry = AlgorithmRegistry()
    
    # Register clustering algorithms
    registry.register_algorithm(
        "clustering", 
        "Hierarchical", 
        HierarchicalClustering,
        {"n_clusters": 3, "linkage": "complete", "metric": "precomputed"}
    )
    
    registry.register_algorithm(
        "clustering",
        "DBSCAN", 
        DBSCANClustering,
        {"eps": 0.6, "min_samples": 4}
    )
    
    # Register clustering quality measures
    registry.register_algorithm(
        "clustering_quality",
        "Calinski_Harabasz",
        CalinskiHarabaszIndex,
        {}
    )
    
    registry.register_algorithm(
        "clustering_quality",
        "Davies_Bouldin",
        DaviesBouldinIndex,
        {}
    )
    
    return registry


def load_distance_measure(distance_name: str):
    """Load distance measure from configuration name."""
    distance_measures = {
        "Manhattan": ManhattanDistance,
        "Euclidean": EuclideanDistance,
    }
    
    if distance_name not in distance_measures:
        raise ValueError(f"Unknown distance measure: {distance_name}. Available: {list(distance_measures.keys())}")
    
    return distance_measures[distance_name]()


def load_dataset(dataset_path: str) -> CSVDataset:
    """Load dataset from file path."""
    dataset_file = Path(dataset_path)
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset: {dataset_path}")
    
    # Create CSVDataset instance
    dataset = CSVDataset(str(dataset_file))
    
    print(f"Dataset loaded: {dataset.get_rows()} rows, {dataset.get_columns()} columns")
    return dataset


def run_benchmark_from_config(config_path: str, verbose: bool = True) -> None:
    """
    Run benchmark from YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Enable verbose output
    """
    try:
        print("="*60)
        print("CLUSTERING ALGORITHM BENCHMARK")
        print("="*60)
        
        # Load configuration
        print(f"\n1. Loading configuration from: {config_path}")
        config = BenchmarkConfigLoader.load_config(config_path)
        
        # Validate configuration
        issues = BenchmarkConfigLoader.validate_config(config)
        if issues:
            print("Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return
        
        print(f"   ✓ Configuration loaded: {config.name}")
        print(f"   ✓ Dataset: {config.dataset_path}")
        print(f"   ✓ Iterations: {config.iterations}")
        print(f"   ✓ Output formats: {config.output_formats}")
        
        # Load dataset
        print(f"\n2. Loading dataset")
        dataset = load_dataset(config.dataset_path)
        
        # Set up algorithm registry
        print(f"\n3. Setting up algorithms")
        registry = setup_algorithm_registry()
        
        # Create distance measure from config
        distance_measure = load_distance_measure(config.distance_measure)
        print(f"   ✓ Using {config.distance_measure} distance measure")
        
        # Print algorithm combinations
        combinations = config.generate_combinations()
        print(f"   ✓ Algorithm combinations to test: {len(combinations)}")
        for i, combo in enumerate(combinations, 1):
            combo_str = " + ".join([f"{k}:{v}" for k, v in combo.items()])
            print(f"      {i}. {combo_str}")
        
        # Create and run benchmark
        print(f"\n4. Running benchmark")
        benchmark = PipelineBenchmark(registry)
        benchmark.distance_measure = distance_measure  # Set distance measure from config
        
        results = benchmark.run_benchmark(config, dataset)
        
        # Generate summary statistics
        print(f"\n5. Generating summary")
        summary_stats = benchmark.get_summary_statistics()
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Total runs: {summary_stats.get('total_runs', 0)}")
        print(f"Successful runs: {summary_stats.get('successful_runs', 0)}")
        print(f"Failed runs: {summary_stats.get('failed_runs', 0)}")
        print(f"Success rate: {summary_stats.get('success_rate', 0):.1f}%")
        
        if 'avg_execution_time' in summary_stats:
            print(f"\nTiming Results:")
            print(f"Average execution time: {summary_stats['avg_execution_time']:.3f}s")
            print(f"Fastest execution time: {summary_stats['min_execution_time']:.3f}s")
            print(f"Slowest execution time: {summary_stats['max_execution_time']:.3f}s")
            print(f"Total benchmark time: {summary_stats['total_benchmark_time']:.3f}s")
        
        if 'fastest_combination' in summary_stats:
            fastest = summary_stats['fastest_combination']
            print(f"\nFastest Algorithm Combination:")
            combo_str = " + ".join([f"{k}:{v}" for k, v in fastest['algorithms'].items()])
            print(f"  {combo_str}")
            print(f"  Time: {fastest['time']:.3f}s")
            if fastest['quality_scores']:
                quality_str = ", ".join([f"{k}: {v:.3f}" for k, v in fastest['quality_scores'].items()])
                print(f"  Quality: {quality_str}")
        
        if 'best_quality_combination' in summary_stats:
            best_quality = summary_stats['best_quality_combination']
            print(f"\nBest Quality Algorithm Combination:")
            combo_str = " + ".join([f"{k}:{v}" for k, v in best_quality['algorithms'].items()])
            print(f"  {combo_str}")
            print(f"  Time: {best_quality['time']:.3f}s")
            if best_quality['quality_scores']:
                quality_str = ", ".join([f"{k}: {v:.3f}" for k, v in best_quality['quality_scores'].items()])
                print(f"  Quality: {quality_str}")
            print(f"  Metric used: {best_quality.get('metric_used', 'N/A')}")
        
        # Export results
        print(f"\n6. Exporting results")
        BenchmarkResultExporter.export_results(
            results, 
            config.output_directory,
            config.output_formats,
            config.name
        )
        
        print(f"\n✓ Benchmark completed successfully!")
        print(f"✓ Results exported to: {config.output_directory}")
        
    except Exception as e:
        print(f"\n✗ Benchmark failed with error:")
        print(f"  {str(e)}")
        if verbose:
            print(f"\nFull traceback:")
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run clustering algorithm benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py                                    # Use default config
  python run_benchmark.py --config configs/my_benchmark.yaml # Use custom config
  python run_benchmark.py --verbose                          # Enable verbose output
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default="configs/sample_configs/clustering_benchmark.yaml",
        help="Path to YAML configuration file (default: configs/sample_configs/clustering_benchmark.yaml)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output including full error tracebacks"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark_from_config(args.config, args.verbose)


if __name__ == "__main__":
    main()