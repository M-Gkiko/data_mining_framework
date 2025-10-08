#!/usr/bin/env python3
"""
Simplified benchmark runner script for clustering algorithms.

This script loads configuration from YAML files and runs automated benchmarks
comparing different clustering algorithms and quality measures.
"""

import sys
import argparse
from pathlib import Path
import traceback
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmarks import (
    load_benchmark_config,
    SimpleBenchmark,
    export_benchmark_results,
    print_benchmark_summary
)
from implementations.datasets import CSVDataset
from implementations.distance.manhattan import ManhattanDistance
from implementations.distance.euclidean import EuclideanDistance


def load_distance_measure(distance_name: str):
    """
    Dynamically load distance measure from configuration name.
    
    This function automatically discovers distance measure classes from the
    implementations/distance directory, allowing you to use any distance measure
    without hardcoding it.
    """
    import importlib
    import os
    from pathlib import Path
    
    # First, try the common ones for quick access
    common_measures = {
        "Manhattan": ManhattanDistance,
        "Euclidean": EuclideanDistance,
    }
    
    if distance_name in common_measures:
        return common_measures[distance_name]()
    
    # Dynamic discovery: look for distance measure classes
    distance_dir = Path("implementations/distance")
    available_measures = {}
    
    if distance_dir.exists():
        for py_file in distance_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            module_name = py_file.stem
            try:
                # Import the module dynamically
                module = importlib.import_module(f"implementations.distance.{module_name}")
                
                # Look for classes that end with "Distance"
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith("Distance") and 
                        attr_name != "DistanceMeasure"):  # Exclude base class
                        
                        # Match by class name (e.g., "Cosine" matches "CosineDistance")
                        class_key = attr_name.replace("Distance", "")
                        available_measures[class_key] = attr
                        available_measures[attr_name] = attr  # Also allow full name
                        
            except Exception as e:
                print(f"Warning: Could not load distance measure from {py_file}: {e}")
    
    # Try to find the requested distance measure
    if distance_name in available_measures:
        return available_measures[distance_name]()
    
    # If not found, show available options
    all_available = list(set(list(common_measures.keys()) + list(available_measures.keys())))
    raise ValueError(f"Unknown distance measure: '{distance_name}'. Available: {sorted(all_available)}")


def load_dataset(dataset_path: str) -> CSVDataset:
    """Load dataset from file path."""
    dataset_file = Path(dataset_path)
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset: {dataset_path}")
    
    # Create CSVDataset instance
    dataset = CSVDataset(str(dataset_file))
    
    # Handle different return types from get_columns()
    columns = dataset.get_columns()
    if isinstance(columns, int):
        num_columns = columns
    else:
        num_columns = len(columns)
    
    print(f"Dataset loaded: {dataset.get_rows()} rows, {num_columns} columns")
    
    return dataset


def run_benchmark(config_path: str, verbose: bool = False):
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
        
        # Load configuration with automatic validation
        print(f"\n1. Loading configuration from: {config_path}")
        config = load_benchmark_config(config_path)
        
        print(f"   ✓ Configuration loaded: {config.name}")
        print(f"   ✓ Dataset: {config.dataset}")
        print(f"   ✓ Iterations: {config.iterations}")
        print(f"   ✓ Output formats: {config.output_formats}")
        
        # Load dataset
        print(f"\n2. Loading dataset")
        dataset = load_dataset(config.dataset)
        
        # Create distance measure from config
        print(f"\n3. Setting up algorithms")
        distance_measure = load_distance_measure(config.distance_measure)
        print(f"   ✓ Using {config.distance_measure} distance measure")
        
        # Print algorithm combinations
        combinations = config.generate_combinations()
        print(f"   ✓ Algorithm combinations to test: {len(combinations)}")
        for i, combo in enumerate(combinations, 1):
            combo_str = " + ".join([f"{k}:{v}" for k, v in combo.items()])
            print(f"      {i}. {combo_str}")
        
        # Create and run simplified benchmark
        print(f"\n4. Running benchmark")
        benchmark = SimpleBenchmark(config, distance_measure)
        results = benchmark.run(dataset)
        
        # Print summary
        print(f"\n5. Generating summary")
        print_benchmark_summary(results, config)
        
        # Export results
        print(f"\n6. Exporting results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_benchmark_results(results, config, f"_{timestamp}")
        
        print(f"\n✓ Benchmark completed successfully!")
        print(f"✓ Results exported to: {config.output_directory}")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {str(e)}")
        if verbose:
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
    run_benchmark(args.config, args.verbose)


if __name__ == "__main__":
    main()