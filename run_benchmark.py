#!/usr/bin/env python3
"""
Simplified benchmark runner script for clustering algorithms.
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


def load_dataset(dataset_path: str) -> CSVDataset:
    """Load dataset from file path."""
    dataset_file = Path(dataset_path)
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset: {dataset_path}")
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
    """Run benchmark from YAML configuration file."""
    try:
        print("="*60)
        print("CLUSTERING ALGORITHM BENCHMARK")
        print("="*60)
        
        # Load configuration
        print(f"\n1. Loading configuration from: {config_path}")
        config = load_benchmark_config(config_path)
        
        print(f"   ✓ Configuration loaded: {config.name}")
        print(f"   ✓ Dataset: {config.dataset}")
        print(f"   ✓ Distance measure: {config.distance_measure}")
        print(f"   ✓ Iterations: {config.iterations}")
        print(f"   ✓ Output formats: {config.output_formats}")
        
        # Load dataset
        print(f"\n2. Loading dataset")
        dataset = load_dataset(config.dataset)
        
        # Print algorithm combinations
        print(f"\n3. Setting up algorithms")
        combinations = config.generate_combinations()
        print(f"   ✓ Algorithm combinations to test: {len(combinations)}")
        for i, combo in enumerate(combinations, 1):
            combo_str = " + ".join([f"{k}:{v}" for k, v in combo.items()])
            print(f"      {i}. {combo_str}")
        
        # Create and run benchmark 
        print(f"\n4. Running benchmark")
        benchmark = SimpleBenchmark(config)  
        results = benchmark.run(dataset)
        
        # Print summary and export
        print(f"\n5. Generating summary")
        print_benchmark_summary(results, config)
        
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
        default="configs/sample_configs/dr_cl_quality.yaml",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    run_benchmark(args.config, args.verbose)


if __name__ == "__main__":
    main()