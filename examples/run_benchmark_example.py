#!/usr/bin/env python3
"""
Comprehensive benchmark runner script for all data mining framework components.

Supports:
- Clustering algorithms (Hierarchical, DBSCAN, K-Means)
- Dimensionality Reduction (PCA, MDS, t-SNE, Sammon Mapping)
- Network Analysis (Community Detection, Node Measures, Edge Measures)
- Quality measures for clustering and DR
- Distance measures
"""

import sys
import argparse
from pathlib import Path
import traceback
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_mining_framework import (
    # Core benchmarking
    load_benchmark_config,
    SimpleBenchmark,
    export_benchmark_results,

    # Dataset implementations
    CSVDataset,
    NumpyDataset,

    # Core interfaces
    Dataset,
    Network,

    # Network implementations
    NetworkXWrapper,
    EdgeListNetwork,
    AdjacencyMatrixNetwork,
)
from data_mining_framework.benchmarks.utils import print_benchmark_summary


def load_dataset(dataset_path: str):
    """
    Load dataset from file path.

    Supports:
    - CSV files for clustering/DR (.csv)
    - Edge lists for networks (.edgelist, .edges)
    - Graph formats for networks (.gml, .graphml, .gexf)
    - Adjacency matrices (.csv with square matrix)

    Returns:
        Dataset or Network object
    """
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")

    # Determine file type and load accordingly
    suffix = dataset_file.suffix.lower()

    # Network file formats
    if suffix in ['.edgelist', '.edges']:
        dataset = EdgeListNetwork(str(dataset_file))
        print(f"Network loaded: {dataset.node_count()} nodes, {dataset.edge_count()} edges")
    elif suffix in ['.gml', '.graphml', '.gexf', '.adjlist']:
        dataset = NetworkXWrapper.from_file(str(dataset_file))
        print(f"Network loaded: {dataset.node_count()} nodes, {dataset.edge_count()} edges")
    # CSV - could be dataset or adjacency matrix
    elif suffix == '.csv':
        # Try loading as regular dataset first
        dataset = CSVDataset(str(dataset_file))

        # Handle different return types from get_columns()
        columns = dataset.get_columns()
        if isinstance(columns, int):
            num_columns = columns
        else:
            num_columns = len(columns)

        print(f"Dataset loaded: {dataset.get_rows()} rows, {num_columns} columns")
    else:
        # Default to CSV dataset
        dataset = CSVDataset(str(dataset_file))
        columns = dataset.get_columns()
        if isinstance(columns, int):
            num_columns = columns
        else:
            num_columns = len(columns)
        print(f"Dataset loaded: {dataset.get_rows()} rows, {num_columns} columns")

    return dataset


def detect_benchmark_type(config) -> str:
    """
    Detect the type of benchmark from configuration.

    Returns:
        'clustering', 'dr', 'network', or 'mixed'
    """
    combinations = config.generate_combinations()
    if not combinations:
        return 'unknown'

    # Check first combination for algorithm types
    first_combo = combinations[0]

    # Network analysis indicators
    network_types = {'community_detection', 'node_measure', 'edge_measure'}
    has_network = any(key in first_combo for key in network_types)

    # Traditional ML indicators
    has_clustering = 'clustering' in first_combo
    has_dr = 'dimensionality_reduction' in first_combo

    if has_network:
        return 'network'
    elif has_dr and has_clustering:
        return 'mixed'
    elif has_dr:
        return 'dr'
    elif has_clustering:
        return 'clustering'
    else:
        return 'unknown'


def run_benchmark(config_path: str, verbose: bool = False):
    """
    Run benchmark from YAML configuration file.

    Supports all framework components:
    - Clustering algorithms
    - Dimensionality Reduction algorithms
    - Network Analysis (community detection, node measures, edge measures)
    - Quality measures
    - Distance measures
    """
    try:
        print("="*70)
        print("DATA MINING FRAMEWORK - COMPREHENSIVE BENCHMARK")
        print("="*70)

        # Load configuration
        print(f"\n1. Loading configuration from: {config_path}")
        config = load_benchmark_config(config_path)

        # Detect benchmark type
        benchmark_type = detect_benchmark_type(config)

        print(f"   [OK] Configuration loaded: {config.name}")
        print(f"   [OK] Benchmark type: {benchmark_type.upper()}")
        print(f"   [OK] Dataset: {config.dataset}")

        # Print type-specific configuration details
        if hasattr(config, 'distance_measure') and config.distance_measure:
            print(f"   [OK] Distance measure: {config.distance_measure}")
        print(f"   [OK] Iterations: {config.iterations}")
        print(f"   [OK] Output formats: {config.output_formats}")

        # Load dataset
        print(f"\n2. Loading dataset/network")
        dataset = load_dataset(config.dataset)

        # Print algorithm combinations
        print(f"\n3. Setting up algorithms")
        combinations = config.generate_combinations()
        print(f"   [OK] Algorithm combinations to test: {len(combinations)}")

        # Print combinations with better formatting
        for i, combo in enumerate(combinations, 1):
            if benchmark_type == 'network':
                # Format network combinations more clearly
                parts = []
                for k, v in combo.items():
                    if k in ['community_detection', 'node_measure', 'edge_measure']:
                        parts.append(f"{k.replace('_', ' ').title()}: {v}")
                combo_str = " + ".join(parts)
            else:
                combo_str = " + ".join([f"{k}: {v}" for k, v in combo.items()])
            print(f"      {i}. {combo_str}")

        # Create and run benchmark
        print(f"\n4. Running benchmark")
        print(f"   Running {len(combinations)} combinations × {config.iterations} iterations...")
        benchmark = SimpleBenchmark(config)
        results = benchmark.run(dataset)

        # Print summary and export
        print(f"\n5. Generating summary")
        print_benchmark_summary(results, config)

        print(f"\n6. Exporting results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_benchmark_results(results, config, f"_{timestamp}")

        print(f"\n{'='*70}")
        print(f"[SUCCESS] Benchmark completed successfully!")
        print(f"[SUCCESS] Results exported to: {config.output_directory}")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"[ERROR] Benchmark failed with error: {str(e)}")
        print(f"{'='*70}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for comprehensive benchmark runner."""
    parser = argparse.ArgumentParser(
        description="""
Data Mining Framework - Comprehensive Benchmark Runner

Supports benchmarking of:
  - Clustering algorithms (Hierarchical, DBSCAN, K-Means)
  - Dimensionality Reduction (PCA, MDS, t-SNE, Sammon Mapping)
  - Network Analysis (Community Detection, Node/Edge Measures)
  - Quality measures for clustering and DR
  - Distance measures (Manhattan, Euclidean, Cosine)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run clustering benchmark
  python run_benchmark.py --config examples/clustering_benchmark.yaml

  # Run dimensionality reduction benchmark
  python run_benchmark.py --config examples/dr_cl_quality.yaml

  # Run network analysis benchmark
  python run_benchmark.py --config examples/network_benchmark.yaml

  # Enable verbose output for debugging
  python run_benchmark.py --config my_config.yaml --verbose

Configuration File Format:
  The YAML configuration file should specify:
    - benchmark.name: Name of the benchmark
    - benchmark.dataset: Path to dataset/network file
    - benchmark.iterations: Number of iterations to run
    - pipeline_template: List of algorithm configurations to test

  See examples/ directory for configuration templates.
        """
    )

    parser.add_argument(
        "--config", "-c",
        default="examples/dr_cl_quality.yaml",
        help="Path to YAML benchmark configuration file (default: %(default)s)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with full error traces"
    )

    args = parser.parse_args()

    # Display startup banner
    print("\n" + "="*70)
    print("DATA MINING FRAMEWORK - BENCHMARK RUNNER")
    print("="*70)
    print(f"Version: 0.1.0")
    print(f"Configuration: {args.config}")
    print(f"Verbose: {args.verbose}")
    print("="*70 + "\n")

    run_benchmark(args.config, args.verbose)


if __name__ == "__main__":
    main()