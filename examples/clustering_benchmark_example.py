"""
Clustering Benchmark Example: Testing All DR + Clustering + Quality Combinations

This example demonstrates benchmarking using the programmatic API.
For simpler usage, consider using YAML configuration files (see yaml_benchmark_example.py).

This example tests combinations of:
- Multiple dimensionality reduction algorithms (PCA, MDS, t-SNE)
- Multiple clustering algorithms (Hierarchical, DBSCAN, K-Means)
- Multiple quality measures (Calinski-Harabasz, Davies-Bouldin)

The benchmark tests all combinations and reports timing and quality metrics.
"""

from data_mining_framework import (
    CSVDataset,
    load_benchmark_config,
    SimpleBenchmark,
    export_benchmark_results
)


def main():
    print("Clustering Benchmark Example: All DR + Clustering + Quality Combinations\n")
    print("=" * 80)

    # Load configuration from YAML file
    # For this example, we use the pre-configured clustering_benchmark.yaml
    import os
    yaml_path = 'clustering_benchmark.yaml' if os.path.exists('clustering_benchmark.yaml') else 'examples/clustering_benchmark.yaml'
    config = load_benchmark_config(yaml_path)

    print(f"Benchmark: {config.name}")
    print(f"Dataset: {config.dataset}")
    print(f"Iterations: {config.iterations}")

    # Load dataset
    dataset = CSVDataset(config.dataset)
    print(f"Dataset loaded: {dataset.get_rows()} samples, {len(dataset.get_features())} features\n")

    # Generate and display combinations
    combinations = config.generate_combinations()
    print(f"Testing {len(combinations)} algorithm combinations:")
    for i, combo in enumerate(combinations, 1):
        clustering = combo.get('clustering', 'None')
        quality = combo.get('clustering_quality', 'None')
        print(f"  {i}. {clustering} + {quality}")

    # Create and run benchmark
    print(f"\nRunning benchmark ({config.iterations} iterations per combination)...\n")
    benchmark = SimpleBenchmark(config)
    results = benchmark.run(dataset)

    # Display summary
    print("\nBenchmark Results Summary:")
    print("=" * 80)

    for i, result in enumerate(results.results[:5], 1):  # Show top 5
        combo_name = f"{result.combination.get('clustering', 'None')} + " \
                     f"{result.combination.get('clustering_quality', 'None')}"
        print(f"\n{i}. {combo_name}:")
        print(f"   Average Time: {result.avg_time:.4f}s")
        if hasattr(result, 'avg_quality') and result.avg_quality is not None:
            print(f"   Quality Score: {result.avg_quality:.4f}")
        print(f"   Std Dev: {result.std_time:.4f}s")

    # Export results
    export_benchmark_results(results, config)
    print(f"\n{'=' * 80}")
    print(f"Full results exported to: {config.output_directory}/")
    print(f"  - CSV format: timing and quality data")
    print(f"  - JSON format: complete results with metadata")
    print(f"\nTIP: You can also run this benchmark using:")
    print(f"  python examples/run_benchmark_example.py -c examples/clustering_benchmark.yaml")


if __name__ == "__main__":
    main()
