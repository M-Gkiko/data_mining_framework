"""
YAML Benchmark Example: Running Benchmarks from YAML Configuration Files

This example demonstrates using YAML configuration files to run benchmarks.
YAML configs allow you to:
- Define benchmark parameters externally
- Test multiple algorithm combinations easily
- Share and reuse benchmark configurations
- Run benchmarks from command line or programmatically

Example YAML configs available:
- clustering_benchmark.yaml: Clustering algorithms
- network_benchmark.yaml: Network analysis algorithms
- dr_cl_quality.yaml: DR + Clustering + Quality
"""

from data_mining_framework import (
    load_benchmark_config,
    SimpleBenchmark,
    export_benchmark_results,
    CSVDataset,
    NetworkXWrapper
)


def run_clustering_yaml_benchmark():
    """Example 1: Run clustering benchmark from YAML config"""
    print("Example 1: Clustering Benchmark from YAML\n")
    print("=" * 80)

    # Load configuration from YAML file
    import os
    yaml_path = 'clustering_benchmark.yaml' if os.path.exists('clustering_benchmark.yaml') else 'examples/clustering_benchmark.yaml'
    config = load_benchmark_config(yaml_path)

    print(f"Benchmark: {config.name}")
    print(f"Dataset: {config.dataset}")
    print(f"Iterations: {config.iterations}")

    # Generate and display combinations
    combinations = config.generate_combinations()
    print(f"\nAlgorithm Combinations ({len(combinations)} total):")
    for i, combo in enumerate(combinations, 1):
        clustering = combo.get('clustering', 'None')
        quality = combo.get('clustering_quality', 'None')
        print(f"  {i}. {clustering} + {quality}")

    # Load dataset
    dataset = CSVDataset(config.dataset)
    print(f"\nDataset: {dataset.get_rows()} rows, {len(dataset.get_features())} features")

    # Run benchmark
    print(f"\nRunning benchmark...")
    benchmark = SimpleBenchmark(config)
    results = benchmark.run(dataset)

    # Display results summary
    print("\nSample Results (first 3):")
    print("-" * 80)
    results_list = results if isinstance(results, list) else results.results
    for i, result in enumerate(results_list[:3], 1):
        combo_name = f"{result.combination.get('clustering', 'None')} + " \
                     f"{result.combination.get('clustering_quality', 'None')}"
        print(f"{i}. {combo_name}")
        print(f"   Time: {result.execution_time:.4f}s")
        if result.quality_scores:
            qual_str = ", ".join([f"{k}: {v:.4f}" for k, v in result.quality_scores.items()])
            print(f"   Quality: {qual_str}")

    # Export results
    export_benchmark_results(results, config)
    print(f"\nResults exported to: {config.output_directory}/")
    print("=" * 80)


def run_network_yaml_benchmark():
    """Example 2: Run network analysis benchmark from YAML config"""
    print("\n\nExample 2: Network Analysis Benchmark from YAML\n")
    print("=" * 80)

    # Load configuration from YAML file
    import os
    yaml_path = 'network_benchmark.yaml' if os.path.exists('network_benchmark.yaml') else 'examples/network_benchmark.yaml'
    config = load_benchmark_config(yaml_path)

    print(f"Benchmark: {config.name}")
    print(f"Dataset: {config.dataset}")
    print(f"Iterations: {config.iterations}")

    # Generate and display combinations
    combinations = config.generate_combinations()
    print(f"\nAlgorithm Combinations ({len(combinations)} total):")
    for i, combo in enumerate(combinations[:5], 1):  # Show first 5
        community = combo.get('community_detection', 'None')
        node = combo.get('node_measures', 'None')
        edge = combo.get('edge_measures', 'None')
        print(f"  {i}. {community} + {node} + {edge}")
    if len(combinations) > 5:
        print(f"  ... and {len(combinations) - 5} more combinations")

    # Load network
    network = NetworkXWrapper(filepath=config.dataset, format='edgelist')
    print(f"\nNetwork: {network.node_count()} nodes, {network.edge_count()} edges")

    # Run benchmark
    print(f"\nRunning benchmark...")
    benchmark = SimpleBenchmark(config)
    results = benchmark.run(network)

    # Display results summary
    print("\nSample Results (first 3):")
    print("-" * 80)
    results_list = results if isinstance(results, list) else results.results
    for i, result in enumerate(results_list[:3], 1):
        combo_name = f"{result.combination.get('community_detection', 'None')} + " \
                     f"{result.combination.get('node_measures', 'None')} + " \
                     f"{result.combination.get('edge_measures', 'None')}"
        print(f"{i}. {combo_name}")
        print(f"   Time: {result.execution_time:.4f}s")
        if result.quality_scores:
            qual_str = ", ".join([f"{k}: {v:.4f}" for k, v in result.quality_scores.items()])
            print(f"   Quality: {qual_str}")

    # Export results
    export_benchmark_results(results, config)
    print(f"\nResults exported to: {config.output_directory}/")
    print("=" * 80)


def demonstrate_yaml_structure():
    """Example 3: Show what a YAML config looks like"""
    print("\n\nExample 3: YAML Configuration Structure\n")
    print("=" * 80)

    yaml_example = """
# Example: clustering_benchmark.yaml
benchmark:
  name: "My_Clustering_Benchmark"
  dataset: "data/iris.csv"

pipeline_template:
  - type: "clustering"
    algorithms: ["Hierarchical", "DBSCAN"]
    params:
      Hierarchical:
        n_clusters: 3
        linkage: "complete"
        distance_measure: "Euclidean"
      DBSCAN:
        eps: 0.5
        min_samples: 5

  - type: "clustering_quality"
    algorithms: ["Calinski_Harabasz", "Davies_Bouldin"]

iterations: 5
output:
  directory: "benchmark_results"
  format: ["csv", "json"]
"""

    print("YAML Config Structure:")
    print(yaml_example)
    print("This generates combinations like:")
    print("  1. Hierarchical + Calinski_Harabasz")
    print("  2. Hierarchical + Davies_Bouldin")
    print("  3. DBSCAN + Calinski_Harabasz")
    print("  4. DBSCAN + Davies_Bouldin")
    print("\nEach combination is run 5 times and results are averaged.")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("YAML Benchmark Examples")
    print("=" * 80 + "\n")

    # Show YAML structure first
    demonstrate_yaml_structure()

    # Run clustering benchmark
    run_clustering_yaml_benchmark()

    # Run network benchmark
    run_network_yaml_benchmark()

    print("\n" + "=" * 80)
    print("TIP: Use run_benchmark_example.py for command-line execution:")
    print("  python examples/run_benchmark_example.py -c examples/clustering_benchmark.yaml")
    print("  python examples/run_benchmark_example.py -c examples/network_benchmark.yaml")
    print("=" * 80)


if __name__ == "__main__":
    main()
