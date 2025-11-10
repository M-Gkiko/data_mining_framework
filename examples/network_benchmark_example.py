"""
Network Benchmark Example: Testing All Community + Edge + Node Measure Combinations

This example demonstrates benchmarking network analysis using the programmatic API.
For simpler usage, consider using YAML configuration files (see yaml_benchmark_example.py).

This example tests combinations of:
- Multiple community detection algorithms (Louvain, Girvan-Newman, Label Propagation)
- Multiple edge measures (Betweenness, Weight, Jaccard Coefficient)
- Multiple node measures (PageRank, Degree Centrality, Closeness Centrality)

The benchmark tests all combinations and reports timing and modularity metrics.
"""

from data_mining_framework import (
    NetworkXWrapper,
    load_benchmark_config,
    SimpleBenchmark,
    export_benchmark_results
)


def main():
    print("Network Benchmark Example: All Community + Edge + Node Measure Combinations\n")
    print("=" * 80)

    # Load configuration from YAML file
    # For this example, we use the pre-configured network_benchmark.yaml
    import os
    yaml_path = 'network_benchmark.yaml' if os.path.exists('network_benchmark.yaml') else 'examples/network_benchmark.yaml'
    config = load_benchmark_config(yaml_path)

    print(f"Benchmark: {config.name}")
    print(f"Dataset: {config.dataset}")
    print(f"Iterations: {config.iterations}")

    # Load network
    network = NetworkXWrapper(filepath=config.dataset, format='edgelist')
    print(f"Network loaded: {network.node_count()} nodes, {network.edge_count()} edges\n")

    # Generate and display combinations
    combinations = config.generate_combinations()
    print(f"Testing {len(combinations)} algorithm combinations:")
    for i, combo in enumerate(combinations[:5], 1):  # Show first 5
        community = combo.get('community_detection', 'None')
        node = combo.get('node_measures', 'None')
        edge = combo.get('edge_measures', 'None')
        print(f"  {i}. {community} + {node} + {edge}")
    if len(combinations) > 5:
        print(f"  ... and {len(combinations) - 5} more combinations")

    # Create and run benchmark
    print(f"\nRunning benchmark ({config.iterations} iterations per combination)...\n")
    benchmark = SimpleBenchmark(config)
    results = benchmark.run(network)

    # Display summary
    print("\nBenchmark Results Summary:")
    print("=" * 80)

    for i, result in enumerate(results.results[:5], 1):  # Show top 5
        combo_name = f"{result.combination.get('community_detection', 'None')} + " \
                     f"{result.combination.get('node_measures', 'None')} + " \
                     f"{result.combination.get('edge_measures', 'None')}"
        print(f"\n{i}. {combo_name}:")
        print(f"   Average Time: {result.avg_time:.4f}s")
        if hasattr(result, 'modularity') and result.modularity is not None:
            print(f"   Modularity: {result.modularity:.4f}")
        print(f"   Std Dev: {result.std_time:.4f}s")

    # Export results
    export_benchmark_results(results, config)
    print(f"\n{'=' * 80}")
    print(f"Full results exported to: {config.output_directory}/")
    print(f"  - CSV format: timing and modularity data")
    print(f"  - JSON format: complete results with communities and centrality scores")
    print(f"\nTIP: You can also run this benchmark using:")
    print(f"  python examples/run_benchmark_example.py -c examples/network_benchmark.yaml")


if __name__ == "__main__":
    main()
