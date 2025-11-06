"""
Benchmarking Example

This example demonstrates how to use the framework's benchmarking capabilities
to compare different algorithm combinations and analyze their performance.
"""

import os
from pathlib import Path
from data_mining_framework import run_benchmark


def run_clustering_benchmark():
    """Run a basic clustering benchmark comparing different algorithms."""
    print("=== Clustering Benchmark Example ===")
    print("This benchmark compares Hierarchical and DBSCAN clustering algorithms")
    print("using different distance measures and quality metrics.\n")
    
    # Get the path to the example config
    example_dir = Path(__file__).parent
    config_path = example_dir / "clustering_benchmark.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Make sure clustering_benchmark.yaml is in the examples directory.")
        return
    
    try:
        print(f"📁 Running benchmark from: {config_path}")
        print("=" * 60)
        
        # Run the benchmark with verbose output
        results = run_benchmark(str(config_path), verbose=True)
        
        print("\n" + "=" * 60)
        print("🎉 Benchmark Analysis Complete!")
        print("\nKey Takeaways:")
        print("- Compare execution times between algorithms")
        print("- Analyze quality scores (higher Calinski-Harabasz = better)")
        print("- Davies-Bouldin scores (lower = better)")
        print("- Results exported to CSV for further analysis")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def run_dr_clustering_benchmark():
    """Run a complete DR + Clustering + Quality benchmark."""
    print("\n" + "=" * 80)
    print("=== Dimensionality Reduction + Clustering Benchmark ===")
    print("This benchmark tests complete pipelines: DR → Clustering → Quality")
    print("Comparing PCA, MDS, and t-SNE with different clustering methods.\n")
    
    # Get the path to the DR+Clustering config
    example_dir = Path(__file__).parent
    config_path = example_dir / "dr_cl_quality.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Make sure dr_cl_quality.yaml is in the examples directory.")
        return
    
    try:
        print(f"📁 Running DR+Clustering benchmark from: {config_path}")
        print("=" * 60)
        
        # Run the benchmark
        results = run_benchmark(str(config_path), verbose=True)
        
        print("\n" + "=" * 60)
        print("🎉 DR+Clustering Benchmark Complete!")
        print("\nAnalysis Insights:")
        print("- PCA vs MDS vs t-SNE: Which preserves clustering structure best?")
        print("- How does dimensionality reduction affect clustering quality?")
        print("- Performance trade-offs: Speed vs Quality")
        print("- Check the CSV results for detailed metrics")
        
    except Exception as e:
        print(f"❌ DR+Clustering benchmark failed: {e}")


def create_custom_benchmark_config():
    """Show how to create a custom benchmark configuration."""
    print("\n" + "=" * 80)
    print("=== Creating Custom Benchmark Configurations ===")
    
    custom_config = """
# Custom Benchmark Example
benchmark:
  name: "My_Custom_Benchmark"
  dataset: "your_dataset.csv"  # Replace with your data file

pipeline_template:
  # Test only clustering (no DR)
  - type: "clustering"
    algorithms: ["Hierarchical"]
    params:
      Hierarchical:
        n_clusters: 4  # Adjust for your data
        linkage: "ward"
        distance_measure: "Manhattan"

  # Evaluate clustering quality
  - type: "clustering_quality"
    algorithms: ["Calinski_Harabasz", "Davies_Bouldin"]
    params:
      Calinski_Harabasz: {}
      Davies_Bouldin: {}

# Run settings
iterations: 5
output:
  directory: "my_results"
  format: ["csv"]
timeout: 120
verbose: true
"""
    
    print("Here's an example of a custom benchmark configuration:")
    print("=" * 40)
    print(custom_config)
    print("=" * 40)
    print("To use this:")
    print("1. Save it as 'my_benchmark.yaml'")
    print("2. Replace 'your_dataset.csv' with your data file")
    print("3. Adjust n_clusters for your dataset")
    print("4. Run: dm-benchmark my_benchmark.yaml")


def main():
    """Run all benchmark examples."""
    print("🚀 Data Mining Framework - Benchmark Examples")
    print("=" * 80)
    
    # Run basic clustering benchmark
    run_clustering_benchmark()
    
    # Run DR + Clustering benchmark
    run_dr_clustering_benchmark()
    
    # Show custom config example
    create_custom_benchmark_config()
    
    print("\n" + "=" * 80)
    print("✅ All Examples Complete!")
    print("\nNext Steps:")
    print("- Check the generated CSV files in benchmark_results/")
    print("- Try creating your own benchmark configurations")
    print("- Experiment with different algorithms and parameters")
    print("- Use your own datasets for real-world analysis")


if __name__ == "__main__":
    main()