# Examples

This directory contains example scripts and configuration files showing how to use the Data Mining Framework.

## Available Examples

### Python Scripts
- **`basic_clustering.py`** - Simple clustering example with quality evaluation
- **`pipeline_example.py`** - Complete pipeline: DR → Clustering → Quality
- **`network_pipeline_example.py`** - Network analysis: Community Detection + Centrality Measures
- **`run_pipeline_example.py`** - Comprehensive demo with 4 pipeline scenarios (clustering + network analysis)
- **`run_benchmark_example.py`** - Benchmark runner with command-line interface

### YAML Configuration Files
- **`clustering_benchmark.yaml`** - Basic clustering benchmark configuration
- **`dr_cl_quality.yaml`** - Complete DR→Clustering→Quality pipeline benchmark
- **`network_benchmark.yaml`** - Network analysis benchmark (community detection + node/edge measures)

## Running Examples

After installing the framework:

```bash
# Install the framework first
pip install -e .

# Run Python examples
cd examples
python basic_clustering.py
python pipeline_example.py
python network_pipeline_example.py
python run_pipeline_example.py

# Run benchmark configs using the benchmark script
python run_benchmark_example.py --config clustering_benchmark.yaml
python run_benchmark_example.py --config dr_cl_quality.yaml
python run_benchmark_example.py --config network_benchmark.yaml --verbose
```

## Using the YAML Configurations

The YAML files demonstrate how to structure benchmark configurations:

1. **Copy and modify** the YAML files for your own datasets
2. **Adjust parameters** like `n_clusters`, `eps`, `n_components` for your data
3. **Change the dataset path** to point to your CSV or network file (use `../data/` prefix when running from examples directory)
4. **Add or remove algorithms** based on your needs

Example modification:
```yaml
benchmark:
  name: "My_Custom_Analysis"
  dataset: "../data/my_data.csv"  # Your dataset here

pipeline_template:
  - type: "clustering"
    algorithms: ["Hierarchical", "DBSCAN"]
    params:
      Hierarchical:
        n_clusters: 5  # Adjust for your data
        distance_measure: "Manhattan"
```

## Available Sample Datasets

The framework includes sample datasets in the `../data/` directory:
- **`iris.csv`** - Classic iris dataset for clustering/DR examples
- **`karate.edgelist`** - Zachary's karate club network for network analysis
- **`les_miserables.edgelist`** - Les Misérables character network
- **`three_communities.edgelist`** - Synthetic network with clear community structure

## Example Details

### basic_clustering.py
Demonstrates:
- Loading CSV dataset
- Hierarchical clustering with Manhattan distance
- Quality evaluation with Calinski-Harabasz Index

### pipeline_example.py
Demonstrates:
- Building a complete pipeline: PCA → Hierarchical Clustering → Quality Evaluation
- Pipeline execution and timing
- Using adapters for different algorithm types

### network_pipeline_example.py
Demonstrates:
- Community detection (Louvain, Girvan-Newman, Label Propagation)
- Node centrality measures (PageRank, Degree Centrality)
- Comparing multiple algorithms on the same network

### run_pipeline_example.py
Comprehensive examples covering:
1. PCA → Hierarchical → Quality evaluation
2. t-SNE → DBSCAN → Quality evaluation
3. Network community detection + node measures
4. Comparing community detection algorithms

### run_benchmark_example.py
Full-featured benchmark runner that:
- Loads YAML configuration files
- Runs multiple algorithm combinations
- Performs multiple iterations for reliability
- Exports results to CSV/JSON
- Provides detailed timing and quality metrics
