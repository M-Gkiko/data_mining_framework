# Benchmark Configuration Files

This directory contains sample YAML configuration files for running benchmarks with the Data Mining Framework.

## Available Configurations

### Clustering Benchmarks
- **`clustering_benchmark.yaml`** - Tests clustering algorithms (Hierarchical, DBSCAN) on Iris dataset
  - Includes clustering quality measures (Calinski-Harabasz, Davies-Bouldin)
  - 3 iterations per combination
  - Quick to run (~1-2 minutes)

### Dimensionality Reduction + Clustering
- **`dr_cl_quality.yaml`** - Comprehensive DR + clustering pipeline
  - Tests: PCA, MDS, t-SNE dimensionality reduction
  - Tests: Hierarchical, DBSCAN, K-Means clustering
  - Includes quality measures for both DR and clustering
  - 3 iterations per combination
  - Medium runtime (~5-10 minutes)

### Network Analysis Benchmarks

#### **`network_benchmark.yaml`** - Standard network analysis benchmark
- Dataset: **Zachary's Karate Club** (34 nodes, 78 edges)
- Tests all 3 community detection algorithms (Louvain, Girvan-Newman, Label Propagation)
- Tests all 3 node measures (PageRank, Degree Centrality, Closeness Centrality)
- Tests all 3 edge measures (Edge Betweenness, Edge Weight, Jaccard Coefficient)
- **27 total combinations** (3×3×3)
- 5 iterations per combination
- Runtime: ~5-10 minutes

#### **`network_comprehensive.yaml`** - Comprehensive network benchmark
- Dataset: **Zachary's Karate Club**
- Tests all 27 algorithm combinations
- **10 iterations** per combination for robust statistics
- Saves communities, node measures, and edge measures
- Runtime: ~10-20 minutes
- Best for: Production benchmarking, research papers, detailed analysis

#### **`network_quick_test.yaml`** - Quick network test
- Dataset: **Zachary's Karate Club**
- Tests only 1 combination (Louvain + PageRank + EdgeWeight)
- 3 iterations
- Runtime: < 1 minute
- Best for: Quick testing, development, CI/CD pipelines

#### **`network_les_miserables.yaml`** - Les Miserables character network
- Dataset: **Les Miserables** character co-appearance network
- Tests all 27 algorithm combinations
- 5 iterations per combination
- Community detection set to find 5 communities (main character groups)
- Runtime: ~5-15 minutes
- Best for: Larger network analysis, character relationship studies

#### **`network_three_communities.yaml`** - Synthetic three-community network
- Dataset: **Three Communities** synthetic network with known ground truth
- Tests all 27 algorithm combinations
- 5 iterations per combination
- Community detection set to find 3 communities (matches ground truth)
- Runtime: ~3-8 minutes
- Best for: Algorithm validation, comparing detection accuracy

## Usage

Run any configuration with:

```bash
python run_benchmark.py --config configs/sample_configs/<config_name>.yaml
```

### Clustering & DR Examples:

```bash
# Clustering benchmark (Iris dataset)
python run_benchmark.py --config configs/sample_configs/clustering_benchmark.yaml

# DR + Clustering pipeline (Iris dataset)
python run_benchmark.py --config configs/sample_configs/dr_cl_quality.yaml
```

### Network Analysis Examples:

```bash
# Quick network test (< 1 min)
python run_benchmark.py --config configs/sample_configs/network_quick_test.yaml

# Standard network benchmark (Karate Club)
python run_benchmark.py --config configs/sample_configs/network_benchmark.yaml

# Comprehensive network analysis (Karate Club)
python run_benchmark.py --config configs/sample_configs/network_comprehensive.yaml

# Les Miserables character network
python run_benchmark.py --config configs/sample_configs/network_les_miserables.yaml

# Three communities synthetic network
python run_benchmark.py --config configs/sample_configs/network_three_communities.yaml
```

## Configuration File Format

All configuration files follow this structure:

```yaml
benchmark:
  name: "Benchmark_Name"
  dataset: "path/to/data.csv"  # or .edgelist, .gml, etc. for networks

pipeline_template:
  - type: "algorithm_type"  # clustering, dimensionality_reduction, community_detection, etc.
    algorithms: ["Algorithm1", "Algorithm2"]
    params:
      Algorithm1:
        param1: value1
        param2: value2
      Algorithm2:
        param1: value1

iterations: 5  # Number of times to run each combination

output:
  directory: "benchmark_results"
  format: ["csv", "json"]

timeout: 300
verbose: true
```

## Creating Custom Configurations

To create your own configuration:

1. Copy one of the sample files as a template
2. Modify the `dataset` path to your data
3. Adjust algorithm combinations and parameters
4. Set appropriate iterations and timeout values
5. Run with `python run_benchmark.py --config your_config.yaml`

## Supported Algorithm Types

- **`clustering`** - Hierarchical, DBSCAN, KMeans
- **`dimensionality_reduction`** - PCA, MDS, TSNE, SammonMapping
- **`community_detection`** - Louvain, GirvanNewman, LabelPropagation
- **`node_measure`** - PageRank, DegreeCentrality, ClosenessCentrality
- **`edge_measure`** - EdgeBetweenness, EdgeWeight, JaccardCoefficient
- **`clustering_quality`** - Calinski_Harabasz, Davies_Bouldin, Silhouette
- **`dr_quality`** - Trustworthiness, Continuity, ReconstructionError

## Available Datasets

### Clustering/DR Datasets (in `data/`):
- **`iris.csv`** - Famous Iris flower dataset (150 samples, 4 features, 3 classes)
- **`iris_full.csv`** - Extended Iris dataset with additional features

### Network Datasets (in `data/`):
- **`karate.edgelist`** - Zachary's Karate Club (34 nodes, 78 edges)
  - Classic social network of karate club members
  - 2 known communities (split after instructor dispute)

- **`les_miserables.edgelist`** - Les Miserables character network
  - Character co-appearance network from the novel
  - Multiple character groups/storylines
  - Larger and denser than Karate Club

- **`three_communities.edgelist`** - Synthetic network with 3 communities
  - Designed with clear community structure
  - Ground truth: exactly 3 communities
  - Ideal for algorithm validation

## Dataset Formats

### For Clustering/DR:
- CSV files with numerical features
- Each row is a sample, each column is a feature

### For Network Analysis:
- **Edge lists** (`.edgelist`, `.edges`) - Two columns: source, target
- **NetworkX formats** (`.gml`, `.graphml`, `.gexf`) - Graph markup languages
- **Adjacency matrix** (`.csv`) - Square matrix where entry (i,j) indicates edge

## Output

Results are saved in the specified `output.directory` with:
- CSV files with timing and quality metrics
- JSON files with detailed results (if enabled)
- Separate files for each algorithm combination
- Summary statistics across all runs

## Tips

- Start with quick test configs to verify setup
- Use fewer iterations during development (2-3)
- Use more iterations for production benchmarks (10+)
- Adjust timeout based on dataset size and algorithm complexity
- Enable verbose mode for debugging issues
