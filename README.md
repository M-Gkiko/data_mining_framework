# Data Mining Framework

A flexible framework for benchmarking data mining algorithms including clustering, dimensionality reduction, and network analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Installation

### From TestPyPI (https://test.pypi.org/project/data-mining-framework/)_

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ data-mining-framework
```

Note: The `--extra-index-url` ensures dependencies are installed from the main PyPI repository.

### From Source (Development)

```bash
git clone https://github.com/M-Gkiko/data_mining_framework.git
cd data_mining_framework
pip install -e .

# With development dependencies
pip install -e ".[dev]" 
pip install -r requirements.txt # Alternative 
```

## Quick Start

### Running Benchmarks

```bash
# Run a benchmark from YAML configuration
python examples/run_benchmark_example.py --config examples/clustering_benchmark.yaml

# Run with verbose output
python examples/run_benchmark_example.py --config examples/dr_cl_quality.yaml --verbose

# Network analysis benchmark
python examples/run_benchmark_example.py --config examples/network_benchmark.yaml
```

### Python API Examples

#### Simple Clustering

```python
from data_mining_framework import CSVDataset, HierarchicalClustering, ManhattanDistance

dataset = CSVDataset('data/iris.csv')
distance = ManhattanDistance()
clustering = HierarchicalClustering(distance_measure=distance, n_clusters=3, linkage='complete')

clustering.fit(dataset)
labels = clustering.get_labels()
```

#### Pipeline: DR → Clustering → Quality

```python
from data_mining_framework import (
    CSVDataset, PCAProjection, HierarchicalClustering,
    CalinskiHarabaszIndex, ManhattanDistance, Pipeline
)
from data_mining_framework.implementations.pipelines import (
    DRAdapter, ClusteringAdapter, ClusteringQualityAdapter
)

dataset = CSVDataset('data/iris.csv')
distance = ManhattanDistance()

pipeline = Pipeline("PCA_Hierarchical_Quality")

# Add dimensionality reduction
pca = PCAProjection(n_components=2)
pipeline.add_component(DRAdapter(pca))

# Add clustering
clustering = HierarchicalClustering(distance_measure=distance, n_clusters=3)
pipeline.add_component(ClusteringAdapter(clustering, distance))

# Add quality measure
quality = CalinskiHarabaszIndex()
pipeline.add_component(ClusteringQualityAdapter(quality))

results = pipeline.execute(dataset)
```

#### Network Analysis Pipeline

```python
from data_mining_framework import (
    NetworkXWrapper, LouvainCommunityDetection,
    PageRankMeasure, EdgeBetweennessMeasure,
    Pipeline, NetworkAdapter, CommunityDetectionAdapter,
    NodeMeasureAdapter, EdgeMeasureAdapter
)

# Create network
network = NetworkXWrapper(filepath='data/karate.edgelist', format='edgelist')

# Build pipeline: Community Detection → Edge Measures → Node Measures
pipeline = Pipeline("Network_Analysis")
pipeline.add_component(NetworkAdapter(network))
pipeline.add_component(CommunityDetectionAdapter(LouvainCommunityDetection(resolution=1.0)))
pipeline.add_component(EdgeMeasureAdapter(EdgeBetweennessMeasure()))
pipeline.add_component(NodeMeasureAdapter(PageRankMeasure(alpha=0.85)))

# Execute and get all results
results = pipeline.execute(None)
# Results contain: communities, modularity, edge_scores, node_scores
```

#### Run Benchmarks from Python

```python
from data_mining_framework.benchmarks import run_benchmark

results = run_benchmark('examples/dr_cl_quality.yaml')
print(f"Completed {results.total_runs} runs")
print(f"Average time: {results.average_time:.3f}s")
```

## Running Example Scripts

The `examples/` directory contains organized examples demonstrating different usage patterns. All examples can run from the project root directory.

📖 **For detailed documentation of all examples, see [`examples/README.md`](examples/README.md)**

### Quick Example Overview

#### 1. Basic Component Usage (Learning)

Direct component usage without pipelines - great for learning the API.

```bash
python examples/basic_component_usage.py
```

Shows both clustering and network analysis workflows step-by-step.

#### 2. Pipeline Examples (Single Analysis)

Complete workflows using the Pipeline framework.

```bash
# Clustering: DR → Clustering → Quality
python examples/clustering_pipeline_example.py

# Network: Community → Edge → Node measures
python examples/network_pipeline_example.py
```

#### 3. Benchmark Examples (Compare Algorithms)

Test multiple algorithm combinations with timing metrics.

```bash
# From examples/ directory (recommended)
cd examples
python clustering_benchmark_example.py  # Tests 4 combinations
python network_benchmark_example.py     # Tests 27 combinations

# Or from project root
python examples/clustering_benchmark_example.py
python examples/network_benchmark_example.py
```

#### 4. YAML Configuration Examples

Run benchmarks from YAML configuration files.

```bash
# Programmatic YAML usage tutorial
cd examples
python yaml_benchmark_example.py

# Command-line YAML benchmark runner
python run_benchmark_example.py -c clustering_benchmark.yaml
python run_benchmark_example.py -c network_benchmark.yaml --verbose
```

### Example Categories

| Category | Files | Purpose |
|----------|-------|---------|
| **Learning** | `basic_component_usage.py` | Understand the API |
| **Pipelines** | `clustering_pipeline_example.py`<br>`network_pipeline_example.py` | Single workflow execution |
| **Benchmarks** | `clustering_benchmark_example.py`<br>`network_benchmark_example.py` | Compare algorithms |
| **YAML Configs** | `yaml_benchmark_example.py`<br>`run_benchmark_example.py` | Configuration-based execution |

### Troubleshooting

**Import errors:** Make sure you've installed the package:
```bash
pip install -e .
```

**Path issues when running benchmarks:** Some examples work best from the `examples/` directory:
```bash
cd examples
python clustering_benchmark_example.py
```

**Missing dependencies:**
```bash
pip install -e ".[dev]"
```

### Next Steps

- 📖 Read [`examples/README.md`](examples/README.md) for detailed descriptions
- 📝 See YAML configuration examples in `examples/*.yaml`
- 🏗️ Check [`ARCHITECTURE.md`](ARCHITECTURE.md) for framework design

## YAML Configuration Reference

### Basic Structure

```yaml
benchmark:
  name: "My_Benchmark"
  dataset: "path/to/data.csv"

pipeline_template:
  - type: "step_type"
    algorithms: ["Algorithm1", "Algorithm2"]
    params:
      Algorithm1:
        param1: value1
      Algorithm2:
        param2: value2

iterations: 3
output:
  directory: "results"
  format: ["csv"]
```

### Pipeline Step Types

#### Dimensionality Reduction (`dimensionality_reduction`)

**Algorithms:** `PCA`, `MDS`, `TSNE`, `Sammon`

```yaml
- type: "dimensionality_reduction"
  algorithms: ["PCA", "MDS", "TSNE", "Sammon"]
  params:
    PCA:
      n_components: 2
    MDS:
      n_components: 2
      max_iter: 300
      distance_measure: "Manhattan"  # Optional: Manhattan, Euclidean, Cosine
    TSNE:
      n_components: 2
      perplexity: 30
      max_iter: 1000
      distance_measure: "Manhattan"
    Sammon:
      n_components: 2
      max_iter: 500
      distance_measure: "Manhattan"
      init: "pca"  # or "random"
```

#### Clustering (`clustering`)

**Algorithms:** `Hierarchical`, `DBSCAN`, `KMeans`

```yaml
- type: "clustering"
  algorithms: ["Hierarchical", "DBSCAN", "KMeans"]
  params:
    Hierarchical:
      n_clusters: 3
      linkage: "complete"  # complete, average, single, ward
      distance_measure: "Manhattan"
    DBSCAN:
      eps: 0.5
      min_samples: 5
      distance_measure: "Euclidean"
    KMeans:
      n_clusters: 3
      max_iter: 300
      n_init: 10
```

#### Clustering Quality (`clustering_quality`)

**Algorithms:** `Calinski_Harabasz`, `Davies_Bouldin`, `Silhouette`

```yaml
- type: "clustering_quality"
  algorithms: ["Calinski_Harabasz", "Davies_Bouldin", "Silhouette"]
  params:
    Calinski_Harabasz: {}
    Davies_Bouldin: {}
    Silhouette: {}
```

#### DR Quality (`dr_quality`)

**Algorithms:** `Trustworthiness`, `Continuity`, `Reconstruction_Error`

```yaml
- type: "dr_quality"
  algorithms: ["Trustworthiness", "Continuity", "Reconstruction_Error"]
  params:
    Trustworthiness:
      n_neighbors: 12
    Continuity:
      n_neighbors: 12
    Reconstruction_Error: {}
```

#### Community Detection (`community_detection`)

**Algorithms:** `Louvain`, `GirvanNewman`, `LabelPropagation`

```yaml
- type: "community_detection"
  algorithms: ["Louvain", "GirvanNewman", "LabelPropagation"]
  params:
    Louvain:
      resolution: 1.0
      random_state: 42
    GirvanNewman:
      k: 2  # Number of communities
    LabelPropagation:
      max_iterations: 100
      random_seed: 42
```

#### Node Measures (`node_measures`)

**Algorithms:** `PageRank`, `DegreeCentrality`, `ClosenessCentrality`

```yaml
- type: "node_measures"
  algorithms: ["PageRank", "DegreeCentrality", "ClosenessCentrality"]
  params:
    PageRank:
      alpha: 0.85
      max_iter: 100
      tol: 0.000001
    DegreeCentrality:
      normalized: true
    ClosenessCentrality:
      normalized: true
```

#### Edge Measures (`edge_measures`)

**Algorithms:** `EdgeBetweenness`, `EdgeWeight`, `JaccardCoefficient`

```yaml
- type: "edge_measures"
  algorithms: ["EdgeBetweenness", "EdgeWeight", "JaccardCoefficient"]
  params:
    EdgeBetweenness:
      normalized: true
    EdgeWeight:
      weight_attribute: "weight"
      default_weight: 1.0
    JaccardCoefficient: {}
```

### Global Configuration Options

```yaml
benchmark:
  name: "Benchmark_Name"      # Benchmark identifier
  dataset: "data/file.csv"    # Path to dataset

iterations: 3                 # Number of runs per configuration

output:
  directory: "results"        # Output directory
  format: ["csv", "json"]     # Output formats
  save_communities: true      # Save community results (network only)
  save_centralities: true     # Save centrality scores (network only)

timeout: 300                  # Timeout per run in seconds
verbose: true                 # Enable detailed logging
random_seed: 42               # Random seed for reproducibility
```

### Distance Measures

Available distance measures: `Manhattan`, `Euclidean`, `Cosine`

Use in algorithm params:
```yaml
params:
  AlgorithmName:
    distance_measure: "Manhattan"
```

## Available Implementations

### Clustering
- `Hierarchical` - Agglomerative clustering (linkage: complete, average, single, ward)
- `DBSCAN` - Density-based clustering
- `KMeans` - K-means clustering

### Dimensionality Reduction
- `PCA` - Principal Component Analysis
- `MDS` - Multidimensional Scaling
- `TSNE` - t-Distributed Stochastic Neighbor Embedding
- `Sammon` - Sammon Mapping

### Network Analysis
- **Community Detection:** Louvain, Girvan-Newman, Label Propagation
- **Node Measures:** PageRank, Degree Centrality, Closeness Centrality
- **Edge Measures:** Edge Betweenness, Edge Weight, Jaccard Coefficient

### Quality Measures
- **Clustering:** Calinski-Harabasz Index, Davies-Bouldin Index, Silhouette Score
- **DR:** Trustworthiness, Continuity, Reconstruction Error

## Example Configurations

The `examples/` directory contains both YAML configs and Python examples:

**YAML Configs:**
- `clustering_benchmark.yaml` - Clustering algorithm comparisons
- `dr_cl_quality.yaml` - Complete DR + Clustering + Quality pipeline
- `network_benchmark.yaml` - Network analysis with all combinations

**Python Examples:**
- `basic_component_usage.py` - Learn the API without pipelines
- `clustering_pipeline_example.py` - Complete clustering workflow
- `network_pipeline_example.py` - Chained network analysis
- `clustering_benchmark_example.py` - Benchmark all clustering combinations
- `network_benchmark_example.py` - Benchmark all network combinations
- `yaml_benchmark_example.py` - YAML usage tutorial
- `run_benchmark_example.py` - Command-line benchmark runner

📖 **See [`examples/README.md`](examples/README.md) for detailed documentation**

## Project Structure

```
data_mining_framework/
├── core/                    # Abstract base classes
├── implementations/         # Algorithm implementations
│   ├── clustering/
│   ├── dr/
│   ├── networks/
│   ├── community_detection/
│   ├── node_measures/
│   ├── edge_measures/
│   └── pipelines/          # Pipeline adapters
├── benchmarks/             # Benchmarking system
├── utils/                  # Utilities
├── examples/               # Usage examples and configs
└── data/                   # Sample datasets
```
