# Data Mining Framework

A comprehensive framework for benchmarking data mining algorithms, focusing on clustering and dimensionality reduction techniques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔬 **Comprehensive Algorithm Support**: Clustering (Hierarchical, DBSCAN), Dimensionality Reduction (PCA, MDS, t-SNE)
- 📊 **Quality Measures**: Built-in evaluation metrics for both clustering and DR results
- 🎯 **Flexible Distance Metrics**: Support for Manhattan, Euclidean, and Cosine distances
- 🚀 **Pipeline Architecture**: Chain algorithms together (DR → Clustering → Quality evaluation)
- 📈 **Benchmarking System**: Automated performance and quality benchmarking with CSV export
- 🔧 **Extensible Design**: Easy to add new algorithms, distance measures, and quality metrics

## Installation

### From PyPI (when published)
```bash
pip install data-mining-framework
```

### From Source
```bash
git clone https://github.com/M-Gkiko/data_mining_framework.git
cd data_mining_framework
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/M-Gkiko/data_mining_framework.git
cd data_mining_framework
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface
```bash
# Run a benchmark from configuration file
dm-benchmark examples/clustering_benchmark.yaml

# Run with verbose output
dm-benchmark examples/dr_cl_quality.yaml --verbose

# List available example configurations
dm-benchmark --list-configs
```

### Python API

#### Simple Clustering Example
```python
from data_mining_framework import CSVDataset, HierarchicalClustering, ManhattanDistance

# Load data
dataset = CSVDataset('data/iris.csv')

# Create distance measure and clustering algorithm
distance_measure = ManhattanDistance()
clustering = HierarchicalClustering(
    distance_measure=distance_measure,
    n_clusters=3,
    linkage='complete'
)

# Fit and get results
clustering.fit(dataset)
labels = clustering.get_labels()
print(f"Cluster labels: {labels}")
```

#### Complete Pipeline Example
```python
from data_mining_framework import (
    CSVDataset, PCAProjection, HierarchicalClustering, 
    CalinskiHarabaszIndex, ManhattanDistance, Pipeline
)

# Load dataset
dataset = CSVDataset('data/iris.csv')
distance_measure = ManhattanDistance()

# Create pipeline: DR → Clustering → Quality
pipeline = Pipeline("PCA_Hierarchical_Quality")

# Add dimensionality reduction
pca = PCAProjection(n_components=2)
dr_adapter = DRAdapter(pca)
pipeline.add_component(dr_adapter)

# Add clustering
clustering = HierarchicalClustering(
    distance_measure=distance_measure,
    n_clusters=3
)
clustering_adapter = ClusteringAdapter(clustering, distance_measure)
pipeline.add_component(clustering_adapter)

# Add quality evaluation
quality_measure = CalinskiHarabaszIndex()
quality_adapter = ClusteringQualityAdapter(quality_measure)
pipeline.add_component(quality_adapter)

# Execute pipeline
results = pipeline.execute(dataset)
print(f"Quality score: {results}")
```

#### Benchmark from Python
```python
from data_mining_framework import run_benchmark

# Run benchmark and get results
results = run_benchmark('examples/dr_cl_quality.yaml')
print(f"Benchmark completed: {results.total_runs} runs")
print(f"Average execution time: {results.average_time:.3f}s")
```

## Configuration Files

The framework uses YAML configuration files to define benchmarks:

```yaml
benchmark:
  name: "My_Benchmark"
  dataset: "data/iris.csv"

pipeline_template:
  - type: "dimensionality_reduction"
    algorithms: ["PCA", "MDS", "TSNE"]
    params:
      PCA:
        n_components: 2
      MDS:
        n_components: 2
        distance_measure: "Manhattan"
      TSNE:
        n_components: 2
        perplexity: 30
        distance_measure: "Manhattan"

  - type: "clustering"
    algorithms: ["Hierarchical", "DBSCAN"]
    params:
      Hierarchical:
        n_clusters: 3
        linkage: "complete"
        distance_measure: "Manhattan"
      DBSCAN:
        eps: 0.6
        min_samples: 4
        distance_measure: "Manhattan"

  - type: "clustering_quality"
    algorithms: ["Calinski_Harabasz", "Davies_Bouldin"]

iterations: 3
output:
  directory: "benchmark_results"
  format: ["csv"]
```

## Available Algorithms

### Clustering Algorithms
- **Hierarchical Clustering**: Agglomerative clustering with various linkage criteria
- **DBSCAN**: Density-based clustering for discovering clusters of arbitrary shape

### Dimensionality Reduction
- **PCA**: Principal Component Analysis for linear dimensionality reduction
- **MDS**: Multidimensional Scaling for preserving distances
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for non-linear reduction

### Distance Measures
- **Manhattan Distance**: L1 norm distance
- **Euclidean Distance**: L2 norm distance (when available)
- **Cosine Distance**: Angular distance measure (when available)

### Quality Measures
- **Clustering Quality**: Calinski-Harabasz Index, Davies-Bouldin Index
- **DR Quality**: Trustworthiness, Continuity, Reconstruction Error

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

### Building Package
```bash
python -m build
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


```