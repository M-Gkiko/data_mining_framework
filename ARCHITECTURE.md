# Data Mining Framework Architecture

## Design Philosophy

The framework uses the **Strategy Pattern** to provide flexible, interchangeable algorithm implementations. Each component type (clustering, distance measures, quality metrics, etc.) shares a common interface, allowing algorithms to be swapped without changing client code.

## Core Abstractions

### Base Interfaces

All algorithms inherit from abstract base classes that define standard contracts:

```python
# Data Sources
class Dataset(ABC):
    def get_data() -> np.ndarray
    def get_features() -> List[str]
    def shape() -> Tuple[int, int]

# Distance Calculations
class DistanceMeasure(ABC):
    def calculate(point1, point2) -> float

# Clustering
class ClusteringAlgorithm(ABC):
    def fit(dataset, **kwargs) -> None
    def get_labels() -> List[int]

# Dimensionality Reduction
class DimensionalityReduction(ABC):
    def fit_transform(dataset, **kwargs) -> np.ndarray

# Network Structures
class Network(ABC):
    def get_nodes() -> List[Any]
    def get_edges() -> List[Tuple[Any, Any]]
    def node_count() -> int
    def edge_count() -> int

# Community Detection
class CommunityDetection(ABC):
    def fit(network, **kwargs) -> None
    def get_communities() -> List[Set]
    def get_modularity() -> float

# Quality Metrics
class ClusteringQualityMeasure(ABC):
    def evaluate(dataset, labels) -> float

class DRQualityMeasure(ABC):
    def evaluate(original_data, projected_data, **kwargs) -> float

class NodeMeasure(ABC):
    def calculate(network, **kwargs) -> Dict[Any, float]

class EdgeMeasure(ABC):
    def calculate(network, **kwargs) -> Dict[Tuple, float]
```

## Component Types

### 1. Data Handling

**Implementations:**
- `CSVDataset` - Load data from CSV files
- `NumpyDataset` - Wrap numpy arrays

**Network Representations:**
- `NetworkXWrapper` - NetworkX graph adapter
- `EdgeListNetwork` - Edge list format
- `AdjacencyMatrixNetwork` - Matrix format

### 2. Distance Measures

- `ManhattanDistance` - L1 norm (city block)
- `EuclideanDistance` - L2 norm
- `CosineDistance` - Angular distance

### 3. Clustering Algorithms

- `HierarchicalClustering` - Agglomerative clustering with linkage methods
- `DBSCAN` - Density-based spatial clustering
- `KMeans` - K-means partitioning

### 4. Dimensionality Reduction

- `PCAProjection` - Principal Component Analysis
- `MDSProjection` - Multidimensional Scaling
- `TSNEProjection` - t-SNE embedding
- `SammonMapping` - Sammon projection (MATLAB port)

### 5. Network Analysis

**Community Detection:**
- `LouvainCommunityDetection` - Modularity optimization
- `GirvanNewmanCommunityDetection` - Edge betweenness-based
- `LabelPropagationCommunityDetection` - Label propagation method

**Node Centrality:**
- `PageRank` - PageRank centrality
- `DegreeCentrality` - Degree-based importance
- `ClosenessCentrality` - Distance-based centrality

**Edge Metrics:**
- `EdgeBetweenness` - Edge betweenness centrality
- `EdgeWeight` - Edge weight extraction
- `JaccardCoefficient` - Neighborhood similarity

### 6. Quality Measures

**Clustering Quality:**
- `CalinskiHarabaszIndex` - Variance ratio criterion
- `DaviesBouldinIndex` - Cluster separation measure
- `SilhouetteScore` - Silhouette coefficient

**DR Quality:**
- `Trustworthiness` - Local structure preservation
- `Continuity` - Neighborhood preservation
- `ReconstructionError` - Reconstruction accuracy

## Pipeline Architecture

### Pipeline Components

Pipelines chain algorithms using **adapter classes** that standardize inputs/outputs:

```python
class Pipeline:
    def __init__(name: str)
    def add_component(component: PipelineComponent)
    def execute(input_data) -> Any
```

### Adapters

Each algorithm type has a corresponding adapter:

**Clustering Pipelines:**
- `DRAdapter` - Wraps dimensionality reduction algorithms
- `ClusteringAdapter` - Wraps clustering algorithms
- `ClusteringQualityAdapter` - Wraps clustering quality measures
- `DRQualityAdapter` - Wraps DR quality measures

**Network Pipelines:**
- `NetworkAdapter` - Wraps network data structures
- `CommunityDetectionAdapter` - Wraps community detection algorithms
- `NodeMeasureAdapter` - Wraps node centrality measures
- `EdgeMeasureAdapter` - Wraps edge measures

### Pipeline Flow

**Clustering Pipeline:**
```
Dataset → DRAdapter → ClusteringAdapter → ClusteringQualityAdapter → Results
```

**Network Analysis Pipeline:**
```
Network → NetworkAdapter → CommunityDetectionAdapter → EdgeMeasureAdapter → NodeMeasureAdapter → Results
```

Adapters handle:
- Input/output format conversions
- Passing results between stages (accumulates all results)
- Error handling and validation
- Type checking for proper data flow

## Benchmarking System

### Components

**Core (`benchmarks/core.py`):**
- `BenchmarkRunner` - Executes benchmark configurations
- `BenchmarkResult` - Stores individual run results
- Pydantic models for configuration validation

**Registry (`benchmarks/registry.py`):**
- Algorithm registration and lazy loading
- Factory functions for creating instances
- Distance measure creation

**Utilities (`benchmarks/utils.py`):**
- YAML configuration loading
- Result export (CSV, JSON)
- Configuration validation

### Benchmark Execution Flow

```
1. Load YAML config → Parse with Pydantic
2. Create algorithm combinations → Generate pipeline permutations
3. For each iteration:
   - Build pipeline from template
   - Execute with timing
   - Collect results
4. Export results → CSV/JSON files
```

## Key Design Benefits

### 1. Extensibility
Add new algorithms by implementing the appropriate base class:

```python
class MyClusteringAlgorithm(ClusteringAlgorithm):
    def fit(self, dataset, **kwargs):
        # Implementation

    def get_labels(self):
        # Return cluster labels
```

Register in `benchmarks/registry.py`:

```python
ALGORITHMS['clustering']['MyAlgorithm'] = 'path.to.MyClusteringAlgorithm'
```

### 2. Composability
Chain algorithms in pipelines:

**Clustering Pipeline:**
```python
pipeline.add_component(DRAdapter(pca))
pipeline.add_component(ClusteringAdapter(kmeans, distance))
pipeline.add_component(ClusteringQualityAdapter(silhouette))
results = pipeline.execute(dataset)
```

**Network Pipeline:**
```python
pipeline.add_component(NetworkAdapter(network))
pipeline.add_component(CommunityDetectionAdapter(louvain))
pipeline.add_component(EdgeMeasureAdapter(betweenness))
pipeline.add_component(NodeMeasureAdapter(pagerank))
results = pipeline.execute(None)
# results = {'communities': ..., 'modularity': ..., 'edge_scores': ..., 'node_scores': ...}
```

### 3. Configurability
Define complex experiments in YAML:

```yaml
pipeline_template:
  - type: "dimensionality_reduction"
    algorithms: ["PCA", "TSNE"]
  - type: "clustering"
    algorithms: ["Hierarchical", "DBSCAN"]
  - type: "clustering_quality"
    algorithms: ["Silhouette"]
```

## Directory Structure

```
data_mining_framework/
├── core/                           # Abstract base classes (interfaces)
│   ├── clustering.py
│   ├── community_detection.py
│   ├── dataset.py
│   ├── dimensionality_reduction.py
│   ├── distance_measure.py
│   ├── edge_measure.py
│   ├── network.py
│   ├── node_measure.py
│   ├── pipeline.py
│   └── *_quality_measure.py
│
├── implementations/                # Concrete implementations
│   ├── clustering/                # Hierarchical, DBSCAN, KMeans
│   ├── community_detection/       # Louvain, GirvanNewman, etc.
│   ├── datasets.py                # CSVDataset, NumpyDataset
│   ├── distance/                  # Manhattan, Euclidean, Cosine
│   ├── dr/                        # PCA, MDS, TSNE, Sammon
│   │   └── quality/              # DR quality measures
│   ├── edge_measures/            # Edge betweenness, weight, Jaccard
│   ├── networks/                 # NetworkX wrapper, edge list, matrix
│   ├── node_measures/            # PageRank, centrality measures
│   ├── pipelines/                # Adapters for pipeline components
│   └── quality/                  # Clustering quality measures
│
├── benchmarks/                    # Benchmarking system
│   ├── core.py                   # Benchmark execution engine
│   ├── registry.py               # Algorithm registry and factories
│   └── utils.py                  # Config loading and export
│
├── utils/                        # Utilities
│   ├── distance_utils.py         # Distance matrix computation
│   └── timer.py                  # Performance timing
│
└── examples/                     # Example scripts and configs
    ├── README.md                # Detailed example documentation
    ├── *.yaml                   # Benchmark configurations
    ├── run_benchmark_example.py # CLI benchmark runner
    ├── yaml_benchmark_example.py # YAML usage tutorial
    ├── clustering_pipeline_example.py    # Clustering workflow
    ├── network_pipeline_example.py       # Network workflow
    ├── clustering_benchmark_example.py   # Compare clustering algorithms
    ├── network_benchmark_example.py      # Compare network algorithms
    ├── basic_component_usage.py # Direct API usage (no pipelines)
    └── *.py                     # Other examples
```

## Implementation Guidelines

### Adding a New Algorithm

1. **Create implementation** in appropriate subdirectory
2. **Inherit from base class** and implement required methods
3. **Register in registry** (`benchmarks/registry.py`)
4. **Add to exports** (`implementations/__init__.py`)
5. **Create adapter** if needed (for pipelines)

### Adding a New Pipeline Step Type

1. **Define base class** in `core/`
2. **Create adapter** in `implementations/pipelines/`
3. **Update registry** with factory function
4. **Add to benchmark core** step type handling
5. **Document in README** YAML configuration

## Dependencies

**Core:**
- numpy - Array operations
- pandas - Data handling
- scikit-learn - ML algorithms
- scipy - Scientific computing
- networkx - Network analysis

**Benchmarking:**
- pydantic - Configuration validation
- PyYAML - Config file parsing

**Development:**
- build - Package building

## Performance Considerations

- **Lazy loading** in registry reduces startup time
- **Distance matrix caching** avoids recomputation
- **Pipeline results** passed through adapters to minimize copies
- **Pydantic validation** catches config errors early

## Examples and Usage

For practical examples demonstrating these architectural patterns:

- 📖 **[`examples/README.md`](examples/README.md)** - Complete guide to all examples
- 🔧 **`basic_component_usage.py`** - Direct component usage
- 🔗 **`clustering_pipeline_example.py`** - Pipeline pattern for clustering
- 🌐 **`network_pipeline_example.py`** - Pipeline pattern for networks
- ⚖️ **Benchmark examples** - Algorithm comparison workflows
- 📝 **YAML configs** - Declarative configuration approach

See the main [`README.md`](README.md) for quick start instructions.
