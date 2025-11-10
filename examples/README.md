# Data Mining Framework Examples

This directory contains example scripts demonstrating different ways to use the framework. Examples are organized by usage pattern: pipelines, benchmarks, YAML configs, and direct component usage.

## Quick Start Guide

### For Beginners: Start Here!
- **basic_component_usage.py** - Learn component basics without pipelines
- **clustering_pipeline_example.py** - See how pipelines chain operations

### For Production Use
- **yaml_benchmark_example.py** - Run benchmarks from YAML configs
- **run_benchmark_example.py** - Command-line benchmark runner

---

## Example Files Overview

### 1. Pipeline Examples (Single Execution)
Chain operations using the Pipeline framework for single-run workflows.

| File | Description | Use Case |
|------|-------------|----------|
| **clustering_pipeline_example.py** | PCA → Hierarchical Clustering → Quality | Run one complete clustering workflow |
| **network_pipeline_example.py** | Community Detection → Edge Measures → Node Measures | Comprehensive network analysis in one go |

### 2. Benchmark Examples (Multiple Combinations)
Test and compare multiple algorithm combinations with timing metrics.

| File | Description | Combinations |
|------|-------------|--------------|
| **clustering_benchmark_example.py** | Tests all DR + Clustering + Quality combinations | 18 combinations (3×3×2) |
| **network_benchmark_example.py** | Tests all Community + Edge + Node combinations | 27 combinations (3×3×3) |

### 3. YAML Configuration Examples
Use YAML files for reproducible, shareable configurations.

| File | Description | Use Case |
|------|-------------|----------|
| **yaml_benchmark_example.py** | Demonstrates YAML-based benchmarks programmatically | Learn YAML configs |
| **run_benchmark_example.py** | Command-line YAML benchmark runner | Production benchmark execution |

### 4. Direct Component Usage

| File | Description | Use Case |
|------|-------------|----------|
| **basic_component_usage.py** | Direct component calls (no pipelines) | Learning, debugging, fine control |

---

## Detailed Example Descriptions

### clustering_pipeline_example.py
**Pipeline for Clustering + Quality Evaluation**

```bash
python examples/clustering_pipeline_example.py
```

**What it does:**
- Loads Iris dataset (150 samples, 4 features)
- Applies PCA dimensionality reduction → 2 dimensions
- Performs Hierarchical clustering → 3 clusters
- Evaluates with Calinski-Harabasz quality measure
- Reports execution times for each stage

**Output example:**
```
Pipeline Results:
  Calinski-Harabasz Score: 629.6640 (higher is better)

Execution Times:
  DRAdapter_PCAProjection: 0.0012s
  ClusteringAdapter_HierarchicalClustering: 0.1205s
  Total: 0.1221s
```

---

### network_pipeline_example.py
**Pipeline for Network Analysis**

```bash
python examples/network_pipeline_example.py
```

**What it does:**
- Loads Karate Club network (34 nodes, 78 edges)
- Detects communities with Louvain algorithm
- Calculates edge betweenness centrality
- Calculates node PageRank scores
- All results accumulated and returned together

**Output example:**
```
Community Detection:
  Communities detected: 4
  Modularity: 0.4439

Edge Measures (Top 3 by Betweenness):
  Edge (0, 11): 0.0303

Node Measures (Top 5 by PageRank):
  Node 33: 0.0970
```

---

### clustering_benchmark_example.py
**Benchmark All Clustering Combinations**

```bash
python examples/clustering_benchmark_example.py
```

**What it does:**
- Tests **18 combinations** of:
  - DR: PCA, MDS, t-SNE
  - Clustering: Hierarchical, DBSCAN, K-Means
  - Quality: Calinski-Harabasz, Davies-Bouldin
- Runs 3 iterations per combination
- Reports average timing and quality scores
- Exports results to CSV and JSON

**Use when:**
- Comparing algorithm performance
- Choosing best algorithm for your data
- Need metrics for publication/reporting

---

### network_benchmark_example.py
**Benchmark All Network Analysis Combinations**

```bash
python examples/network_benchmark_example.py
```

**What it does:**
- Tests **27 combinations** of:
  - Community: Louvain, Girvan-Newman, Label Propagation
  - Edge Measures: Betweenness, Weight, Jaccard
  - Node Measures: PageRank, Degree, Closeness Centrality
- Runs 5 iterations per combination
- Reports timing and modularity scores
- Exports complete results with communities and centralities

**Use when:**
- Analyzing social/biological networks
- Comparing community detection algorithms
- Performance testing network measures

---

### yaml_benchmark_example.py
**YAML Configuration Demonstration**

```bash
python examples/yaml_benchmark_example.py
```

**What it does:**
- Example 1: Runs clustering benchmark from `clustering_benchmark.yaml`
- Example 2: Runs network benchmark from `network_benchmark.yaml`
- Example 3: Shows YAML structure and explains how configs work

**Learn:**
- How to structure YAML benchmark configs
- How to load and run configs programmatically
- How combinations are generated from YAML

---

### run_benchmark_example.py
**Command-Line YAML Benchmark Runner**

```bash
# Run clustering benchmark
python examples/run_benchmark_example.py -c examples/clustering_benchmark.yaml

# Run network benchmark
python examples/run_benchmark_example.py -c examples/network_benchmark.yaml

# With verbose output
python examples/run_benchmark_example.py -c examples/network_benchmark.yaml --verbose
```

**Features:**
- Auto-detects benchmark type (clustering/network)
- Loads any YAML configuration file
- Detailed progress reporting
- Error handling with verbose mode
- Exports to CSV/JSON with timestamps

**Use when:**
- Running benchmarks from command line
- Batch/scripted execution
- Need detailed logging and error traces

---

### basic_component_usage.py
**Direct Component Usage Without Pipelines**

```bash
python examples/basic_component_usage.py
```

**What it does:**
- **Example 1:** Clustering workflow step-by-step
  - Load dataset → Apply PCA → Cluster → Evaluate quality
- **Example 2:** Network analysis step-by-step
  - Load network → Detect communities → Edge measures → Node measures
- No pipelines, no benchmarks, just direct component calls

**Use when:**
- Learning framework basics
- Need fine-grained control over each step
- Debugging or experimenting with components
- Understanding component interfaces

---

## YAML Configuration Files

### clustering_benchmark.yaml
Basic clustering benchmark with 2 algorithms and 2 quality measures.

```yaml
benchmark:
  name: "Iris_Clustering_Benchmark"
  dataset: "../data/iris.csv"

pipeline_template:
  - type: "clustering"
    algorithms: ["Hierarchical", "DBSCAN"]
    # ... parameters ...

  - type: "clustering_quality"
    algorithms: ["Calinski_Harabasz", "Davies_Bouldin"]

iterations: 3
```

### network_benchmark.yaml
Comprehensive network analysis with 27 combinations.

```yaml
benchmark:
  name: "Network_Analysis_Benchmark"
  dataset: "../data/karate.edgelist"

pipeline_template:
  - type: "community_detection"
    algorithms: ["Louvain", "GirvanNewman", "LabelPropagation"]

  - type: "node_measures"
    algorithms: ["PageRank", "DegreeCentrality", "ClosenessCentrality"]

  - type: "edge_measures"
    algorithms: ["EdgeBetweenness", "EdgeWeight", "JaccardCoefficient"]

iterations: 5
```

### dr_cl_quality.yaml
Complete DR → Clustering → Quality pipeline benchmark.

---

## Quick Decision Tree

**"Which example should I run?"**

```
Want to LEARN the framework?
  └─→ basic_component_usage.py

Want to run ONE specific analysis?
  ├─ Clustering? → clustering_pipeline_example.py
  └─ Networks?   → network_pipeline_example.py

Want to COMPARE multiple algorithms?
  ├─ Clustering? → clustering_benchmark_example.py
  └─ Networks?   → network_benchmark_example.py

Want to use YAML configs?
  ├─ Learn how? → yaml_benchmark_example.py
  └─ Run CLI?   → run_benchmark_example.py -c <config.yaml>
```

---

## Available Sample Datasets

Located in `../data/` directory:

| Dataset | Type | Description |
|---------|------|-------------|
| **iris.csv** | Tabular | 150 samples, 4 features, 3 classes |
| **karate.edgelist** | Network | Zachary's karate club (34 nodes, 78 edges) |
| **les_miserables.edgelist** | Network | Character co-appearances |
| **three_communities.edgelist** | Network | Synthetic network with clear structure |

---

## Installation & Setup

```bash
# Install the framework
pip install -e .

# Navigate to examples
cd examples

# Run any example
python clustering_pipeline_example.py
```

---

## Understanding Output

### Pipeline Output
```
Pipeline Results:
  Calinski-Harabasz Score: 629.66 (higher is better)

Execution Times:
  Stage1: 0.0012s
  Stage2: 0.1205s
  Total: 0.1221s
```

### Benchmark Output
```
Top 3 Results:
1. PCA + Hierarchical + Calinski_Harabasz
   Avg Time: 0.1150s | Quality: 629.66
   Std Dev: 0.0012s

Results exported to: benchmark_results/
  - CSV format: timing_data.csv
  - JSON format: results.json
```

---

## Tips

1. **Start simple**: Begin with `basic_component_usage.py`
2. **Use pipelines**: Move to pipeline examples for production
3. **Benchmark wisely**: Use benchmarks when comparing algorithms
4. **YAML for reproducibility**: Use configs for experiments
5. **Check timing**: All examples show execution times
6. **Modify configs**: Copy and adapt YAML files for your data

---

## Other Files

- **appendix_a_examples.py** - Simplified examples for documentation
- **run_pipeline_example.py** - multi-example file

---

## Need Help?

- Main documentation: `../README.md`
- Architecture guide: `../ARCHITECTURE.md`
- Example docstrings: Read comments in each Python file
- GitHub issues: Report problems or ask questions
