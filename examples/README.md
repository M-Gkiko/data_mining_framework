# Examples

This directory contains example scripts and configuration files showing how to use the Data Mining Framework.

## Available Examples

### Python Scripts
- **`basic_clustering.py`** - Simple clustering example with quality evaluation
- **`pipeline_example.py`** - Complete pipeline: DR → Clustering → Quality  
- **`benchmark_example.py`** - Comprehensive benchmarking examples

### YAML Configuration Files
- **`clustering_benchmark.yaml`** - Basic clustering benchmark configuration
- **`dr_cl_quality.yaml`** - Complete DR→Clustering→Quality pipeline benchmark

## Running Examples

After installing the framework:

```bash
# Install the framework first
pip install -e .

# Run Python examples
cd examples
python basic_clustering.py
python pipeline_example.py
python benchmark_example.py

# Run benchmark configs directly
dm-benchmark clustering_benchmark.yaml
dm-benchmark dr_cl_quality.yaml --verbose
```

## Using the YAML Configurations

The YAML files demonstrate how to structure benchmark configurations:

1. **Copy and modify** the YAML files for your own datasets
2. **Adjust parameters** like `n_clusters`, `eps`, `n_components` for your data
3. **Change the dataset path** to point to your CSV file
4. **Add or remove algorithms** based on your needs

Example modification:
```yaml
benchmark:
  name: "My_Custom_Analysis"
  dataset: "my_data.csv"  # Your dataset here

pipeline_template:
  - type: "clustering"
    algorithms: ["Hierarchical", "DBSCAN"]
    params:
      Hierarchical:
        n_clusters: 5  # Adjust for your data
        distance_measure: "Manhattan"
```

## Note on Data

The examples reference placeholder datasets. For real usage:

- Use CSV files with numeric data
- Ensure proper formatting (rows = samples, columns = features)
- Handle missing values before analysis
- Choose appropriate numbers of clusters for your domain

## Creating Your Own Examples

Feel free to create additional examples and configurations! The framework is designed to be flexible and extensible.