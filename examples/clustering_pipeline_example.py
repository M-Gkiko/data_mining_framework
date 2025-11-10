"""
Clustering Pipeline Example: Dimensionality Reduction + Clustering + Quality Evaluation

This example demonstrates a complete clustering pipeline using the framework:
1. Load a dataset (Iris)
2. Apply dimensionality reduction (PCA)
3. Perform clustering (Hierarchical)
4. Evaluate quality (Calinski-Harabasz Index)

The pipeline chains these operations together and tracks execution times.
"""

from data_mining_framework import (
    CSVDataset,
    PCAProjection,
    HierarchicalClustering,
    CalinskiHarabaszIndex,
    ManhattanDistance,
    Pipeline,
    DRAdapter,
    ClusteringAdapter,
    ClusteringQualityAdapter
)


def main():
    print("Clustering Pipeline Example: PCA -> Hierarchical Clustering -> Quality\n")

    # Load dataset
    dataset = CSVDataset('data/iris.csv')
    print(f"Dataset: {dataset.get_rows()} samples, {len(dataset.get_features())} features\n")

    # Create pipeline with three stages
    distance_measure = ManhattanDistance()
    pipeline = Pipeline("PCA_Hierarchical_Quality")

    # Stage 1: Dimensionality Reduction
    pipeline.add_component(DRAdapter(PCAProjection(n_components=2)))

    # Stage 2: Clustering
    pipeline.add_component(ClusteringAdapter(
        HierarchicalClustering(
            distance_measure=distance_measure,
            n_clusters=3,
            linkage='complete'
        ),
        distance_measure
    ))

    # Stage 3: Quality Evaluation
    pipeline.add_component(ClusteringQualityAdapter(CalinskiHarabaszIndex()))

    # Execute pipeline
    results = pipeline.execute(dataset)

    # Display results
    print("Pipeline Results:")
    print(f"  Calinski-Harabasz Score: {list(results.values())[0]:.4f} (higher is better)\n")

    print("Execution Times:")
    for component, time in pipeline.get_execution_times().items():
        print(f"  {component}: {time:.4f}s")
    print(f"  Total: {pipeline.get_total_time():.4f}s")


if __name__ == "__main__":
    main()
