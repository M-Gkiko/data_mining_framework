"""
Pipeline example: Dimensionality Reduction + Clustering + Quality Evaluation.
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
    dataset = CSVDataset('../data/iris.csv')
    print(f"Dataset: {dataset.get_rows()} rows, {len(dataset.get_features())} features\n")

    distance_measure = ManhattanDistance()
    pipeline = Pipeline("PCA_Hierarchical_Quality")

    pca = PCAProjection(n_components=2)
    pipeline.add_component(DRAdapter(pca))

    clustering = HierarchicalClustering(
        distance_measure=distance_measure,
        n_clusters=3,
        linkage='complete'
    )
    pipeline.add_component(ClusteringAdapter(clustering, distance_measure))

    quality_measure = CalinskiHarabaszIndex()
    pipeline.add_component(ClusteringQualityAdapter(quality_measure))

    results = pipeline.execute(dataset)

    print("Pipeline Results:")
    print(f"  Execution times: {pipeline.get_execution_times()}")
    print(f"  Quality scores: {results}")


if __name__ == "__main__":
    main()
