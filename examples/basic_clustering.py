"""
Basic clustering example using the data mining framework.
"""

from data_mining_framework import (
    CSVDataset,
    HierarchicalClustering,
    ManhattanDistance,
    CalinskiHarabaszIndex
)


def main():
    dataset = CSVDataset('../data/iris.csv')
    print(f"Dataset: {dataset.get_rows()} rows, {len(dataset.get_features())} features")

    distance_measure = ManhattanDistance()
    clustering = HierarchicalClustering(
        distance_measure=distance_measure,
        n_clusters=3,
        linkage='complete'
    )

    clustering.fit(dataset)
    labels = clustering.get_labels()
    print(f"Clusters found: {len(set(labels))}")

    quality_measure = CalinskiHarabaszIndex()
    score = quality_measure.evaluate(dataset, labels)
    print(f"Calinski-Harabasz Score: {score:.3f}")


if __name__ == "__main__":
    main()
