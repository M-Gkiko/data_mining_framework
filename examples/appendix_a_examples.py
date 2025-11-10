"""
Appendix A: Code Examples
"""

from data_mining_framework import (
    CSVDataset, PCAProjection, HierarchicalClustering, DaviesBouldinIndex,
    Pipeline, DRAdapter, ClusteringAdapter, ClusteringQualityAdapter,
    NetworkXWrapper, LouvainCommunityDetection, PageRankMeasure,
    NetworkAdapter, CommunityDetectionAdapter, NodeMeasureAdapter
)

# Example 1: Clustering Pipeline
print("\nExample 1: Clustering Pipeline (Iris dataset, 150 samples, 4 features)\n")

dataset = CSVDataset('data/iris.csv')
pipeline = Pipeline("PCA_Hierarchical_Quality")
pipeline.add_component(DRAdapter(PCAProjection(n_components=2)))
pipeline.add_component(ClusteringAdapter(
    HierarchicalClustering(n_clusters=3, linkage='ward', metric='euclidean'), None))
pipeline.add_component(ClusteringQualityAdapter(DaviesBouldinIndex()))

results = pipeline.execute(dataset)

score = list(results.values())[0] if isinstance(results, dict) else results
print(f"Results: Davies-Bouldin Score: {score:.2f} (lower is better)\n")


# Example 2: Network Analysis Pipeline
print("Example 2: Network Analysis Pipeline (Karate Club network, 34 nodes, 78 edges)\n")

network = NetworkXWrapper(filepath='data/karate.edgelist', format='edgelist')
pipeline = Pipeline("Louvain_PageRank")
pipeline.add_component(NetworkAdapter(network))
pipeline.add_component(CommunityDetectionAdapter(LouvainCommunityDetection(resolution=1.0)))
pipeline.add_component(NodeMeasureAdapter(PageRankMeasure(alpha=0.85)))

results = pipeline.execute(None)

num_communities = len(set(results['communities'].values()))
modularity = results['modularity']
pr_scores = results['node_scores']
top_node = max(pr_scores.items(), key=lambda x: x[1])

print(f"Results: {num_communities} communities detected | Modularity: {modularity:.2f}")
print(f"Results: Top node by PageRank: Node {top_node[0]} ({top_node[1]:.4f})\n")

