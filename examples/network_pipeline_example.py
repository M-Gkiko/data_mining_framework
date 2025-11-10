"""
Network analysis example demonstrating community detection and centrality measures.
"""

import networkx as nx
from data_mining_framework import (
    NetworkXWrapper,
    LouvainCommunityDetection,
    PageRankMeasure,
    DegreeCentralityMeasure,
    GirvanNewmanCommunity,
    LabelPropagationCommunity
)


def simple_community_detection():
    print("Simple Community Detection Example\n")

    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)

    print(f"Network: {network.node_count()} nodes, {network.edge_count()} edges")

    louvain = LouvainCommunityDetection(resolution=1.0)
    louvain.fit(network)
    communities = louvain.get_communities()
    modularity = louvain.get_modularity()

    print(f"Communities detected: {len(set(communities.values()))}")
    print(f"Modularity: {modularity:.4f}\n")


def node_centrality_example():
    print("Node Centrality Measures Example\n")

    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)

    pagerank = PageRankMeasure(alpha=0.85)
    pr_scores = pagerank.calculate(network)

    top_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 nodes by PageRank:")
    for node, score in top_nodes:
        print(f"  Node {node}: {score:.4f}")
    print()


def compare_algorithms():
    print("Compare Community Detection Algorithms\n")

    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)

    algorithms = {
        "Louvain": LouvainCommunityDetection(resolution=1.0),
        "Girvan-Newman": GirvanNewmanCommunity(k=2),
        "Label Propagation": LabelPropagationCommunity(max_iterations=100)
    }

    print(f"Network: {network.node_count()} nodes, {network.edge_count()} edges\n")

    for name, algo in algorithms.items():
        algo.fit(network)
        communities = algo.get_communities()
        modularity = algo.get_modularity()

        if isinstance(communities, dict):
            num_communities = len(set(communities.values()))
        else:
            num_communities = len(communities)

        print(f"{name}:")
        print(f"  Communities: {num_communities}")
        print(f"  Modularity: {modularity:.4f}\n")


if __name__ == "__main__":
    simple_community_detection()
    node_centrality_example()
    compare_algorithms()
