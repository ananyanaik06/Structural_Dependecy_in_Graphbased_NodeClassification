import pandas as pd
import networkx as nx
import numpy as np

CORA_CITES = "data/cora/cora.cites"
CORA_CONTENT = "data/cora/cora.content"

NODES_FILE = "data/nodes.csv"
LABELS_FILE = "data/node_labels.csv"
EDGES_FILE = "data/graph_edges.csv"
FEATURES_FILE = "data/graph_features.csv"

# STEP 1: PROCESS cora.content
# cora.content format:
# paper_id | word_1 ... word_1433 | label

content = pd.read_csv(
    CORA_CONTENT,
    sep="\t",
    header=None
)

#   nodes.csv  
nodes_df = content[[0]]
nodes_df.columns = ["node_id"]
nodes_df.to_csv(NODES_FILE, index=False)

#  node_labels.csv 
labels_df = content[[0, content.shape[1] - 1]]
labels_df.columns = ["node_id", "label"]

# Encode string labels to integers
labels_df["label"] = labels_df["label"].astype("category").cat.codes
labels_df.to_csv(LABELS_FILE, index=False)

print("Created nodes.csv and node_labels.csv")

# PROCESS cora.cites
# cora.cites format:
# cited_paper  citing_paper

edges = pd.read_csv(
    CORA_CITES,
    sep="\t",
    header=None,
    names=["dst", "src"]
)

edges = edges[["src", "dst"]]
edges.to_csv(EDGES_FILE, index=False)

print("Created graph_edges.csv")

# STEP 3: FEATURE EXTRACTION

# Load edge list
edges = pd.read_csv(EDGES_FILE)

# Directed graph (degree features)
G_directed = nx.from_pandas_edgelist(
    edges,
    source="src",
    target="dst",
    create_using=nx.DiGraph()
)

# Undirected graph (cohesion + centrality)
G_undirected = G_directed.to_undirected()

nodes = list(G_directed.nodes())

#  Degree features 
in_degree = dict(G_directed.in_degree())
out_degree = dict(G_directed.out_degree())

#  Local cohesion 
clustering = nx.clustering(G_undirected)
triangle_count = nx.triangles(G_undirected)

#  Centrality 
betweenness = nx.betweenness_centrality(G_undirected, normalized=True)
closeness = nx.closeness_centrality(G_undirected)

try:
    eigenvector = nx.eigenvector_centrality(G_undirected, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eigenvector = {n: 0.0 for n in nodes}

#---neighbour_features---

# ==============================
# NEIGHBOUR-BASED FEATURES
# ==============================

# Helper dictionaries for reuse
degree_undirected = dict(G_undirected.degree())

mean_nbr_degree = {}
max_nbr_degree = {}
min_nbr_degree ={}
mean_nbr_clustering = {}
mean_nbr_betweenness = {}
mean_nbr_closeness = {}

for node in nodes:
    neighbors = list(G_undirected.neighbors(node))

    # Handle isolated nodes safely
    if len(neighbors) == 0:
        mean_nbr_degree[node] = 0.0
        max_nbr_degree[node] = 0.0
        mean_nbr_clustering[node] = 0.0
        mean_nbr_betweenness[node] = 0.0
        mean_nbr_closeness[node] = 0.0
        continue

    #  Degree-based neighbour features 
    nbr_degrees = [degree_undirected[nbr] for nbr in neighbors]
    mean_nbr_degree[node] = np.mean(nbr_degrees)
    max_nbr_degree[node] = np.max(nbr_degrees)
    min_nbr_degree[node] = np.min(nbr_degrees)

    #  Clustering-based neighbour features 
    nbr_clustering = [clustering[nbr] for nbr in neighbors]
    mean_nbr_clustering[node] = np.mean(nbr_clustering)

    # Centrality-based neighbour features 
    nbr_betweenness = [betweenness[nbr] for nbr in neighbors]
    mean_nbr_betweenness[node] = np.mean(nbr_betweenness)

    nbr_closeness = [closeness[nbr] for nbr in neighbors]
    mean_nbr_closeness[node] = np.mean(nbr_closeness)

#  Build feature table]
rows = []
for node in nodes:
    rows.append({
        "node_id": node,
        "in_degree": in_degree.get(node, 0),
        "out_degree": out_degree.get(node, 0),
        "clustering_coefficient": clustering.get(node, 0.0),
        "triangle_count": triangle_count.get(node, 0),
        "betweenness_centrality": betweenness.get(node, 0.0),
        "closeness_centrality": closeness.get(node, 0.0),
        "eigenvector_centrality": eigenvector.get(node, 0.0),

        # Neighbour-level features
        "mean_neighbor_degree": mean_nbr_degree[node],
        "max_neighbor_degree": max_nbr_degree[node],
        "mean_neighbor_clustering": mean_nbr_clustering[node],
        "mean_neighbor_betweenness": mean_nbr_betweenness[node],
        "mean_neighbor_closeness": mean_nbr_closeness[node],
    })

features_df = pd.DataFrame(rows)
features_df.to_csv(FEATURES_FILE, index=False)

print("Graph features saved to graph_features.csv")
