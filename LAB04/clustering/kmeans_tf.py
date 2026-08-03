import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_elbow_method(X_scaled, max_k=10, output_path="outputs/01_elbow.png"):
    """
    Run Elbow Method to find optimal number of clusters and save plot.
    """
    wcss = [] # Within-Cluster Sum of Squares
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Save Elbow Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='r')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Elbow plot saved to '{output_path}'")

def fit_kmeans(X_scaled, n_clusters=3):
    """
    Fit K-Means model with chosen K.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels