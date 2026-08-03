import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(X_scaled, cluster_labels, output_path="outputs/02_clusters.png"):
    """
    Visualize high-dimensional clusters using PCA 2D reduction.
    """
    # Reduce dimensions to 2D for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.title('Patient Health Clusters (PCA 2D Projection)')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Cluster visualization saved to '{output_path}'")