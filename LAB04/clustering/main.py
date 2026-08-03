import os
from data_loader import load_clustering_data
from kmeans_tf import run_elbow_method, fit_kmeans
from knn_tools import save_cluster_summaries
from visualize import visualize_clusters

def main():
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    print("1. Loading Clustering Data...")
    X_scaled, df_raw, feature_cols = load_clustering_data()

    print("\n2. Generating Elbow Curve...")
    run_elbow_method(X_scaled)

    # Choose K = 3 as default optimal clusters for health risk grouping
    optimal_k = 3
    print(f"\n3. Running K-Means with K={optimal_k}...")
    kmeans, cluster_labels = fit_kmeans(X_scaled, n_clusters=optimal_k)

    print("\n4. Visualizing Clusters...")
    visualize_clusters(X_scaled, cluster_labels)

    print("\n5. Generating & Saving Cluster Summaries...")
    save_cluster_summaries(df_raw, cluster_labels, feature_cols)

if __name__ == "__main__":
    main()