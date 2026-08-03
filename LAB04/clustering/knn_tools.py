import pandas as pd

def save_cluster_summaries(df, cluster_labels, feature_cols, 
                           output_summary="outputs/cluster_summary.csv", 
                           output_patients="outputs/clustered_patients.csv"):
    """
    Summarize cluster profiles and save CSV outputs.
    """
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels

    # 1. Save Full Clustered Dataset
    df_clustered.to_csv(output_patients, index=False)

    # 2. Compute Mean Profiles per Cluster
    summary = df_clustered.groupby('Cluster')[feature_cols].mean().reset_index()
    summary['Patient_Count'] = df_clustered.groupby('Cluster').size().values
    summary.to_csv(output_summary, index=False)

    print(f"Clustered patients saved to '{output_patients}'")
    print(f"Cluster summary saved to '{output_summary}'")
    
    print("\n--- Cluster Summary Profiles ---")
    print(summary.to_string(index=False))