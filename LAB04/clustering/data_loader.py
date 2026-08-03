import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_clustering_data(data_path="C:\\ML-CPE\\LAB04\\data-heart\\synthetic_heart_disease_dataset.csv"):
    """
    Load and preprocess numerical features for Heart Disease Clustering.
    """
    df = pd.read_csv(data_path)
    
    # Select continuous numerical features for clustering
    feature_cols = ['Cholesterol_Total', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Age']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(feature_cols) < 2:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Handle Missing Values
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    # Feature Scaling (Crucial for Distance-based K-Means!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df, feature_cols