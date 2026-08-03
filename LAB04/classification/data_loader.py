import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_classification_data(data_path="C:\\ML-CPE\\LAB04\\data-heart\\synthetic_heart_disease_dataset.csv"):
    """
    Load and preprocess dataset for Heart Disease Classification.
    """
    # 1. Load Dataset
    df = pd.read_csv(data_path)
    
    # 2. Select Target & Feature Columns
    target_col = 'Heart_Disease' if 'Heart_Disease' in df.columns else df.columns[-1]
    
    # Select numerical features for KNN
    feature_cols = ['Cholesterol_Total', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Age']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(feature_cols) < 2:
        # Fallback to all numeric columns except target
        feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()

    # 3. Handle Missing Values
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    X = X.fillna(X.median())
    
    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Feature Scaling (Crucial for KNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_cols