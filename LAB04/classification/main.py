import os
from data_loader import load_classification_data
from knn_tf import KNNClassifierModel, find_best_k
from evaluate import evaluate_and_save

def main():
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    print("1. Loading Classification Data...")
    X_train, X_test, y_train, y_test, feature_cols = load_classification_data()

    print("\n2. Finding Best K Value...")
    best_k = find_best_k(X_train, X_test, y_train, y_test)

    print(f"\n3. Training KNN Model with K={best_k}...")
    knn_model = KNNClassifierModel(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)

    print("\n4. Predicting on Test Data...")
    y_pred = knn_model.predict(X_test)

    print("\n5. Evaluating Model Performance...")
    evaluate_and_save(y_test, y_pred)

if __name__ == "__main__":
    main()