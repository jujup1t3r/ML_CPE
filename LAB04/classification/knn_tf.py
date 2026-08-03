import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class KNNClassifierModel:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

def find_best_k(X_train, X_test, y_train, y_test, max_k=20, output_path="outputs/01_k_curve.png"):
    """
    Find best K value and save K-Curve plot.
    """
    k_range = range(1, max_k + 1, 2)  # Use odd K values
    accuracies = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)

    # Save K-Curve Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, accuracies, marker='o', linestyle='--', color='b')
    plt.title('KNN Accuracy vs. K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    best_k = k_range[np.argmax(accuracies)]
    print(f"Best K found: {best_k} with Accuracy: {max(accuracies)*100:.2f}%")
    return best_k