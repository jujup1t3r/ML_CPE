# 🫀 LAB 04: KNN Classification & K-Means Clustering

This lab implements an end-to-end Machine Learning pipeline using K-Nearest Neighbors (KNN) for disease classification and K-Means for patient health clustering on a Heart Disease Dataset.

---

## 📌 Lab Overview

1. **Classification (KNN):** Predicts binary target `Heart_Disease` (0 or 1), evaluates optimal $K$ values using $K$-Accuracy Curves, and normalizes confusion matrices.
2. **Clustering (K-Means):** Groups patients based on physiological metrics, finds optimal cluster counts via the Elbow Method, and projects high-dimensional metrics using 2D PCA.

---

## 📁 LAB04 Directory Structure

```text
LAB04/
├── data-heart/
│   └── heart_disease_dataset.csv
│
├── classification/
│   ├── main.py
│   ├── data_loader.py
│   ├── knn_tf.py
│   ├── evaluate.py
│   └── outputs/
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       └── predictions.csv
│
└── clustering/
    ├── main.py
    ├── data_loader.py
    ├── kmeans_tf.py
    ├── knn_tools.py
    ├── visualize.py
    └── outputs/
        ├── 01_elbow.png
        ├── 02_clusters.png
        ├── cluster_summary.csv
        └── clustered_patients.csv
