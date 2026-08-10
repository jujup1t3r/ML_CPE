# ⚡ LAB 05: Support Vector Machine (SVM) Classification

This lab implements an end-to-end Support Vector Machine (SVM) pipeline for **ECG (Electrocardiogram)** classification, covering data loading, preprocessing, model training, evaluation, and artifact saving.

---

## 📌 Lab Overview

1. **Preprocessing & Feature Extraction:** Processes raw ECG data, normalizes feature distributions using `StandardScaler`, and prepares train/test splits (`.npy` binary files).
2. **SVM Classification:** Trains a Support Vector Machine model (`svm_model.py`) to classify ECG signals/images across target classes.
3. **Evaluation:** Evaluates model performance using accuracy metrics, confusion matrix visualizations, and saved evaluation reports (`classes.json`, `confusion_matrix.png`).

---

## 📁 LAB05 Directory Structure

```text
LAB05/
├── archive (1)/
│   └── ECG_DATA/
│       ├── test/
│       └── train/
├── outputs/   <-- (Generated after running main.py)
│   ├── classes.json
│   ├── confusion_matrix.png
│   ├── images.npy
│   ├── labels.npy
│   ├── scaler.pkl
│   ├── svm_model.pkl
│   ├── X_test.npy
│   ├── X_train.npy
│   ├── y_test.npy
│   └── y_train.npy
├── data_load.py
├── preprocess.py
├── split_data.py
├── svm_model.py
├── evaluate.py
├── test_svm.py
├── main.py
└── README.md
