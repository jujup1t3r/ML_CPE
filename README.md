# 🤖 ML-CPE: Machine Learning Pipeline Project

This repository contains Machine Learning coursework and lab implementations, covering data preprocessing, regression analysis, classification, clustering, support vector machines, and deep neural networks with medical image classification.

---

## 📌 Project Overview

This project builds a full-stack Machine Learning pipeline designed to handle real-world medical data challenges. It spans across:

1. **Data Preprocessing & Cleaning:** Handling missing values, bad clinical data, and feature encoding.
2. **Regression Analysis:** Continuous prediction of patient age using clinical attributes.
3. **Classification Modeling:** Categorical prediction of heart disease risks (Logistic Regression & KNN).
4. **Clustering Analysis:** Unsupervised patient health risk grouping using K-Means.
5. **Support Vector Machines (SVM):** Supervised decision boundary classification on medical imaging and clinical features.
6. **Deep Neural Networks (CNN/MLP):** Image classification of Electrocardiogram (ECG) heart disease data using TensorFlow/Keras.
7. **Model Performance Evaluation:** Comparative analysis across models, loss curves, confusion matrices, and classification metrics.

---

## 📂 Repository Structure & Navigation

Click on any lab link below to view its specific documentation, code, and findings:

```text
ML-CPE/
├── 📁 [LAB02/](./LAB02/README.md) — Data Preprocessing & Data Pipeline
│   ├── 📁 [dataset/](./LAB02/dataset/)
│   │   └── 📄 synthetic_heart_disease_dataset.csv
│   ├── 📓 [ML_LAB2.ipynb](./LAB02/ML_LAB2.ipynb)
│   └── 📄 [README.md](./LAB02/README.md)
│
├── 📁 [LAB03/](./LAB03/README.md) — Regression, Classification & Model Comparison
│   ├── 📓 [ML_LAB3.ipynb](./LAB03/ML_LAB3.ipynb)
│   └── 📄 [README.md](./LAB03/README.md)
│
├── 📁 [LAB04/](./LAB04/README.md) — KNN Classification & K-Means Clustering
│   ├── 📁 [data-heart/](./LAB04/data-heart/)
│   │   └── 📄 heart_disease_dataset.csv
│   ├── 📁 [classification/](./LAB04/classification/)
│   ├── 📁 [clustering/](./LAB04/clustering/)
│   ├── 📄 requirements.txt
│   ├── 📄 link-data.txt
│   └── 📄 [README.md](./LAB04/README.md)
│
├── 📁 [LAB05/](./LAB05/README.md) — Support Vector Machine (SVM) Image Recognition
│   ├── 📁 [ECG_DATA/](./LAB05/ECG_DATA/)
│   ├── 📄 data_load.py
│   ├── 📄 preprocess.py
│   ├── 📄 split_data.py
│   ├── 📄 svm_model.py
│   ├── 📄 evaluate.py
│   ├── 📄 main.py
│   └── 📄 [README.md](./LAB05/README.md)
│
├── 📁 [LAB06/](./LAB06/README.md) — Deep Neural Networks & ECG Classification
│   ├── 📄 data_loader.py
│   ├── 📄 preprocessing.py
│   ├── 📄 split_data.py
│   ├── 📄 nn_model.py
│   ├── 📄 evaluate.py
│   ├── 📄 main.py
│   ├── 📁 outputs/
│   └── 📄 [README.md](./LAB06/README.md)
│
├── 📄 .gitignore
└── 📄 README.md
---

## 🚀 Quick Links to Labs

| Lab | Topic | Primary Files | Documentation |
| :--- | :--- | :---: | :---: |
| **LAB 02** | Data Preprocessing & Pipeline | [ML_LAB2.ipynb](./LAB02/ML_LAB2.ipynb) | [View README](./LAB02/README.md) |
| **LAB 03** | Regression, Classification & Comparison | [ML_LAB3.ipynb](./LAB03/ML_LAB3.ipynb) | [View README](./LAB03/README.md) |
| **LAB 04** | KNN Classification & K-Means Clustering | [classification/](./LAB04/classification/), [clustering/](./LAB04/clustering/) | [View README](./LAB04/README.md) |
| **LAB 05** | Support Vector Machine (SVM) | [svm_model.py](./LAB05/svm_model.py), [main.py](./LAB05/main.py) | [View README](./LAB05/README.md) |
| **LAB 06** | Neural Networks & ECG Image Classification | [nn_model.py](./LAB06/nn_model.py), [main.py](./LAB06/main.py) | [View README](./LAB06/README.md) |

---

## 🛠️ Requirements & Setup

To run the labs and notebooks in this repository, clone the repo and install the required dependencies:

```bash
# 1. Clone repository
git clone [https://github.com/jujup1t3r/ML-CPE.git](https://github.com/jujup1t3r/ML-CPE.git)

# 2. Navigate into the project directory
cd ML-CPE

# 3. Install all required dependencies
pip install tensorflow opencv-python matplotlib seaborn scikit-learn numpy pandas joblib jupyter
