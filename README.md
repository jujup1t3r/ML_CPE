# 🤖 ML-CPE: Machine Learning Pipeline Project

This repository contains Machine Learning coursework and lab implementations, covering data preprocessing, regression analysis, classification, clustering, and model performance evaluation.

---

## 📌 Project Overview

This project builds a full-stack Machine Learning pipeline designed to handle real-world medical data challenges. It spans across:

1. **Data Preprocessing & Cleaning:** Handling missing values, bad clinical data, and feature encoding.
2. **Regression Analysis:** Continuous prediction of patient age using clinical attributes.
3. **Classification Modeling:** Categorical prediction of heart disease risks (Logistic Regression & KNN).
4. **Clustering Analysis:** Unsupervised patient health risk grouping using K-Means.
5. **Model Performance Evaluation:** Comparative analysis across models, $K$-Curves, and performance metrics.

---

## 📂 Repository Structure & Navigation

Click on any lab link below to view its specific documentation, code, and findings:

ML-CPE/
├── 📁 [LAB02/](./LAB02/README.md) — *Data Preprocessing & Data Pipeline*  
│   ├── 📁 [dataset/](./LAB02/dataset/)  
│   │   └── 📄 `synthetic_heart_disease_dataset.csv`  
│   ├── 📓 [ML_LAB2.ipynb](./LAB02/ML_LAB2.ipynb)  
│   └── 📄 [README.md](./LAB02/README.md) *(Lab 2 Details)*  
│  
├── 📁 [LAB03/](./LAB03/README.md) — *Regression, Classification & Model Comparison*  
│   ├── 📓 [ML_LAB3.ipynb](./LAB03/ML_LAB3.ipynb)  
│   └── 📄 [README.md](./LAB03/README.md) *(Lab 3 Details)*  
│  
├── 📁 [LAB04/](./LAB04/README.md) — *KNN Classification & K-Means Clustering*  
│   ├── 📁 [data-heart/](./LAB04/data-heart/)  
│   │   └── 📄 `heart_disease_dataset.csv`  
│   ├── 📁 [classification/](./LAB04/classification/)  
│   ├── 📁 [clustering/](./LAB04/clustering/)  
│   ├── 📄 [requirements.txt](./LAB04/requirements.txt)  
│   ├── 📄 [link-data.txt](./LAB04/link-data.txt)  
│   └── 📄 [README.md](./LAB04/README.md) *(Lab 4 Details)*  
│  
├── 📄 `.gitignore`  
└── 📄 `README.md` *(Main Overview)*  

---

## 🚀 Quick Links to Labs

| Lab | Topic | Primary Files | Documentation |
| :--- | :--- | :---: | :---: |
| **LAB 02** | Data Preprocessing & Pipeline | [ML_LAB2.ipynb](./LAB02/ML_LAB2.ipynb) | [View README](./LAB02/README.md) |
| **LAB 03** | Regression, Classification & Comparison | [ML_LAB3.ipynb](./LAB03/ML_LAB3.ipynb) | [View README](./LAB03/README.md) |
| **LAB 04** | KNN Classification & K-Means Clustering | [classification/](./LAB04/classification/), [clustering/](./LAB04/clustering/) | [View README](./LAB04/README.md) |

---

## 🛠️ Requirements & Setup

To run the labs in this repository, clone the repo and install the required dependencies:

```bash
# Clone repository
git clone [https://github.com/your-username/ML-CPE.git](https://github.com/your-username/ML-CPE.git)

# Navigate into the project
cd ML-CPE

# Install required dependencies
pip install -r LAB04/requirements.txt
