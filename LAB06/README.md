# Lab 06: Neural Network for ECG Image Classification

An end-to-end Deep Learning pipeline designed to classify Electrocardiogram (ECG) heartbeat images across multiple cardiac conditions using Convolutional Neural Networks (CNN) implemented in TensorFlow/Keras.

---

## 📌 Project Overview & Objectives

* **Data Ingestion**: Dynamically load, read, and index image files from multi-class subdirectories.
* **Standardization & Preprocessing**: Resize images and handle channel normalization (`[0, 1]` feature scaling) natively within the model graph to optimize memory efficiency.
* **Stratified Splitting**: Partition data into Training, Validation, and Test sets while preserving class distribution ratios across all subsets.
* **Model Architecture**: Implement a deep CNN architecture with Batch Normalization, Dropout regularization, and Adaptive Learning Rate scheduling.
* **Model Evaluation**: Track and evaluate multi-class performance using Accuracy, Classification Reports, Confusion Matrices, and Loss/Accuracy convergence curves.

---

## 🗂️ Project Structure

```text
LAB06/
├── data_loader.py       # Traverses directories and loads raw images into NumPy arrays
├── preprocessing.py     # Resizes and standardizes image formats (BGR to RGB)
├── split_data.py        # Performs 3-way stratified train/val/test data splitting
├── nn_model.py          # Builds, compiles, trains, and saves the CNN model
├── evaluate.py          # Evaluates performance, exports confusion matrices and loss curves
├── main.py              # Central execution pipeline orchestrating the full workflow
└── outputs/             # Generated artifacts (saved models, metrics, JSON history, and plots) 
