# neural-net-from-scratch
# 🧠 Neural Network from Scratch – Titanic Survival Prediction

This project builds a complete **production-level neural network from scratch using only NumPy** to predict survival outcomes on the Titanic dataset from Kaggle. It follows a full machine learning pipeline, including data preprocessing, feature engineering, model architecture design, training with backpropagation, and evaluation — all without using deep learning libraries like TensorFlow or PyTorch.

---

## 🎯 Problem Statement

Using passenger data (such as age, gender, ticket class), predict whether a passenger survived the Titanic shipwreck. This is a **binary classification** task (0 = did not survive, 1 = survived) based on real-world data.

---

## 🧱 Project Structure

neural-net-from-scratch/
├── data/ ← Titanic dataset (train.csv, test.csv)
├── src/ ← Modular source code
│ ├── main.py ← Pipeline logic (train, eval, predict)
│ ├── preprocess.py ← Encoding, scaling, missing values
│ ├── model.py ← Neural network architecture
│ ├── layers.py ← Dense & activation layers
│ ├── optimizers.py ← SGD, Adam optimizer
│ ├── evaluate.py ← Accuracy, F1, Confusion Matrix
│ ├── utils.py ← Mutual info, plotting, helpers
│ └── config.py ← Hyperparameters
├── notebooks/ ← Optional EDA
├── outputs/ ← Saved weights, metrics
├── requirements.txt
└── README.md


---

## 🚀 Features

- Clean and modular code using only **NumPy + pandas**
- Full **data preprocessing** pipeline:
  - Handling missing values
  - Label encoding and one-hot encoding
  - Feature scaling (standardization)
  - Mutual information feature selection
- **Neural network from scratch**:
  - Forward pass & backpropagation
  - Support for multiple hidden layers
  - Activation functions: ReLU, Sigmoid
  - Loss functions: Binary Cross-Entropy, MSE
- **Custom optimizer** implementations:
  - SGD
  - Adam (with momentum + bias correction)
- Model evaluation:
  - Accuracy, Precision, Recall, F1
  - Confusion matrix visualization
  - Learning curves
- Regularization:
  - L2 weight decay
  - Dropout (optional)
  - Early stopping (optional)
- Generalization checks: overfitting/underfitting
- Final model used for **inference and Kaggle submission**

---

## 📊 Dataset

- Source: [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- Format: CSV
- Target column: `Survived`
- Key features: `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Embarked`, etc.

