# 💳 Bank Fraud Detection using Machine Learning

## 📌 Project Overview
This project builds a Machine Learning model to detect fraudulent credit card transactions.  
The dataset is highly imbalanced, so advanced techniques like **SMOTE (Synthetic Minority Oversampling Technique)** were used to improve model performance and correctly identify fraud cases.

---

## 🚀 Features
- Data preprocessing and feature scaling
- Handling imbalanced dataset using SMOTE
- Model training using:
  - Logistic Regression
  - Random Forest (Final Selected Model)
- Model evaluation using:
  - Confusion Matrix
  - Precision & Recall
  - ROC-AUC Score
- Interactive Web Application using Streamlit

---

## 📊 Dataset Information
- Total Transactions: **284,807**
- Total Features: **31**
- Fraud Cases: **492**
- Highly imbalanced dataset (0.17% fraud cases)

---

## 🧠 Final Model Performance (Random Forest)
- **ROC-AUC Score:** 0.956
- **Fraud Precision:** 0.87
- **Fraud Recall:** 0.80
- **F1-Score:** 0.83

The model successfully detects fraudulent transactions with strong recall and balanced precision.

---

## 🛠 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Joblib

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```bash
python train.py
```

### 3️⃣ Run the Web Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```
bank-fraud-detection/
│
├── data/
│   └── creditcard.csv
├── train.py
├── app.py
├── fraud_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
```

---

## 🎯 Future Improvements
- Hyperparameter tuning
- Implementation of XGBoost / Gradient Boosting
- Deployment using FastAPI
- Cloud hosting
- Real-time fraud detection API