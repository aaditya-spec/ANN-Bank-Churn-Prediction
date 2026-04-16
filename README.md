# 🧠 ANN Bank Churn Prediction

## 📌 Project Overview

Customer churn is a major challenge for banks. This project uses an Artificial Neural Network (ANN) to predict whether a customer is likely to leave the bank.

By identifying high-risk customers in advance, banks can take proactive steps to improve retention and reduce revenue loss.

---

## 🎯 Objectives

* Predict customer churn using machine learning
* Build a deep learning model using ANN
* Improve prediction accuracy with feature preprocessing
* Understand customer behavior patterns

---

## 📊 Dataset Description

The dataset contains bank customer information with multiple features:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

👉 Target Variable:

* **Exited (0 = No, 1 = Yes)**

---

## ⚙️ Data Preprocessing

* Removed unnecessary columns (RowNumber, CustomerId, Surname)
* Applied **One-Hot Encoding** for categorical variables
* Feature Scaling using **StandardScaler**
* Train-Test Split (80% training, 20% testing)

---

## 🧠 Model Architecture

Artificial Neural Network (ANN):

* Input Layer
* Hidden Layer 1 → 6 neurons (ReLU)
* Hidden Layer 2 → 6 neurons (ReLU)
* Output Layer → 1 neuron (Sigmoid)

👉 Loss Function: Binary Crossentropy
👉 Optimizer: Adam

---

## 🚀 Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn

---

## 🛠️ Installation & Setup

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
py model.py
```

---

## 📈 Model Performance

* Accuracy: **~84%**
* Good balance between bias and variance
* Suitable for real-world churn prediction scenarios

---

## 📂 Project Structure

```
ANN-Bank-Churn-Prediction/
│── model.py
│── requirements.txt
│── Artificial_Neural_Network_Case_Study_data.csv
│── README.md
```

---

## 🔍 Key Insights

* Customers with low activity are more likely to churn
* Geography and balance significantly impact churn
* ANN captures non-linear relationships effectively

---

## 🚧 Future Improvements

* Hyperparameter tuning
* Add Dropout layers to reduce overfitting
* Try advanced models (XGBoost, Random Forest)
* Deploy using Flask / Streamlit

---

## 👨‍💻 Author

**Aaditya Pundir**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
