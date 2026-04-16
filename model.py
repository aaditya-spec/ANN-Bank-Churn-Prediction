import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Features & target
X = data.iloc[:, 3:13]
y = data.iloc[:, 13]

# Encoding (simple)
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN Model
model = Sequential()
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)
import joblib
import json

model.save("artifacts/churn_model.keras")
joblib.dump(sc, "artifacts/scaler.pkl")

with open("artifacts/columns.json", "w") as f:
    json.dump(list(X.columns), f)