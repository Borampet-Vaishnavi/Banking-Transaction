import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

print("🔹 Loading dataset...")
data = pd.read_csv("AIML Dataset.csv")

# ✅ Use only 50,000 random samples for faster training
data = data.sample(n=50000, random_state=42)

print("🔹 Preprocessing data...")

# Drop irrelevant columns (customize based on your dataset)
if 'nameOrig' in data.columns:
    data = data.drop(['nameOrig', 'nameDest'], axis=1)

# Replace 'type' categorical feature with numerical encoding
if 'type' in data.columns:
    data['type'] = data['type'].astype('category').cat.codes

# Define features (X) and target (y)
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🔹 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# Create folder for models
os.makedirs("models", exist_ok=True)

# Save model, scaler, and columns
with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("🎉 Model training completed! Files saved in 'models/' folder.")
