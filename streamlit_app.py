import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import pickle
from datetime import datetime

# --------------------- DATABASE CONNECTION ---------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin@123",
        database="bank_fraud_db"
    )

# --------------------- LOAD MODEL FILES ---------------------
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# --------------------- STREAMLIT UI ---------------------
st.set_page_config(page_title="Bank Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Bank Transaction Fraud Detection System")
st.markdown("### Detect suspicious transactions using Machine Learning")

# --------------------- INPUT FORM ---------------------
st.sidebar.header("🧾 Enter Transaction Details")

type_option = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.sidebar.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
oldbalanceOrg = st.sidebar.number_input("Sender Old Balance", min_value=0.0, value=5000.0, step=100.0)
newbalanceOrig = st.sidebar.number_input("Sender New Balance", min_value=0.0, value=4000.0, step=100.0)
oldbalanceDest = st.sidebar.number_input("Receiver Old Balance", min_value=0.0, value=2000.0, step=100.0)
newbalanceDest = st.sidebar.number_input("Receiver New Balance", min_value=0.0, value=3000.0, step=100.0)

if st.sidebar.button("🔍 Predict Fraud"):
    # Encode transaction type
    type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
    type_code = type_map[type_option]

    # Prepare input data
    input_data = pd.DataFrame([[type_code, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]],
                              columns=["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])

    # Ensure column order matches model training
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_columns]

    # Scale input
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    # Show prediction result
    if prediction == 1:
        st.error("🚨 Alert! This transaction is **Fraudulent**.")
    else:
        st.success("✅ This transaction appears **Legitimate**.")

    # --------------------- SAVE RESULT TO MYSQL ---------------------
           # --------------------- SAVE RESULT TO MYSQL ---------------------
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO `transaction` (
            cust_id, transaction_amount, transaction_type, trans_date, is_fraud
        )
        VALUES (%s, %s, %s, %s, %s)
        """
        # You can set a dummy customer ID (e.g., 1) or link it from login later
        values = (1, amount, type_option, datetime.now(), int(prediction))
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        st.info("🗄️ Transaction saved to database successfully!")
    except Exception as e:
        st.warning(f"⚠️ Database insert failed: {e}")
