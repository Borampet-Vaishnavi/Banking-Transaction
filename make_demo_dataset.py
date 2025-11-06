import pandas as pd
import numpy as np
import random

random.seed(42)

rows = []

# normal transactions
for _ in range(200):
    t = random.choice(["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"])
    amt = random.uniform(500, 5000)
    sender_old = random.uniform(amt, amt + 10000)
    sender_new  = sender_old - amt if t in ["CASH_OUT","TRANSFER","DEBIT","PAYMENT"] else sender_old
    recv_old = random.uniform(1000, 8000)
    recv_new = recv_old + amt if t in ["CASH_IN","TRANSFER","PAYMENT"] else recv_old
    rows.append([t, amt, sender_old, sender_new, recv_old, recv_new, 0])

# fraudulent transactions
for _ in range(60):
    t = random.choice(["CASH_OUT","TRANSFER","PAYMENT"])
    amt = random.uniform(8000, 40000)
    sender_old = random.uniform(1000, 10000)
    # keep sender balance unchanged or inconsistent
    sender_new  = sender_old - random.uniform(0, amt/4)
    recv_old = random.uniform(1000, 5000)
    recv_new = recv_old + random.uniform(0, amt/10)
    rows.append([t, amt, sender_old, sender_new, recv_old, recv_new, 1])

df = pd.DataFrame(rows, columns=[
    "type","amount","oldbalanceOrg","newbalanceOrig",
    "oldbalanceDest","newbalanceDest","isFraud"
])

df.to_csv("AIML Dataset.csv", index=False)
print("✅ Synthetic dataset saved as 'AIML Dataset.csv'")
print(df['isFraud'].value_counts())
