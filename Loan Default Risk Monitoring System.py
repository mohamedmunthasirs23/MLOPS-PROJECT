import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import mysql.connector
from datetime import datetime

# Load data
df = pd.read_csv(r"D:\MLOPS\2nd part\Project\loan.csv", low_memory=False)

# Select a subset of important features
features = ['loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
            'home_ownership', 'annual_inc', 'purpose', 'dti',
            'delinq_2yrs', 'revol_util', 'total_acc', 'loan_status', 'addr_state']

df = df[features].dropna()

# Filter for binary classification (Fully Paid vs Charged Off)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# Encode target
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# Encode categorical features
categorical_cols = ['term', 'grade', 'emp_length', 'home_ownership', 'purpose']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save addr_state before splitting, for region mapping
addr_state_test = df['addr_state']

# Drop addr_state for modeling
df = df.drop(['addr_state'], axis=1)

# Split data
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Clean column names
X_train.columns = X_train.columns.str.replace('[\[\]<>]', '', regex=True)
X_test.columns = X_test.columns.str.replace('[\[\]<>]', '', regex=True)

# Train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
pred_proba = model.predict_proba(X_test)[:, 1]
predictions = (pred_proba > 0.5).astype(int)

# Build the result DataFrame
results = pd.DataFrame(X_test.copy())
results['actual'] = y_test.values
results['predicted_proba'] = pred_proba
results['prediction'] = predictions
results['timestamp'] = pd.Timestamp.now()

# Add addr_state and region mapping
results['addr_state'] = addr_state_test.loc[results.index].values

state_to_region = {
    'CA': 'West', 'NV': 'West', 'AZ': 'West', 'OR': 'West', 'WA': 'West',
    'NY': 'Northeast', 'NJ': 'Northeast', 'MA': 'Northeast',
    'TX': 'South', 'FL': 'South', 'GA': 'South',
    'IL': 'Midwest', 'OH': 'Midwest', 'MI': 'Midwest'
    # Add other states as needed
}

results['region'] = results['addr_state'].map(state_to_region)

# Convert timestamp to MySQL format
results['timestamp'] = results['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

# MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="loan_risk_db"
)

cursor = conn.cursor()

# Drop and recreate table
cursor.execute("DROP TABLE IF EXISTS loan_predictions")
cursor.execute("""
CREATE TABLE IF NOT EXISTS loan_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    actual INT,
    predicted_proba FLOAT,
    prediction INT,
    timestamp DATETIME,
    addr_state VARCHAR(10),
    region VARCHAR(50)
)
""")

# Insert into MySQL
for _, row in results.iterrows():
    cursor.execute("""
        INSERT INTO loan_predictions (actual, predicted_proba, prediction, timestamp, addr_state, region)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        int(row['actual']),
        float(row['predicted_proba']),
        int(row['prediction']),
        row['timestamp'],
        row['addr_state'],
        row['region']
    ))

# Commit and close
conn.commit()
cursor.close()
conn.close()

print("Data uploaded to MySQL with region info successfully!")
