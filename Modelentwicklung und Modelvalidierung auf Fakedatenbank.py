#!/usr/bin/env python
# coding: utf-8

import os
from faker import Faker
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import sqlite3
import joblib

# Function to list all tables in the SQLite database


def list_tables(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return [table[0] for table in tables]
    except Exception as e:
        return f"Fehler: {e}"


# Path to the database file
db_path = "Datenbank/source_allergy.db"

# List tables in the database
tables = list_tables(db_path)
print("Tabellen in der Datenbank:", tables)

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

# Retrieve patient and observation data
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
medications_count = pd.read_sql_query(
    "SELECT PATIENT, COUNT(*) as MEDICATION_COUNT FROM medications GROUP BY PATIENT;", conn)

# Close the connection
conn.close()

# Calculate age
patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'])
patients_df['AGE'] = (pd.to_datetime("today") -
                      patients_df['BIRTHDATE']).dt.days // 365

# Merge data
merged_df = patients_df.merge(
    bmi_values, left_on='Id', right_on='PATIENT', how='left')
merged_df = merged_df.merge(
    medications_count, left_on='Id', right_on='PATIENT', how='left')
merged_df.fillna(
    {'BMI': merged_df['BMI'].median(), 'MEDICATION_COUNT': 0}, inplace=True)

# Define features and target variable
features = ['BMI', 'HEALTHCARE_COVERAGE', 'AGE', 'MEDICATION_COUNT']
X = merged_df[features]
y = merged_df['HEALTHCARE_EXPENSES']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the best XGBoost model
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.2,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 1.0
}

optimized_xgb = XGBRegressor(**best_params, random_state=42)
optimized_xgb.fit(X_train, y_train)

# Path to the model file
model_path = 'optimized_xgb_model.pkl'

# Remove the existing model file if it exists
if os.path.exists(model_path):
    os.remove(model_path)

# Save the model
joblib.dump(optimized_xgb, model_path)
print(f"Model saved to {model_path}")

# Make predictions with the optimized model
y_pred_optimized = optimized_xgb.predict(X_test)

# Evaluate the model
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# Print results
print("\n📊 **Optimiertes XGBoost Modell**")
print(f"MAE: {mae_optimized:.2f}")
print(f"MSE: {mse_optimized:.2f}")
print(f"RMSE: {rmse_optimized:.2f}")
print(f"R² Score: {r2_optimized:.4f}")

# Create a realistic fake database
fake = Faker()
np.random.seed(42)

# Path to the fake database file
ultra_realistic_fake_db_path = "ultra_realistic_fake_healthcare.db"

# Remove the existing fake database file if it exists
if os.path.exists(ultra_realistic_fake_db_path):
    os.remove(ultra_realistic_fake_db_path)

# Connect to the fake database
conn = sqlite3.connect(ultra_realistic_fake_db_path)
cursor = conn.cursor()

# Create tables with extended features
cursor.execute("""CREATE TABLE patients (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    HEALTHCARE_EXPENSES REAL, 
                    HEALTHCARE_COVERAGE REAL, 
                    BIRTHDATE TEXT, 
                    INCOME REAL,
                    CHRONIC_CONDITIONS TEXT,
                    HOSPITALIZATIONS INTEGER,
                    SMOKER INTEGER,
                    GENDER TEXT,
                    EMPLOYMENT_STATUS TEXT,
                    EMERGENCY_VISITS INTEGER)""")

cursor.execute("""CREATE TABLE observations (
                    PATIENT INTEGER, 
                    VALUE REAL, 
                    CODE TEXT, 
                    FOREIGN KEY (PATIENT) REFERENCES patients(Id))""")

cursor.execute("""CREATE TABLE medications (
                    PATIENT INTEGER, 
                    TOTALCOST REAL, 
                    FOREIGN KEY (PATIENT) REFERENCES patients(Id))""")

cursor.execute("""CREATE TABLE encounters (
                    PATIENT INTEGER, 
                    TOTAL_CLAIM_COST REAL, 
                    FOREIGN KEY (PATIENT) REFERENCES patients(Id))""")

# Increase the number of patients
num_patients = 10000

# Lists for fake data
patients_data = []
observations_data = []
medications_data = []
encounters_data = []

# Possible values for health status & employment
chronic_conditions_list = ["None", "Diabetes", "COPD",
                           "Cancer", "Hypertension", "Depression", "Multiple Conditions"]
employment_status_list = ["Unemployed", "Employed", "Self-Employed", "Retired"]
gender_list = ["Male", "Female", "Other"]

# Generate realistically distributed health data
for i in range(1, num_patients + 1):
    # Simulate age
    birthdate = fake.date_of_birth(
        minimum_age=20, maximum_age=90).strftime("%Y-%m-%d")
    age = 2025 - int(birthdate[:4])

    # Health coverage (younger people often have worse coverage)
    healthcare_coverage = np.clip(
        np.random.normal(0.7 - (age / 200), 0.2), 0.1, 1.0)

    # Simulate income (higher age → slightly decreasing income)
    income = np.clip(np.random.normal(
        50000 - (age * 300), 15000), 15000, 150000)

    # BMI (higher values for older people, with natural dispersion)
    bmi = np.clip(np.random.normal(22 + (age / 50), 3), 18, 40)

    # Randomly select gender
    gender = np.random.choice(gender_list)

    # Randomly select employment status (older people more often retired)
    employment_status = np.random.choice(employment_status_list, p=[
                                         0.1, 0.6, 0.2, 0.1] if age < 65 else [0.05, 0.2, 0.15, 0.6])

    # Chronic conditions with age-dependent probabilities
    chronic_condition = np.random.choice(chronic_conditions_list, p=[
                                         0.5 - (age / 200), 0.2, 0.1, 0.05, 0.1, 0.05, 1.0 - (0.5 - (age / 200) + 0.2 + 0.1 + 0.05 + 0.1 + 0.05)])

    # Hospitalizations increase with age & pre-existing conditions
    hospitalizations = np.clip(np.random.poisson(
        0.5 + (0.3 if chronic_condition != "None" else 0) + (age / 80)), 0, 10)

    # Smoking status (older people smoke more often in historical data)
    smoker = np.random.choice([0, 1], p=[0.75, 0.25]
                              if age > 50 else [0.85, 0.15])

    # Emergency visits increase with age & smoking
    emergency_visits = np.clip(np.random.poisson(
        0.2 + (smoker * 0.2) + (hospitalizations * 0.1)), 0, 5)

    # Medication costs depend on age, BMI & pre-existing conditions
    medication_cost = np.clip(np.random.normal(
        500 + (age * 10) + (bmi * 5) + (hospitalizations * 200), 500), 0, 15000)

    # Treatment costs increase with age, pre-existing conditions & smoking status
    encounter_cost = np.clip(np.random.normal(
        2000 + (age * 20) + (bmi * 10) + (hospitalizations * 500) + (smoker * 1000), 1000), 1000, 70000)

    # Health expenses depend on insurance, income, pre-existing conditions & smoking status
    base_expenses = encounter_cost + medication_cost + \
        (hospitalizations * 5000) + (emergency_visits * 2000)
    healthcare_expenses = np.clip(base_expenses * (1 - healthcare_coverage) * (
        1 + smoker * 0.2) * (1 - income / 200000), 5000, 200000)

    # Save data
    patients_data.append((healthcare_expenses, healthcare_coverage, birthdate, income,
                         chronic_condition, hospitalizations, smoker, gender, employment_status, emergency_visits))
    observations_data.append((i, bmi, '39156-5'))
    medications_data.append((i, medication_cost))
    encounters_data.append((i, encounter_cost))

# Insert data into tables
cursor.executemany("INSERT INTO patients (HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE, INCOME, CHRONIC_CONDITIONS, HOSPITALIZATIONS, SMOKER, GENDER, EMPLOYMENT_STATUS, EMERGENCY_VISITS) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", patients_data)
cursor.executemany(
    "INSERT INTO observations (PATIENT, VALUE, CODE) VALUES (?, ?, ?)", observations_data)
cursor.executemany(
    "INSERT INTO medications (PATIENT, TOTALCOST) VALUES (?, ?)", medications_data)
cursor.executemany(
    "INSERT INTO encounters (PATIENT, TOTAL_CLAIM_COST) VALUES (?, ?)", encounters_data)

# Save changes and close connection
conn.commit()
conn.close()

print("✅ Realistische Fake-Datenbank wurde erfolgreich erstellt: ultra_realistic_fake_healthcare.db")

# Connect to the fake database
conn = sqlite3.connect("ultra_realistic_fake_healthcare.db")

# Retrieve data
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE, INCOME, CHRONIC_CONDITIONS, HOSPITALIZATIONS, SMOKER, GENDER, EMPLOYMENT_STATUS, EMERGENCY_VISITS FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
medications_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTALCOST) as TOTAL_MED_COST FROM medications GROUP BY PATIENT;", conn)
encounter_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTAL_CLAIM_COST) as TOTAL_ENC_COST FROM encounters GROUP BY PATIENT;", conn)

# Close the connection
conn.close()

# Calculate age
patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
patients_df["AGE"] = (pd.to_datetime("today") -
                      patients_df["BIRTHDATE"]).dt.days // 365

# Merge data
merged_df = patients_df.merge(
    bmi_values, left_on="Id", right_on="PATIENT", how="left")
merged_df = merged_df.merge(medications_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(encounter_cost, on="PATIENT", how="left")

# Fill missing values with median for numerical features
fill_values = {
    "BMI": merged_df["BMI"].median(),
    "TOTAL_MED_COST": 0,
    "TOTAL_ENC_COST": 0,
}
merged_df.fillna(fill_values, inplace=True)

# Ensure all numeric columns are of numeric type
numeric_columns = ["HOSPITALIZATIONS", "SMOKER", "EMERGENCY_VISITS"]
for col in numeric_columns:
    merged_df[col] = pd.to_numeric(
        merged_df[col], errors="coerce").fillna(0).astype(int)

# Convert categorical variables to numeric codes
merged_df["GENDER"] = merged_df["GENDER"].astype("category").cat.codes
merged_df["EMPLOYMENT_STATUS"] = merged_df["EMPLOYMENT_STATUS"].astype(
    "category").cat.codes
merged_df["CHRONIC_CONDITIONS"] = merged_df["CHRONIC_CONDITIONS"].astype(
    "category").cat.codes

# Define features and target variable
features = ["BMI", "AGE", "HEALTHCARE_COVERAGE", "TOTAL_MED_COST", "TOTAL_ENC_COST", "INCOME",
            "HOSPITALIZATIONS", "SMOKER", "GENDER", "EMPLOYMENT_STATUS", "EMERGENCY_VISITS", "CHRONIC_CONDITIONS"]
X = merged_df[features]
y = merged_df["HEALTHCARE_EXPENSES"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the best XGBoost model
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.2,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 1.0
}

optimized_xgb = XGBRegressor(**best_params, random_state=42)
optimized_xgb.fit(X_train, y_train)

# Path to the model file
model_path = 'optimized_xgb_model.pkl'

# Remove the existing model file if it exists
if os.path.exists(model_path):
    os.remove(model_path)

# Save the model
joblib.dump(optimized_xgb, model_path)
print(f"Model saved to {model_path}")

# Make predictions with the optimized model
y_pred_optimized = optimized_xgb.predict(X_test)

# Evaluate the model
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# Print results
print("\n📊 **Optimiertes XGBoost Modell auf ultra-realistischen Fake-Daten**")
print(f"MAE: {mae_optimized:.2f}")
print(f"MSE: {mse_optimized:.2f}")
print(f"RMSE: {rmse_optimized:.2f}")
print(f"R² Score: {r2_optimized:.4f}")
