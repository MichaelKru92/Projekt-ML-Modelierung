import unittest
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from faker import Faker


class TestXGBoostModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a fake database
        cls.fake = Faker()
        np.random.seed(42)
        cls.conn = sqlite3.connect(":memory:")
        cls.create_fake_database(cls.conn)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    @staticmethod
    def create_fake_database(conn):
        cursor = conn.cursor()
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

        num_patients = 1000
        patients_data = []
        observations_data = []
        medications_data = []
        encounters_data = []

        chronic_conditions_list = ["None", "Diabetes", "COPD",
                                   "Cancer", "Hypertension", "Depression", "Multiple Conditions"]
        employment_status_list = ["Unemployed",
                                  "Employed", "Self-Employed", "Retired"]
        gender_list = ["Male", "Female", "Other"]

        for i in range(1, num_patients + 1):
            birthdate = Faker().date_of_birth(
                minimum_age=20, maximum_age=90).strftime("%Y-%m-%d")
            age = 2025 - int(birthdate[:4])
            healthcare_coverage = np.clip(
                np.random.normal(0.7 - (age / 200), 0.2), 0.1, 1.0)
            income = np.clip(np.random.normal(
                50000 - (age * 300), 15000), 15000, 150000)
            bmi = np.clip(np.random.normal(22 + (age / 50), 3), 18, 40)
            gender = np.random.choice(gender_list)
            employment_status = np.random.choice(employment_status_list, p=[
                                                 0.1, 0.6, 0.2, 0.1] if age < 65 else [0.05, 0.2, 0.15, 0.6])
            chronic_condition = np.random.choice(chronic_conditions_list, p=[
                                                 0.5 - (age / 200), 0.2, 0.1, 0.05, 0.1, 0.05, 1.0 - (0.5 - (age / 200) + 0.2 + 0.1 + 0.05 + 0.1 + 0.05)])
            hospitalizations = np.clip(np.random.poisson(
                0.5 + (0.3 if chronic_condition != "None" else 0) + (age / 80)), 0, 10)
            smoker = np.random.choice(
                [0, 1], p=[0.75, 0.25] if age > 50 else [0.85, 0.15])
            emergency_visits = np.clip(np.random.poisson(
                0.2 + (smoker * 0.2) + (hospitalizations * 0.1)), 0, 5)
            medication_cost = np.clip(np.random.normal(
                500 + (age * 10) + (bmi * 5) + (hospitalizations * 200), 500), 0, 15000)
            encounter_cost = np.clip(np.random.normal(
                2000 + (age * 20) + (bmi * 10) + (hospitalizations * 500) + (smoker * 1000), 1000), 1000, 70000)
            base_expenses = encounter_cost + medication_cost + \
                (hospitalizations * 5000) + (emergency_visits * 2000)
            healthcare_expenses = np.clip(base_expenses * (1 - healthcare_coverage) * (
                1 + smoker * 0.2) * (1 - income / 200000), 5000, 200000)

            patients_data.append((healthcare_expenses, healthcare_coverage, birthdate, income,
                                 chronic_condition, hospitalizations, smoker, gender, employment_status, emergency_visits))
            observations_data.append((i, bmi, '39156-5'))
            medications_data.append((i, medication_cost))
            encounters_data.append((i, encounter_cost))

        cursor.executemany("INSERT INTO patients (HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE, INCOME, CHRONIC_CONDITIONS, HOSPITALIZATIONS, SMOKER, GENDER, EMPLOYMENT_STATUS, EMERGENCY_VISITS) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", patients_data)
        cursor.executemany(
            "INSERT INTO observations (PATIENT, VALUE, CODE) VALUES (?, ?, ?)", observations_data)
        cursor.executemany(
            "INSERT INTO medications (PATIENT, TOTALCOST) VALUES (?, ?)", medications_data)
        cursor.executemany(
            "INSERT INTO encounters (PATIENT, TOTAL_CLAIM_COST) VALUES (?, ?)", encounters_data)

        conn.commit()

    def test_xgboost_model(self):
        conn = self.conn

        # Load data from the fake database
        patients_df = pd.read_sql_query(
            "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE, INCOME, CHRONIC_CONDITIONS, HOSPITALIZATIONS, SMOKER, GENDER, EMPLOYMENT_STATUS, EMERGENCY_VISITS FROM patients;", conn)
        bmi_values = pd.read_sql_query(
            "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
        medications_cost = pd.read_sql_query(
            "SELECT PATIENT, SUM(TOTALCOST) as TOTAL_MED_COST FROM medications GROUP BY PATIENT;", conn)
        encounter_cost = pd.read_sql_query(
            "SELECT PATIENT, SUM(TOTAL_CLAIM_COST) as TOTAL_ENC_COST FROM encounters GROUP BY PATIENT;", conn)

        # Calculate age from birthdate
        patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
        patients_df["AGE"] = (pd.to_datetime("today") -
                              patients_df["BIRTHDATE"]).dt.days // 365

        # Merge dataframes
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

        # Load the saved model
        model_path = 'optimized_xgb_model.pkl'
        optimized_xgb = joblib.load(model_path)

        # Make predictions with the loaded model
        y_pred_optimized = optimized_xgb.predict(X)

        # Evaluate the model
        mae_optimized = mean_absolute_error(y, y_pred_optimized)
        mse_optimized = mean_squared_error(y, y_pred_optimized)
        rmse_optimized = np.sqrt(mse_optimized)
        r2_optimized = r2_score(y, y_pred_optimized)

        # Assert that the model performance is within acceptable limits
        self.assertLess(mae_optimized, 10000, "MAE is too high")
        self.assertLess(rmse_optimized, 15000, "RMSE is too high")
        self.assertGreater(r2_optimized, 0.5, "R² Score is too low")

        # Print success messages if assertions pass
        print(f"Mean Basolute Error: {mae_optimized:.2f}")
        print(f"Mean sqaured Error: {mse_optimized:.2f}")
        print(f"Roor Mean Squared Error: {rmse_optimized:.2f}")
        print(f"R² Score: {r2_optimized:.4f}")


if __name__ == '__main__':
    unittest.main()
