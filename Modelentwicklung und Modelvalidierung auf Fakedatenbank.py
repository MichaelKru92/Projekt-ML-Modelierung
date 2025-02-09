#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
from sklearn.model_selection import train_test_split
import pandas as pd
import sqlite3


def list_tables(db_path):
    """ Listet alle Tabellen in einer SQLite-Datenbank auf. """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return [table[0] for table in tables]
    except Exception as e:
        return f"Fehler: {e}"


# Pfad zur Datenbankdatei
db_path = "/Users/tician/Downloads/source_allergy.db"

tables = list_tables(db_path)
print("Tabellen in der Datenbank:", tables)


# Nachdem wir wissen, welche Tabellen vorliegen versuchen wir nun die Healthcare Expenses anhand des Body-Mass-Index vorherzusagen. Wir entscheiden uns zunaechst fuer ein lineares Regressionsmodell, vielleicht gibt es ja einen linearen Zusammenhang.

# Kurzer Reminder, wir messen die Performance von Regressionsmodellen wie folgt:
# Mean Absolute Error (MAE)
# → Der durchschnittliche absolute Fehler zwischen den vorhergesagten und den tatsächlichen Werten.
# Gibt an, wie weit die Vorhersagen im Durchschnitt vom tatsächlichen Wert abweichen.
#
# Mean Squared Error (MSE)
# → Das durchschnittliche Quadrat der Fehler zwischen den Vorhersagen und den tatsächlichen Werten.
# Bestraft große Fehler stärker als kleine, da die Fehler quadriert werden.
#
# Root Mean Squared Error (RMSE)
# → Die Quadratwurzel des MSE, also der Fehler in der gleichen Einheit wie die Zielvariable.
# Besonders nützlich, wenn man verstehen will, wie stark die Vorhersagen im Schnitt abweichen.
#
# R-squared Score (R²)
# → Gibt an, wie viel Prozent der Varianz der Zielvariable durch das Modell erklärt wird (zwischen 0 und 1).

# In[8]:


# Verbindung zur SQLite-Datenbank herstellen
db_path = "/Users/tician/Downloads/source_allergy.db"
conn = sqlite3.connect(db_path)

# Patienten- und Beobachtungsdaten abrufen
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)

# Verbindung schließen
conn.close()

# Daten zusammenführen
bmi_expenses_df = pd.merge(patients_df, bmi_values,
                           left_on='Id', right_on='PATIENT', how='inner')

# Features und Zielvariable definieren
X = bmi_expenses_df[['BMI']]
y = bmi_expenses_df['HEALTHCARE_EXPENSES']

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Lineares Regressionsmodell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Modellbewertung
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Ergebnisse ausgeben
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")


# Die Ergebnisse deuten darauf hin, dass das lineare Regressionsmodell nicht besonders gut darin ist, die zukünftigen Gesundheitsausgaben basierend auf dem BMI vorherzusagen. Die R²-Score von 0.108 zeigt, dass nur ~10.8% der Varianz in den Gesundheitsausgaben durch den BMI erklärt werden kann. Zudem sind die Fehlerwerte (MAE, MSE, RMSE) sehr hoch.
#
# Naechster Schritt:
#
# Wir sollten mehr Features einbeziehen: Aktuell nutzen wir nur den BMI. Weitere Faktoren wie Alter, Geschlecht, bestehende Erkrankungen, Medikationen und frühere Gesundheitskosten könnten das Modell verbessern.
#

# In[9]:


# Verbindung zur SQLite-Datenbank herstellen
db_path = "/Users/tician/Downloads/source_allergy.db"
conn = sqlite3.connect(db_path)

# Patienten- und Beobachtungsdaten abrufen
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
medications_count = pd.read_sql_query(
    "SELECT PATIENT, COUNT(*) as MEDICATION_COUNT FROM medications GROUP BY PATIENT;", conn)

# Verbindung schließen
conn.close()

# Alter berechnen
patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'])
patients_df['AGE'] = (pd.to_datetime("today") -
                      patients_df['BIRTHDATE']).dt.days // 365

# Daten zusammenführen
merged_df = patients_df.merge(
    bmi_values, left_on='Id', right_on='PATIENT', how='left')
merged_df = merged_df.merge(
    medications_count, left_on='Id', right_on='PATIENT', how='left')
merged_df.fillna(
    {'BMI': merged_df['BMI'].median(), 'MEDICATION_COUNT': 0}, inplace=True)

# Features und Zielvariable definieren
features = ['BMI', 'HEALTHCARE_COVERAGE', 'AGE', 'MEDICATION_COUNT']
X = merged_df[features]
y = merged_df['HEALTHCARE_EXPENSES']

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Lineares Regressionsmodell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Modellbewertung
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Ergebnisse ausgeben
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")


# Das Modell hat sich mit den zusätzlichen Features deutlich verbessert:
#
# MAE reduziert von ~514K auf ~335K
# MSE und RMSE haben sich verringert
# R²-Score stieg von 0.108 auf 0.486, was bedeutet, dass nun ~49% der Varianz der Gesundheitsausgaben erklärt wird.
#
# Bei der Implementation mehrerer Praediktoren lohnt es sich auch einmal nicht-lineare Modelle auszuprobieren: Gesundheitsausgaben könnten nicht-linear mit dem BMI zusammenhängen. Modelle wie Random Forest Regression oder Gradient Boosting könnten bessere Ergebnisse liefern.

# In[10]:


# Verbindung zur SQLite-Datenbank herstellen
db_path = "/Users/tician/Downloads/source_allergy.db"
conn = sqlite3.connect(db_path)

# Patienten- und Beobachtungsdaten abrufen
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
medications_count = pd.read_sql_query(
    "SELECT PATIENT, COUNT(*) as MEDICATION_COUNT FROM medications GROUP BY PATIENT;", conn)

# Verbindung schließen
conn.close()

# Alter berechnen
patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'])
patients_df['AGE'] = (pd.to_datetime("today") -
                      patients_df['BIRTHDATE']).dt.days // 365

# Daten zusammenführen
merged_df = patients_df.merge(
    bmi_values, left_on='Id', right_on='PATIENT', how='left')
merged_df = merged_df.merge(
    medications_count, left_on='Id', right_on='PATIENT', how='left')
merged_df.fillna(
    {'BMI': merged_df['BMI'].median(), 'MEDICATION_COUNT': 0}, inplace=True)

# Features und Zielvariable definieren
features = ['BMI', 'HEALTHCARE_COVERAGE', 'AGE', 'MEDICATION_COUNT']
X = merged_df[features]
y = merged_df['HEALTHCARE_EXPENSES']

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Modelle initialisieren
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Modelle trainieren und bewerten
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Results:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")
    print("-" * 50)


# Die lineare Regression zeigt eine relativ schlechte Anpassung an die Daten (R² = 0.49), was bedeutet, dass fast die Hälfte der Variabilität der Gesundheitsausgaben nicht durch die gewählten Features erklärt wird. Der hohe Fehler deutet darauf hin, dass die Beziehung zwischen den Features und der Zielvariablen möglicherweise nicht linear ist oder dass wichtige Features fehlen.
#
# Der Random Forest liefert extrem präzise Vorhersagen mit einem sehr hohen R²-Wert von 0.997. Dies deutet darauf hin, dass das Modell fast die gesamte Varianz in den Gesundheitsausgaben erfasst. Allerdings könnte das Modell overfitted sein, d.h., es passt sich zu stark an die Trainingsdaten an und verallgemeinert möglicherweise nicht gut auf neue Daten.
#
# Gradient Boosting liefert gute Ergebnisse mit einem hohen R²-Wert von 0.96, allerdings mit etwas schlechterer Leistung als Random Forest. Dies deutet darauf hin, dass das Modell weniger overfitted ist als der Random Forest, aber immer noch starke Zusammenhänge in den Daten erkennt.

# In[11]:


# Verbindung zur SQLite-Datenbank herstellen
db_path = "/Users/tician/Downloads/source_allergy.db"
conn = sqlite3.connect(db_path)

# Patienten- und Beobachtungsdaten abrufen
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
medications_count = pd.read_sql_query(
    "SELECT PATIENT, COUNT(*) as MEDICATION_COUNT FROM medications GROUP BY PATIENT;", conn)

# Verbindung schließen
conn.close()

# Alter berechnen
patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'])
patients_df['AGE'] = (pd.to_datetime("today") -
                      patients_df['BIRTHDATE']).dt.days // 365

# Daten zusammenführen
merged_df = patients_df.merge(
    bmi_values, left_on='Id', right_on='PATIENT', how='left')
merged_df = merged_df.merge(
    medications_count, left_on='Id', right_on='PATIENT', how='left')
merged_df.fillna(
    {'BMI': merged_df['BMI'].median(), 'MEDICATION_COUNT': 0}, inplace=True)

# Features und Zielvariable definieren
features = ['BMI', 'HEALTHCARE_COVERAGE', 'AGE', 'MEDICATION_COUNT']
X = merged_df[features]
y = merged_df['HEALTHCARE_EXPENSES']

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Random Forest Modell trainieren
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature Importance analysieren
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame(
    {'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Visualisierung der Feature Importance
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df['Feature'],
         feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance von Random Forest")
plt.gca().invert_yaxis()
plt.show()

# Hyperparameter-Tuning für Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(RandomForestRegressor(
    random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)

# Bestes Modell evaluieren
y_pred = rf_grid_search.best_estimator_.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Optimiertes Random Forest Modell:")
print(f"Beste Parameter: {rf_grid_search.best_params_}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")


# Erklärung des Ergebnisses im Medizinischen Kontext:
#
# Modelloptimierung und Auswahl der besten Parameter
# Das Modell wurde mithilfe eines Grid Search Verfahrens mit 5-facher Kreuzvalidierung optimiert. Das bedeutet:
#
# Das Modell testet 108 verschiedene Kombinationen von Parametern und führt insgesamt 540 Modelltrainings durch.
# Das beste Modell wurde mit den folgenden Hyperparametern ausgewählt:
#
# max_depth: None → Keine Begrenzung der Baumtiefe, das Modell kann komplexe Zusammenhänge erfassen.
# min_samples_leaf: 1 → Jeder Blattknoten kann bis zu einer einzelnen Beobachtung heruntergebrochen werden.
# min_samples_split: 2 → Ein Split wird durchgeführt, sobald mindestens 2 Beobachtungen in einem Knoten vorhanden sind.
# n_estimators: 200 → Das Modell verwendet 200 Entscheidungsbäume, um eine robuste Vorhersage zu treffen.
# Modellgüte und Fehlermetriken.
#
# Das Modell sagt zukünftige Gesundheitsausgaben basierend auf Faktoren wie BMI, Alter, Krankenversicherung und Anzahl der Medikamente vorher.
#
# Mean Absolute Error (MAE): 7.012 USD
#
# Das Modell liegt im Durchschnitt um etwa 7.012 USD daneben.
# Ein niedriger MAE bedeutet eine hohe Vorhersagegenauigkeit.
# Mean Squared Error (MSE): 1.213 Milliarden USD²
#
# Eine quadratische Fehlergröße, die extrem hohe Abweichungen stärker bestraft.
# Ein niedriger MSE bedeutet, dass große Fehler selten vorkommen.
# Root Mean Squared Error (RMSE): 34.841 USD
#
# Ein Maß für die durchschnittliche Fehlergröße.
# Das Modell hat im Durchschnitt eine Abweichung von ca. 34.841 USD bei den vorhergesagten Gesundheitskosten.
# R²-Score: 0.997
#
# Dies bedeutet, dass das Modell 99.7 % der Schwankungen der Gesundheitsausgaben erklären kann.
# Ein R²-Wert nahe 1 ist ideal und zeigt, dass das Modell fast perfekt arbeitet.
#
# Medizinische Interpretation
# Dieses Modell könnte in der Krankenhausverwaltung oder Versicherungsmathematik eingesetzt werden, um zukünftige Gesundheitsausgaben von Patienten vorherzusagen.
#
# Krankenversicherungen könnten es nutzen, um bessere Tarife für Patienten mit höheren erwarteten Ausgaben festzulegen.
# Krankenhäuser und Kliniken könnten damit Budget- und Ressourcenzuweisungen optimieren.
# Ärzte könnten es nutzen, um Risikopatienten mit besonders hohen Gesundheitsausgaben frühzeitig zu identifizieren und gezielt präventive Maßnahmen einzuleiten.
# Das Modell ist sehr präzise, sollte aber mit weiteren klinischen Parametern (z. B. Vorerkrankungen, Laborwerte) ergänzt werden, um eine noch genauere Vorhersage zu ermöglichen.
#
#
# Das Modell laesst sich wie folgt visualisieren:

# In[12]:


# Visualisierung der Modellvorhersagen
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Vorhersagen')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         linestyle='dashed', color='red', label='Perfekte Übereinstimmung')
plt.xlabel("Tatsächliche Gesundheitsausgaben")
plt.ylabel("Vorhergesagte Gesundheitsausgaben")
plt.title("Vergleich: Tatsächliche vs. vorhergesagte Gesundheitsausgaben")
plt.legend()
plt.show()


# Im Folgenden schauen wir uns noch weitere moegliche Praediktoren, Modelle und die Kombination einer unterschiedlichen Anzahl an Praediktoren zueinander an sowie deren Auswirkungen auf die Modellperformances.

# In[14]:


# Connect to the database
# Ensure correct file path is used
db_path = "/Users/tician/Downloads/source_allergy.db"
conn = sqlite3.connect(db_path)

# Load relevant data
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)

# Aggregate additional predictors
medications_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTALCOST) as TOTAL_MED_COST FROM medications GROUP BY PATIENT;", conn)
procedures_count = pd.read_sql_query(
    "SELECT PATIENT, COUNT(*) as PROCEDURE_COUNT FROM procedures GROUP BY PATIENT;", conn)
procedures_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(BASE_COST) as TOTAL_PROC_COST FROM procedures GROUP BY PATIENT;", conn)
encounter_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTAL_CLAIM_COST) as TOTAL_ENC_COST FROM encounters GROUP BY PATIENT;", conn)
payer_coverage = pd.read_sql_query(
    "SELECT PATIENT, AVG(PAYER_COVERAGE) as AVG_PAYER_COV FROM encounters GROUP BY PATIENT;", conn)

# Close connection
conn.close()

# Calculate age from birthdate
patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
patients_df["AGE"] = (pd.to_datetime("today") -
                      patients_df["BIRTHDATE"]).dt.days // 365

# Merge dataframes
merged_df = patients_df.merge(
    bmi_values, left_on="Id", right_on="PATIENT", how="left")
merged_df = merged_df.merge(medications_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(procedures_count, on="PATIENT", how="left")
merged_df = merged_df.merge(procedures_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(encounter_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(payer_coverage, on="PATIENT", how="left")

# Fill missing values with median for numerical features
fill_values = {
    "BMI": merged_df["BMI"].median(),
    "TOTAL_MED_COST": 0,
    "PROCEDURE_COUNT": 0,
    "TOTAL_PROC_COST": 0,
    "TOTAL_ENC_COST": 0,
    "AVG_PAYER_COV": merged_df["AVG_PAYER_COV"].median(),
}
merged_df.fillna(fill_values, inplace=True)

# Keep relevant columns
final_df = merged_df[["HEALTHCARE_EXPENSES", "BMI", "HEALTHCARE_COVERAGE", "AGE",
                      "TOTAL_MED_COST", "PROCEDURE_COUNT", "TOTAL_PROC_COST", "TOTAL_ENC_COST", "AVG_PAYER_COV"]]

# Define predictor sets
predictor_sets = {
    "1 Predictor": ["BMI"],
    "3 Predictors": ["BMI", "AGE", "HEALTHCARE_COVERAGE"],
    "5 Predictors": ["BMI", "AGE", "HEALTHCARE_COVERAGE", "TOTAL_MED_COST", "TOTAL_ENC_COST"],
}

# Target variable
y = final_df["HEALTHCARE_EXPENSES"]

# Store model performance
performance_results = []

# Train and evaluate models
for name, predictors in predictor_sets.items():
    X = final_df[predictors]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store results
    performance_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
    })

# Convert results to DataFrame and display
performance_df = pd.DataFrame(performance_results)
print(performance_df)


# Der Output zeigt die Leistung einer Linearen Regression mit verschiedenen Prädiktorsätzen, um die Gesundheitskosten (HEALTHCARE_EXPENSES) vorherzusagen:
#
# 1 Predictor (Nur BMI):
#
# Schlechteste Leistung mit hohem MAE (492,799) und niedriger R² (0.13).
# Das Modell kann nur 13% der Varianz erklären → BMI allein reicht nicht aus, um Gesundheitskosten vorherzusagen.
# Hohe Fehlerwerte deuten darauf hin, dass es viele unberücksichtigte Einflussfaktoren gibt.
# 3 Predictors (BMI, AGE, HEALTHCARE_COVERAGE):
#
# Deutliche Verbesserung im Vergleich zu nur BMI.
# R² steigt auf 0.45, was bedeutet, dass 45% der Varianz erklärt werden.
# MAE sinkt auf 345,575, also geringerer durchschnittlicher Fehler.
# Alter und Versicherungsschutz helfen dabei, die Gesundheitskosten besser zu modellieren.
# 5 Predictors (BMI, AGE, HEALTHCARE_COVERAGE, TOTAL_MED_COST, TOTAL_ENC_COST):
#
# Beste Performance mit R² = 0.487 → Das Modell erklärt ca. 49% der Varianz der Gesundheitskosten.
# Niedrigster MAE (336,584) → Verbesserte Genauigkeit.
# Zusätzliche Features (Medikamentenkosten, Encounter-Kosten) helfen, die Gesundheitsausgaben besser vorherzusagen.
#
#
# Wieder einmal schneidet die Lineare Regression schlecht ab.
# Versuchen wir nun erneut nicht lineare Modelle:

# In[16]:


# Connect to the database
# Ensure correct file path is used
db_path = "/Users/tician/Downloads/source_allergy.db"
conn = sqlite3.connect(db_path)

# Load relevant data
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)

# Aggregate additional predictors
medications_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTALCOST) as TOTAL_MED_COST FROM medications GROUP BY PATIENT;", conn)
procedures_count = pd.read_sql_query(
    "SELECT PATIENT, COUNT(*) as PROCEDURE_COUNT FROM procedures GROUP BY PATIENT;", conn)
procedures_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(BASE_COST) as TOTAL_PROC_COST FROM procedures GROUP BY PATIENT;", conn)
encounter_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTAL_CLAIM_COST) as TOTAL_ENC_COST FROM encounters GROUP BY PATIENT;", conn)
payer_coverage = pd.read_sql_query(
    "SELECT PATIENT, AVG(PAYER_COVERAGE) as AVG_PAYER_COV FROM encounters GROUP BY PATIENT;", conn)

# Close connection
conn.close()

# Calculate age from birthdate
patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
patients_df["AGE"] = (pd.to_datetime("today") -
                      patients_df["BIRTHDATE"]).dt.days // 365

# Merge dataframes
merged_df = patients_df.merge(
    bmi_values, left_on="Id", right_on="PATIENT", how="left")
merged_df = merged_df.merge(medications_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(procedures_count, on="PATIENT", how="left")
merged_df = merged_df.merge(procedures_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(encounter_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(payer_coverage, on="PATIENT", how="left")

# Fill missing values with median for numerical features
fill_values = {
    "BMI": merged_df["BMI"].median(),
    "TOTAL_MED_COST": 0,
    "PROCEDURE_COUNT": 0,
    "TOTAL_PROC_COST": 0,
    "TOTAL_ENC_COST": 0,
    "AVG_PAYER_COV": merged_df["AVG_PAYER_COV"].median(),
}
merged_df.fillna(fill_values, inplace=True)

# Keep relevant columns
final_df = merged_df[["HEALTHCARE_EXPENSES", "BMI", "HEALTHCARE_COVERAGE", "AGE",
                      "TOTAL_MED_COST", "PROCEDURE_COUNT", "TOTAL_PROC_COST", "TOTAL_ENC_COST", "AVG_PAYER_COV"]]

# Define predictor sets
predictor_sets = {
    "1 Predictor": ["BMI"],
    "3 Predictors": ["BMI", "AGE", "HEALTHCARE_COVERAGE"],
    "5 Predictors": ["BMI", "AGE", "HEALTHCARE_COVERAGE", "TOTAL_MED_COST", "TOTAL_ENC_COST"],
}

# Target variable
y = final_df["HEALTHCARE_EXPENSES"]

# Store model performance
performance_results = []

# Define models to test
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression": SVR(kernel='rbf', C=1000, gamma=0.1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    for name, predictors in predictor_sets.items():
        X = final_df[predictors]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store results
        performance_results.append({
            "Model": f"{model_name} ({name})",
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2,
        })

# Convert results to DataFrame and display
performance_df = pd.DataFrame(performance_results)
print(performance_df)


# Das beste Modell zur Vorhersage der Gesundheitskosten ist XGBoost, das mit fünf Prädiktoren die höchste Genauigkeit erreicht (R² = 0.999, MAE = 2,606). Es übertrifft damit alle anderen Modelle deutlich. Gradient Boosting liefert ebenfalls gute Ergebnisse (R² = 0.97), während Random Forest zwar hohe Genauigkeit zeigt (R² = 0.996), aber vermutlich overfitted ist. Lineare Regression bleibt mit R² = 0.49 deutlich schwächer, und Support Vector Regression (SVR) schneidet mit negativen R²-Werten am schlechtesten ab. Insgesamt ist XGBoost die beste Wahl, während Gradient Boosting als robuste Alternative in Betracht gezogen werden kann. Zur weiteren Optimierung könnten zusätzliche Prädiktoren oder Hyperparameter-Tuning eingesetzt werden.

# In[2]:


# Verbindung zur Datenbank herstellen
db_path = "/Users/tician/Downloads/source_allergy.db"
conn = sqlite3.connect(db_path)

# Daten abrufen
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
medications_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTALCOST) as TOTAL_MED_COST FROM medications GROUP BY PATIENT;", conn)
encounter_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTAL_CLAIM_COST) as TOTAL_ENC_COST FROM encounters GROUP BY PATIENT;", conn)

# Verbindung schließen
conn.close()

# Alter berechnen
patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
patients_df["AGE"] = (pd.to_datetime("today") -
                      patients_df["BIRTHDATE"]).dt.days // 365

# Daten zusammenführen
merged_df = patients_df.merge(
    bmi_values, left_on="Id", right_on="PATIENT", how="left")
merged_df = merged_df.merge(medications_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(encounter_cost, on="PATIENT", how="left")

# Fehlende Werte mit Median ersetzen
fill_values = {
    "BMI": merged_df["BMI"].median(),
    "TOTAL_MED_COST": 0,
    "TOTAL_ENC_COST": 0,
}
merged_df.fillna(fill_values, inplace=True)

# Features & Zielvariable definieren
features = ["BMI", "AGE", "HEALTHCARE_COVERAGE",
            "TOTAL_MED_COST", "TOTAL_ENC_COST"]
X = merged_df[features]
y = merged_df["HEALTHCARE_EXPENSES"]

# Daten aufteilen (Trainings- & Testset)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Hyperparameter-Suchraum definieren
param_grid = {
    'n_estimators': [100, 200, 300],  # Anzahl der Bäume
    'max_depth': [3, 5, 7],  # Maximale Tiefe der Bäume
    'learning_rate': [0.01, 0.1, 0.2],  # Schrittweite des Algorithmus
    'subsample': [0.8, 1.0],  # Anteil der Stichprobe für jeden Baum
    'colsample_bytree': [0.8, 1.0]  # Anteil der Spalten für jeden Baum
}

# XGBoost initialisieren
xgb = XGBRegressor(random_state=42)

# GridSearchCV für Hyperparameter-Tuning
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',  # Bewertungsmaß: Bestes R²
    cv=3,  # 3-fache Kreuzvalidierung
    verbose=1,  # Zeigt Fortschritt an
    n_jobs=-1  # Nutzt alle verfügbaren Kerne
)

# Modell mit GridSearch trainieren
grid_search.fit(X_train, y_train)

# Beste Parameter extrahieren
best_params = grid_search.best_params_
print("Beste Hyperparameter:", best_params)

# Trainieren mit besten Parametern
optimized_xgb = XGBRegressor(**best_params, random_state=42)
optimized_xgb.fit(X_train, y_train)

# Vorhersagen mit optimiertem Modell
y_pred_optimized = optimized_xgb.predict(X_test)

# Modellbewertung
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# Ergebnisse ausgeben
print("\n📊 **Optimiertes XGBoost Modell**")
print(f"MAE: {mae_optimized:.2f}")
print(f"MSE: {mse_optimized:.2f}")
print(f"RMSE: {rmse_optimized:.2f}")
print(f"R² Score: {r2_optimized:.4f}")


# Nach dem Hyperparameter-Tuning mit GridSearchCV hat XGBoost die folgenden optimalen Parameter gefunden:
#
# colsample_bytree: 1.0 → Alle Features werden für jeden Baum genutzt
# learning_rate: 0.2 → Schnellere Konvergenz durch größere Schrittweite
# max_depth: 3 → Flachere Bäume, die Overfitting reduzieren
# n_estimators: 300 → Mehr Bäume für stabilere Vorhersagen
# subsample: 1.0 → Alle Datenpunkte werden verwendet
#
#
# Sehr hohe Vorhersagegenauigkeit mit R² = 0.9997, was bedeutet, dass das Modell nahezu alle Variationen in den Gesundheitskosten erklärt.
#
# MAE von 4,581 bedeutet, dass das Modell im Durchschnitt nur um 4.5k daneben liegt, was eine enorme Verbesserung im Vergleich zu vorherigen Modellen ist.
#
# RMSE von 10,583 zeigt, dass größere Fehler selten sind, aber dennoch auftreten können.

# Abschliessend schauen wir noch welche Feature am Wichtigsten fuer die Vorhersage der Gesundheitskosten sind.

# In[3]:


# Trainiere XGBoost mit den besten Hyperparametern aus GridSearchCV
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.2,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 1.0
}

# Modell trainieren
optimized_xgb = XGBRegressor(**best_params, random_state=42)
optimized_xgb.fit(X_train, y_train)

# Feature Importance aus dem optimierten Modell extrahieren
feature_importance = optimized_xgb.feature_importances_
feature_names = X.columns

# DataFrame für die Visualisierung erstellen
importance_df = pd.DataFrame(
    {'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Feature Importance Diagramm plotten
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance im optimierten XGBoost Modell")
plt.show()


# Interpretation der Feature Importance
# HEALTHCARE_COVERAGE (~ 40%)
# → Die Versicherung deckt einen erheblichen Teil der Gesundheitskosten ab, was einen starken Einfluss auf die Ausgaben hat.
#
# AGE (~ 35%)
# → Ältere Patienten haben in der Regel höhere Gesundheitskosten, was sich hier widerspiegelt.
#
# TOTAL_ENC_COST (~ 15%)
# → Die Gesamtkosten von Begegnungen (Encounters) sind ein wichtiger Prädiktor, da sie Behandlungen und Diagnosen umfassen.
#
# TOTAL_MED_COST (~ 7%)
# → Medikamentenkosten beeinflussen die Gesamtausgaben, aber weniger stark als Versicherung und Alter.
#
# BMI (~ 2%)
# → Überraschenderweise hat der BMI einen sehr geringen Einfluss, was darauf hindeutet, dass er alleine kein guter Prädiktor für Gesundheitskosten ist.
#
#

# Wir erstellen nun eine Fake-Datenbank, um dieses Modell zu validieren.
#
# ANMERKUNG: Es wurde bewusst darauf verzichtet ein Public-Dataset hierfuer zu verwenden (da keine der Verfuegbaren Datenbanken unsere Praediktoranforderungen zur Gaenze erfuellt).

# In[5]:


get_ipython().system('pip install faker')  # noqa: F821


# In[13]:


# Faker initialisieren
fake = Faker()
np.random.seed(42)

# Verbindung zur Fake-Datenbank herstellen
ultra_realistic_fake_db_path = "ultra_realistic_fake_healthcare.db"
conn = sqlite3.connect(ultra_realistic_fake_db_path)
cursor = conn.cursor()

# Tabellen mit erweiterten Features erstellen
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

# Anzahl der Patienten erhöhen
num_patients = 10000

# Listen für die Fake-Daten
patients_data = []
observations_data = []
medications_data = []
encounters_data = []

# Mögliche Werte für Gesundheitsstatus & Beschäftigung
chronic_conditions_list = ["None", "Diabetes", "COPD",
                           "Cancer", "Hypertension", "Depression", "Multiple Conditions"]
employment_status_list = ["Unemployed", "Employed", "Self-Employed", "Retired"]
gender_list = ["Male", "Female", "Other"]

# Generierung realistisch verteilter Gesundheitsdaten
for i in range(1, num_patients + 1):
    # Alter simulieren
    birthdate = fake.date_of_birth(
        minimum_age=20, maximum_age=90).strftime("%Y-%m-%d")
    age = 2025 - int(birthdate[:4])

    # Versicherungsschutz (jüngere Menschen haben oft schlechtere Abdeckung)
    healthcare_coverage = np.clip(
        np.random.normal(0.7 - (age / 200), 0.2), 0.1, 1.0)

    # Einkommen simulieren (höheres Alter → leicht sinkendes Einkommen)
    income = np.clip(np.random.normal(
        50000 - (age * 300), 15000), 15000, 150000)

    # BMI (höhere Werte für ältere Personen, mit natürlicher Streuung)
    bmi = np.clip(np.random.normal(22 + (age / 50), 3), 18, 40)

    # Geschlecht zufällig auswählen
    gender = np.random.choice(gender_list)

    # Beschäftigungsstatus zufällig auswählen (ältere Menschen öfter Rentner)
    employment_status = np.random.choice(employment_status_list, p=[
                                         0.1, 0.6, 0.2, 0.1] if age < 65 else [0.05, 0.2, 0.15, 0.6])

    # Chronische Erkrankungen mit altersabhängigen Wahrscheinlichkeiten
    chronic_condition = np.random.choice(
        chronic_conditions_list,
        p=[0.5 - (age / 200), 0.2, 0.1, 0.05, 0.1, 0.05, 1.0 -
           (0.5 - (age / 200) + 0.2 + 0.1 + 0.05 + 0.1 + 0.05)]
    )

    # Krankenhausaufenthalte steigen mit Alter & Vorerkrankungen
    hospitalizations = np.clip(np.random.poisson(
        0.5 + (0.3 if chronic_condition != "None" else 0) + (age / 80)), 0, 10)

    # Raucherstatus (ältere Menschen rauchen häufiger in historischen Daten)
    smoker = np.random.choice([0, 1], p=[0.75, 0.25]
                              if age > 50 else [0.85, 0.15])

    # Notaufnahme-Besuche steigen mit Alter & Rauchen
    emergency_visits = np.clip(np.random.poisson(
        0.2 + (smoker * 0.2) + (hospitalizations * 0.1)), 0, 5)

    # Medikamentenkosten hängen von Alter, BMI & Vorerkrankungen ab
    medication_cost = np.clip(np.random.normal(
        500 + (age * 10) + (bmi * 5) + (hospitalizations * 200), 500), 0, 15000)

    # Behandlungskosten steigen mit Alter, Vorerkrankungen & Raucherstatus
    encounter_cost = np.clip(np.random.normal(
        2000 + (age * 20) + (bmi * 10) + (hospitalizations * 500) + (smoker * 1000), 1000), 1000, 70000)

    # Gesundheitskosten hängen von Versicherung, Einkommen, Vorerkrankungen & Raucherstatus ab
    base_expenses = encounter_cost + medication_cost + \
        (hospitalizations * 5000) + (emergency_visits * 2000)
    healthcare_expenses = np.clip(base_expenses * (1 - healthcare_coverage) * (
        1 + smoker * 0.2) * (1 - income / 200000), 5000, 200000)

    # Daten speichern
    patients_data.append((healthcare_expenses, healthcare_coverage, birthdate, income,
                         chronic_condition, hospitalizations, smoker, gender, employment_status, emergency_visits))
    observations_data.append((i, bmi, '39156-5'))
    medications_data.append((i, medication_cost))
    encounters_data.append((i, encounter_cost))

# Daten in die Tabellen einfügen
cursor.executemany("INSERT INTO patients (HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE, INCOME, CHRONIC_CONDITIONS, HOSPITALIZATIONS, SMOKER, GENDER, EMPLOYMENT_STATUS, EMERGENCY_VISITS) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", patients_data)
cursor.executemany(
    "INSERT INTO observations (PATIENT, VALUE, CODE) VALUES (?, ?, ?)", observations_data)
cursor.executemany(
    "INSERT INTO medications (PATIENT, TOTALCOST) VALUES (?, ?)", medications_data)
cursor.executemany(
    "INSERT INTO encounters (PATIENT, TOTAL_CLAIM_COST) VALUES (?, ?)", encounters_data)

# Änderungen speichern und Verbindung schließen
conn.commit()
conn.close()

print("✅ realistische Fake-Datenbank wurde erfolgreich erstellt: ultra_realistic_fake_healthcare.db")


# In[15]:


# Verbindung zur Fake-Datenbank herstellen
conn = sqlite3.connect("ultra_realistic_fake_healthcare.db")

# Daten abrufen
patients_df = pd.read_sql_query(
    "SELECT Id, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, BIRTHDATE, INCOME, CHRONIC_CONDITIONS, HOSPITALIZATIONS, SMOKER, GENDER, EMPLOYMENT_STATUS, EMERGENCY_VISITS FROM patients;", conn)
bmi_values = pd.read_sql_query(
    "SELECT PATIENT, VALUE as BMI FROM observations WHERE CODE = '39156-5';", conn)
medications_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTALCOST) as TOTAL_MED_COST FROM medications GROUP BY PATIENT;", conn)
encounter_cost = pd.read_sql_query(
    "SELECT PATIENT, SUM(TOTAL_CLAIM_COST) as TOTAL_ENC_COST FROM encounters GROUP BY PATIENT;", conn)

# Verbindung schließen
conn.close()

# Alter berechnen
patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
patients_df["AGE"] = (pd.to_datetime("today") -
                      patients_df["BIRTHDATE"]).dt.days // 365

# Daten zusammenführen
merged_df = patients_df.merge(
    bmi_values, left_on="Id", right_on="PATIENT", how="left")
merged_df = merged_df.merge(medications_cost, on="PATIENT", how="left")
merged_df = merged_df.merge(encounter_cost, on="PATIENT", how="left")

# Fehlende Werte mit Median ersetzen
fill_values = {
    "BMI": merged_df["BMI"].median(),
    "TOTAL_MED_COST": 0,
    "TOTAL_ENC_COST": 0,
}
merged_df.fillna(fill_values, inplace=True)

# Sicherstellen, dass alle numerischen Variablen auch numerische Typen haben
numeric_columns = ["HOSPITALIZATIONS", "SMOKER", "EMERGENCY_VISITS"]
for col in numeric_columns:
    merged_df[col] = pd.to_numeric(
        merged_df[col], errors="coerce").fillna(0).astype(int)

# Kategorische Variablen in numerische Werte umwandeln
merged_df["GENDER"] = merged_df["GENDER"].astype("category").cat.codes
merged_df["EMPLOYMENT_STATUS"] = merged_df["EMPLOYMENT_STATUS"].astype(
    "category").cat.codes
merged_df["CHRONIC_CONDITIONS"] = merged_df["CHRONIC_CONDITIONS"].astype(
    "category").cat.codes

# Features & Zielvariable definieren
features = ["BMI", "AGE", "HEALTHCARE_COVERAGE", "TOTAL_MED_COST", "TOTAL_ENC_COST",
            "INCOME", "HOSPITALIZATIONS", "SMOKER", "GENDER", "EMPLOYMENT_STATUS", "EMERGENCY_VISITS", "CHRONIC_CONDITIONS"]
X = merged_df[features]
y = merged_df["HEALTHCARE_EXPENSES"]

# Daten aufteilen (Trainings- & Testset)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Bestes XGBoost-Modell trainieren
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.2,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 1.0
}

optimized_xgb = XGBRegressor(**best_params, random_state=42)
optimized_xgb.fit(X_train, y_train)

# Vorhersagen mit optimiertem Modell
y_pred_optimized = optimized_xgb.predict(X_test)

# Modellbewertung
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# Ergebnisse ausgeben
print("\n📊 **Optimiertes XGBoost Modell auf ultra-realistischen Fake-Daten**")
print(f"MAE: {mae_optimized:.2f}")
print(f"MSE: {mse_optimized:.2f}")
print(f"RMSE: {rmse_optimized:.2f}")
print(f"R² Score: {r2_optimized:.4f}")


# Das optimierte XGBoost-Modell zeigt auf der ultra-realistischen Fake-Datenbank eine solide Leistung mit einer guten Vorhersagegenauigkeit. Die durchschnittliche Abweichung der Vorhersagen liegt bei etwa 1.750 USD (MAE), während der mittlere Fehler (RMSE) bei ca. 2.728 USD liegt. Der R²-Wert von 0.65 zeigt, dass das Modell etwa 65% der Varianz der Gesundheitskosten erklären kann, was für ein solches Vorhersagemodell im Gesundheitswesen eine sehr gute Genauigkeit darstellt.
#
# Dennoch gibt es Potenzial für weitere Optimierungen. Zusätzliche Features wie die Schwere der Vorerkrankungen, die Art der Medikation oder eine Operationshistorie könnten die Vorhersagegenauigkeit weiter steigern. Außerdem könnte eine Hyperparameter-Optimierung durch GridSearchCV oder Bayesian Optimization helfen, die Modellleistung noch weiter zu verbessern.
