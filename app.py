from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = 'optimized_xgb_model.pkl'
optimized_xgb = joblib.load(model_path)

# Read the index.html content from the root directory
with open('index.html', 'r') as file:
    index_html = file.read()


@app.route('/')
def home():
    return render_template_string(index_html)


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    bmi = float(request.form['bmi'])
    healthcare_coverage = float(request.form['healthcare_coverage'])
    age = int(request.form['age'])
    total_med_cost = float(request.form['total_med_cost'])
    total_enc_cost = float(request.form['total_enc_cost'])
    income = float(request.form['income'])
    hospitalizations = int(request.form['hospitalizations'])
    smoker = int(request.form['smoker'])
    gender = int(request.form['gender'])
    employment_status = int(request.form['employment_status'])
    emergency_visits = int(request.form['emergency_visits'])
    chronic_conditions = int(request.form['chronic_conditions'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[bmi, age, healthcare_coverage, total_med_cost, total_enc_cost, income,
                                hospitalizations, smoker, gender, employment_status, emergency_visits, chronic_conditions]],
                              columns=['BMI', 'AGE', 'HEALTHCARE_COVERAGE', 'TOTAL_MED_COST', 'TOTAL_ENC_COST', 'INCOME',
                                       'HOSPITALIZATIONS', 'SMOKER', 'GENDER', 'EMPLOYMENT_STATUS', 'EMERGENCY_VISITS', 'CHRONIC_CONDITIONS'])

    # Make predictions
    prediction = optimized_xgb.predict(input_data)[0]

    return render_template_string(index_html, prediction=prediction)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)  # Remove debug=True
