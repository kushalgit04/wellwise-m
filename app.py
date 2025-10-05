from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# --- Load Model and Encoders ---
try:
    model = joblib.load('models/life_expectancy_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    print("✅ Model and encoders loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model or encoder files not found.")
    model = None
    encoders = None

@app.route('/')
def home():
    """Renders the main input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        # --- FIX: Handle both JSON (from API) and Form (from web) data ---
        if request.is_json:
            # Request is from an API client like test_api.py
            form_data = request.get_json()
            is_api_call = True
        else:
            # Request is from the HTML web form
            form_data = request.form.to_dict()
            is_api_call = False

        # Convert numeric fields from string to the correct type
        numeric_fields = ['Age', 'Height', 'Weight', 'BMI', 'Resting Heart Rate', 'SpO2', 
                          'Sleep Duration', 'Daily Activity', 'Stress Score', 'Anxiety Level', 
                          'Air Quality Index', 'Work Hours']
        for field in numeric_fields:
            if form_data.get(field):
                form_data[field] = pd.to_numeric(form_data[field])

        input_df = pd.DataFrame([form_data])

        # --- Data Cleaning and Encoding ---
        bp_split = input_df['Blood Pressure'].str.split('/', expand=True)
        input_df['Systolic_Pressure'] = pd.to_numeric(bp_split[0])
        input_df['Diastolic_Pressure'] = pd.to_numeric(bp_split[1])
        input_df = input_df.drop(columns=['Blood Pressure'])

        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col])

        input_df.columns = [col.replace(' ', '_') for col in input_df.columns]

        # --- Prediction and Logic ---
        raw_model_prediction = model.predict(input_df)[0]
        
        base_le = 72.0
        adjustments = []
        
        if form_data.get('Smoking') in ['Daily', 'Occasionally']: adjustments.append({'factor': 'Smoking', 'impact': -7.0})
        if form_data.get('Alcohol') == 'Daily': adjustments.append({'factor': 'Daily Alcohol', 'impact': -5.0})
        if form_data.get('Exercise Type') != 'None': adjustments.append({'factor': 'Regular Exercise', 'impact': 4.5})
        else: adjustments.append({'factor': 'Lack of Exercise', 'impact': -4.0})
        if form_data.get('Diet Quality') == 'High': adjustments.append({'factor': 'High Quality Diet', 'impact': 5.0})
        elif form_data.get('Diet Quality') == 'Low': adjustments.append({'factor': 'Low Quality Diet', 'impact': -5.0})
        
        total_adjustment = sum(item['impact'] for item in adjustments)
        formula_le = base_le + total_adjustment
        
        blended_prediction = (raw_model_prediction * 0.6) + (formula_le * 0.4)
        
        current_age = form_data['Age']
        final_prediction = blended_prediction
        years_remaining = blended_prediction - current_age

        if current_age < 70 and years_remaining < 8: final_prediction = current_age + 8
        elif current_age >= 70 and years_remaining < 4: final_prediction = current_age + 4
        
        # --- Return response based on the type of request ---
        if is_api_call:
            # Return a full JSON response for API clients
            return jsonify({
                'predicted_life_expectancy': round(final_prediction, 1),
                'base_model_prediction': round(raw_model_prediction, 1),
                'positive_factors': [adj for adj in adjustments if adj['impact'] > 0],
                'negative_factors': [adj for adj in adjustments if adj['impact'] < 0]
            })
        else:
            # Render the HTML results page for web form users
            chart_data = {
                "prediction": round(final_prediction, 1),
                "current_age": current_age,
                "adjustments": adjustments,
                "health_scores": {
                    "Diet": 5 - (['Low', 'Medium', 'High'].index(form_data.get('Diet Quality')) * 2),
                    "Exercise": 5 if form_data.get('Exercise Type') != 'None' else 1,
                    "Sleep": form_data.get('Sleep Duration') / 9 * 5,
                    "Stress": 6 - (form_data.get('Stress Score') / 2),
                    "Habits": 5 - (['Never', 'Occasionally', 'Daily'].index(form_data.get('Smoking'))) - (['Never', 'Occasionally', 'Daily'].index(form_data.get('Alcohol')))
                }
            }
            return render_template('result.html', chart_data=chart_data)

    except Exception as e:
        return jsonify({'error': f"An error occurred: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True)

