from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Get the absolute path of the directory where the script is located ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load Model and Encoders ---
try:
    model_path = os.path.join(script_dir, 'models', 'life_expectancy_model.pkl')
    encoders_path = os.path.join(script_dir, 'models', 'label_encoders.pkl')
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    print("✅ Model and encoders loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model or encoder files not found. Please run the training script first.")
    model = None
    encoders = None

# --- Homepage Route ---
@app.route('/')
def home():
    """Renders the main input form."""
    return render_template('index.html')

# --- Prediction Route (Handles both API and Form submissions) ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        # Check if the request is from the web form or an API call
        if request.is_json:
            json_data = request.get_json()
        else:
            json_data = request.form.to_dict()
            # Convert numeric fields from form strings to appropriate types
            numeric_fields = ['Age', 'Height', 'Weight', 'BMI', 'Resting Heart Rate', 'SpO2', 
                              'Sleep Duration', 'Daily Activity', 'Stress Score', 'Anxiety Level', 
                              'Air Quality Index', 'Work Hours']
            for field in numeric_fields:
                if field in json_data:
                    json_data[field] = pd.to_numeric(json_data[field])

        # --- Shared Prediction Logic ---
        input_df = pd.DataFrame([json_data])
        
        bp_split = input_df['Blood Pressure'].str.split('/', expand=True)
        input_df['Systolic_Pressure'] = pd.to_numeric(bp_split[0])
        input_df['Diastolic_Pressure'] = pd.to_numeric(bp_split[1])
        input_df = input_df.drop(columns=['Blood Pressure'])

        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col])

        input_df.columns = [col.replace(' ', '_') for col in input_df.columns]
        raw_model_prediction = model.predict(input_df)[0]
        
        base_le = 72.0
        adjustments = []
        if json_data['Smoking'] in ['Daily', 'Occasionally']: adjustments.append({'factor': 'Smoking', 'impact': -7.0})
        if json_data['Alcohol'] == 'Daily': adjustments.append({'factor': 'Daily Alcohol', 'impact': -5.0})
        if json_data['Exercise Type'] != 'None': adjustments.append({'factor': 'Regular Exercise', 'impact': 4.5})
        else: adjustments.append({'factor': 'Lack of Exercise', 'impact': -4.0})
        if json_data['Diet Quality'] == 'High': adjustments.append({'factor': 'High Quality Diet', 'impact': 5.0})
        elif json_data['Diet Quality'] == 'Low': adjustments.append({'factor': 'Low Quality Diet', 'impact': -5.0})
        if json_data['Sleep Duration'] < 6: adjustments.append({'factor': 'Insufficient Sleep', 'impact': -3.0})
        if json_data['Stress Score'] > 7: adjustments.append({'factor': 'High Stress', 'impact': -3.5})

        total_adjustment = sum(item['impact'] for item in adjustments)
        formula_le = base_le + total_adjustment
        blended_prediction = (raw_model_prediction * 0.6) + (formula_le * 0.4)

        current_age = json_data['Age']
        final_prediction = blended_prediction
        years_remaining = blended_prediction - current_age

        if current_age < 70 and years_remaining < 8:
            final_prediction = current_age + 8
        elif current_age >= 70 and years_remaining < 4:
            final_prediction = current_age + 4
        
        positive_factors = [adj for adj in adjustments if adj['impact'] > 0]
        negative_factors = [adj for adj in adjustments if adj['impact'] < 0]

        response_data = {
            'predicted_life_expectancy': round(final_prediction, 1),
            'base_model_prediction': round(raw_model_prediction, 1),
            'positive_factors': positive_factors,
            'negative_factors': negative_factors
        }

        # --- Return response based on request type ---
        if request.is_json:
            return jsonify(response_data)
        else:
            return render_template('result.html', data=response_data)

    except Exception as e:
        error_message = f'Prediction failed: "{e}"'
        if request.is_json:
            return jsonify({'error': error_message}), 500
        else:
            return render_template('result.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)

