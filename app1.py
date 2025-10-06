from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # Allow CORS for frontend requests

# --- Load Model and Encoders ---
try:
    model = joblib.load('models/life_expectancy_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    print("✅ Model and encoders loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model or encoder files not found.")
    model = None
    encoders = None


# --- Root Route (Optional) ---
@app.route('/')
def home():
    return jsonify({"message": "Life Expectancy Prediction API is running!"})


# --- Predict Route ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders:
        return jsonify({'error': 'Model or encoders not loaded on server.'}), 500

    try:
        # ✅ Step 1: Receive JSON from frontend
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data received'}), 400

        # ✅ Step 2: Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # ✅ Step 3: Handle Blood Pressure (split into two numeric columns)
        if 'Blood Pressure' in input_df.columns:
            bp_split = input_df['Blood Pressure'].str.split('/', expand=True)
            input_df['Systolic_Pressure'] = pd.to_numeric(bp_split[0], errors='coerce')
            input_df['Diastolic_Pressure'] = pd.to_numeric(bp_split[1], errors='coerce')
            input_df.drop(columns=['Blood Pressure'], inplace=True)

        # ✅ Step 4: Convert numeric fields
        numeric_fields = [
            'Age', 'Height', 'Weight', 'BMI', 'Resting Heart Rate', 'SpO2',
            'Sleep Duration', 'Daily Activity', 'Stress Score', 'Anxiety Level',
            'Air Quality Index', 'Work Hours'
        ]
        for field in numeric_fields:
            if field in input_df.columns:
                input_df[field] = pd.to_numeric(input_df[field], errors='coerce')

        # ✅ Step 5: Apply label encoders (for categorical fields)
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        # ✅ Step 6: Make prediction
        prediction = model.predict(input_df)[0]

        # ✅ Step 7: Return JSON response
        return jsonify({
            'predicted_life_expectancy': round(float(prediction), 2),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Run the Flask app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
