import requests
import json

# The URL where your Flask API is running
API_URL = "http://127.0.0.1:5000/predict"

# --- MODIFY THE DATA BELOW ---
# This dictionary contains all the health data for the prediction.
# Change these values to test different scenarios.
user_data = {
    "Age": 45,
    "Gender": "Male",
    "Ethnicity": "North Indian",
    "Height": 175,
    "Weight": 90,
    "BMI": 29.4,
    "Blood Pressure": "140/90", # Make sure this is a string in 'systolic/diastolic' format
    "Resting Heart Rate": 80,
    "SpO2": 99,
    "Diet Type": "Non-Vegetarian",
    "Protein Intake": "Mediu0m",
    "Junk Food Frequency": "High",
    "Sugar Intake": "High",
    "Diet Quality": "Low",
    "Smoking": "Daily",
    "Alcohol": "Occasionally",
    "Sleep Duration": 6.0,
    "Sleep Quality": "Poor",
    "Daily Activity": 3000,
    "Exercise Type": "Walking", # <-- CHANGED FROM "None" to "Walking"
    "Family History": "Heart Disease",
    "Existing Conditions": "Hypertension",
    "Stress Score": 8,
    "Anxiety Level": 7,
    "Air Quality Index": 200,
    "Exposure": "High",
    "Urban/Rural": "Urban",
    "Work Hours": 10,
    "State": "Delhi",
    "City": "Delhi"
}
# --- NO NEED TO MODIFY BELOW THIS LINE ---

# Set the headers to indicate we are sending JSON data
headers = {
    "Content-Type": "application/json"
}

print("▶️  Sending data to the model for prediction...")

try:
    # Send the POST request to the API
    response = requests.post(API_URL, headers=headers, json=user_data)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        prediction = response.json()
        print("\n✅ Prediction Received Successfully!")
        print("-----------------------------------------")
        # Print the predicted life expectancy
        le = prediction.get('predicted_life_expectancy')
        print(f"  Predicted Life Expectancy: {le} years")
        print("-----------------------------------------")
    else:
        # If the server returned an error, print the details
        print(f"\n❌ Error: The server returned status code {response.status_code}")
        print("   Response from server:")
        print(f"   {response.json()}")

except requests.exceptions.ConnectionError:
    print("\n❌ Connection Error: Could not connect to the server.")
    print("   Please ensure your Flask app (`app.py`) is running in another terminal.")

