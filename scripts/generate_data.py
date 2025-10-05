import pandas as pd
import numpy as np
import random
import os # <-- Import the os module

# --- State-Specific Health & Demographic Data ---
# Data gathered from various public sources (NITI Aayog, Census data, CPCB)
# Note: This is representative data for creating a realistic synthetic dataset.
STATE_DATA = {
    'Andhra Pradesh': {'avg_le': 70.0, 'avg_aqi': 120, 'median_age': 27, 'cities': ['Visakhapatnam', 'Vijayawada', 'Tirupati']},
    'Arunachal Pradesh': {'avg_le': 70.3, 'avg_aqi': 40, 'median_age': 24, 'cities': ['Itanagar', 'Naharlagun']},
    'Assam': {'avg_le': 67.2, 'avg_aqi': 150, 'median_age': 25, 'cities': ['Guwahati', 'Dibrugarh', 'Silchar']},
    'Bihar': {'avg_le': 69.5, 'avg_aqi': 180, 'median_age': 20, 'cities': ['Patna', 'Gaya', 'Bhagalpur']},
    'Chhattisgarh': {'avg_le': 68.9, 'avg_aqi': 140, 'median_age': 23, 'cities': ['Raipur', 'Bhilai', 'Bilaspur']},
    'Goa': {'avg_le': 74.5, 'avg_aqi': 70, 'median_age': 32, 'cities': ['Panaji', 'Margao']},
    'Gujarat': {'avg_le': 72.8, 'avg_aqi': 160, 'median_age': 27, 'cities': ['Ahmedabad', 'Surat', 'Vadodara']},
    'Haryana': {'avg_le': 72.3, 'avg_aqi': 190, 'median_age': 26, 'cities': ['Faridabad', 'Gurugram', 'Chandigarh']},
    'Himachal Pradesh': {'avg_le': 74.6, 'avg_aqi': 80, 'median_age': 29, 'cities': ['Shimla', 'Dharamshala']},
    'Jharkhand': {'avg_le': 69.4, 'avg_aqi': 150, 'median_age': 22, 'cities': ['Ranchi', 'Jamshedpur', 'Dhanbad']},
    'Karnataka': {'avg_le': 72.8, 'avg_aqi': 100, 'median_age': 28, 'cities': ['Bengaluru', 'Mysore', 'Mangalore']},
    'Kerala': {'avg_le': 77.8, 'avg_aqi': 60, 'median_age': 33, 'cities': ['Thiruvananthapuram', 'Kochi', 'Kozhikode']},
    'Madhya Pradesh': {'avg_le': 69.4, 'avg_aqi': 130, 'median_age': 24, 'cities': ['Indore', 'Bhopal', 'Gwalior']},
    'Maharashtra': {'avg_le': 73.6, 'avg_aqi': 110, 'median_age': 29, 'cities': ['Mumbai', 'Pune', 'Nagpur']},
    'Manipur': {'avg_le': 75.0, 'avg_aqi': 50, 'median_age': 24, 'cities': ['Imphal']},
    'Meghalaya': {'avg_le': 72.7, 'avg_aqi': 60, 'median_age': 21, 'cities': ['Shillong']},
    'Mizoram': {'avg_le': 74.3, 'avg_aqi': 30, 'median_age': 23, 'cities': ['Aizawl']},
    'Nagaland': {'avg_le': 73.4, 'avg_aqi': 50, 'median_age': 23, 'cities': ['Kohima', 'Dimapur']},
    'Odisha': {'avg_le': 69.8, 'avg_aqi': 140, 'median_age': 26, 'cities': ['Bhubaneswar', 'Cuttack', 'Rourkela']},
    'Punjab': {'avg_le': 74.4, 'avg_aqi': 170, 'median_age': 29, 'cities': ['Ludhiana', 'Amritsar']},
    'Rajasthan': {'avg_le': 70.8, 'avg_aqi': 160, 'median_age': 24, 'cities': ['Jaipur', 'Jodhpur', 'Udaipur']},
    'Sikkim': {'avg_le': 73.5, 'avg_aqi': 70, 'median_age': 25, 'cities': ['Gangtok']},
    'Tamil Nadu': {'avg_le': 73.8, 'avg_aqi': 90, 'median_age': 30, 'cities': ['Chennai', 'Coimbatore', 'Madurai']},
    'Telangana': {'avg_le': 72.7, 'avg_aqi': 110, 'median_age': 27, 'cities': ['Hyderabad', 'Warangal']},
    'Tripura': {'avg_le': 74.6, 'avg_aqi': 100, 'median_age': 26, 'cities': ['Agartala']},
    'Uttar Pradesh': {'avg_le': 68.7, 'avg_aqi': 200, 'median_age': 22, 'cities': ['Lucknow', 'Kanpur', 'Ghaziabad']},
    'Uttarakhand': {'avg_le': 73.5, 'avg_aqi': 120, 'median_age': 25, 'cities': ['Dehradun', 'Haridwar']},
    'West Bengal': {'avg_le': 72.8, 'avg_aqi': 160, 'median_age': 28, 'cities': ['Kolkata', 'Siliguri', 'Darjeeling']},
    'Delhi': {'avg_le': 75.3, 'avg_aqi': 250, 'median_age': 29, 'cities': ['Delhi']}
}

def calculate_diet_quality(row):
    """Calculates a 'Diet Quality' score based on other dietary inputs."""
    score = 0
    # Score protein intake
    if row['Protein Intake'] == 'High': score += 2
    elif row['Protein Intake'] == 'Low': score -= 2
    
    # Score junk food frequency
    if row['Junk Food Frequency'] == 'Never': score += 2
    elif row['Junk Food Frequency'] == 'Low': score += 1
    elif row['Junk Food Frequency'] == 'High': score -= 2
    
    # Score sugar intake
    if row['Sugar Intake'] == 'Low': score += 2
    elif row['Sugar Intake'] == 'High': score -= 2
    
    # Determine quality based on final score
    if score >= 3:
        return 'High'
    elif score <= -3:
        return 'Low'
    else:
        return 'Medium'

def calculate_logical_le(row, state_avg_le):
    """Calculates a logical life expectancy using state average as a baseline."""
    base_le = state_avg_le
    
    if row['Smoking'] in ['Daily', 'Occasionally']: base_le -= 7
    if row['Alcohol'] == 'Daily': base_le -= 5
    if row['Exercise Type'] != 'None': base_le += 5
    else: base_le -= 4
    # Use the derived Diet Quality for LE calculation
    if row['Diet Quality'] == 'High': base_le += 6
    elif row['Diet Quality'] == 'Low': base_le -= 6
    if row['Sleep Duration'] < 6: base_le -= 3
    if row['Stress Score'] > 7: base_le -= 4
    if row['Family History'] != 'None': base_le -= 3
    if row['BMI'] > 30: base_le -= (row['BMI'] - 30) * 0.5
    
    if row['Air Quality Index'] > (STATE_DATA[row['State']]['avg_aqi'] + 50):
        base_le -= 2.5
        
    base_le += random.uniform(-2, 2)
    return max(base_le, row['Age'] + 5)

def generate_health_data(num_records=20000):
    """Generates a DataFrame with synthetic health data based on state-specific stats."""
    print(f"Generating {num_records} records...")
    
    genders = ['Male', 'Female']
    ethnicities = ['North Indian', 'South Indian', 'Bengali', 'Gujarati', 'Punjabi'] # 'Maharashtrian' removed
    diet_types = ['Non-Vegetarian', 'Vegetarian', 'Vegan', 'Mixed']
    protein_intakes = ['High', 'Medium', 'Low']
    junk_food_freqs = ['High', 'Medium', 'Low', 'Never']
    sugar_intakes = ['High', 'Medium', 'Low']
    smoking_statuses = ['Never', 'Occasionally', 'Daily']
    alcohol_consumptions = ['Never', 'Occasionally', 'Daily']
    sleep_qualities = ['Good', 'Average', 'Poor']
    exercise_types = ['Gym', 'Walking', 'Yoga', 'None']
    family_histories = ['None', 'Diabetes', 'Heart Disease', 'Cancer']
    existing_conditions = ['None', 'Hypertension', 'Asthma', 'COPD']
    exposures = ['Low', 'Medium', 'High']
    urban_statuses = ['Urban', 'Rural']
    
    data = []
    for _ in range(num_records):
        state = random.choice(list(STATE_DATA.keys()))
        state_info = STATE_DATA[state]
        
        row = {
            'Age': max(18, int(random.gauss(state_info['median_age'], 5))),
            'Gender': random.choice(genders),
            'Ethnicity': random.choice(ethnicities),
            'Height': random.randint(150, 190),
            'Weight': random.randint(50, 110),
            'Blood Pressure': f"{random.randint(110, 160)}/{random.randint(70, 100)}",
            'Resting Heart Rate': random.randint(60, 100),
            'SpO2': random.randint(95, 100),
            'Diet Type': random.choice(diet_types),
            'Protein Intake': random.choice(protein_intakes),
            'Junk Food Frequency': random.choice(junk_food_freqs),
            'Sugar Intake': random.choice(sugar_intakes),
            # Diet Quality is now derived, not random
            'Smoking': random.choice(smoking_statuses),
            'Alcohol': random.choice(alcohol_consumptions),
            'Sleep Duration': round(random.uniform(5.0, 9.0), 1),
            'Sleep Quality': random.choice(sleep_qualities),
            'Daily Activity': random.randint(2000, 12000),
            'Exercise Type': random.choice(exercise_types),
            'Family History': random.choice(family_histories),
            'Existing Conditions': random.choice(existing_conditions),
            'Stress Score': random.randint(1, 10),
            'Anxiety Level': random.randint(1, 10),
            'Air Quality Index': max(20, int(random.gauss(state_info['avg_aqi'], 30))),
            'Exposure': random.choice(exposures),
            'Urban/Rural': random.choice(urban_statuses),
            'Work Hours': random.randint(6, 12),
            'State': state,
            'City': random.choice(state_info['cities'])
        }
        # Derive Diet Quality based on the formula
        row['Diet Quality'] = calculate_diet_quality(row)
        
        row['BMI'] = round(row['Weight'] / ((row['Height'] / 100) ** 2), 1)
        row['Life Expectancy'] = round(calculate_logical_le(row, state_info['avg_le']), 1)
        data.append(row)
        
    df = pd.DataFrame(data)
    # Ensure columns are in a consistent order
    final_cols = [
        'Age', 'Gender', 'Ethnicity', 'Height', 'Weight', 'BMI', 'Blood Pressure', 
        'Resting Heart Rate', 'SpO2', 'Diet Type', 'Protein Intake', 
        'Junk Food Frequency', 'Sugar Intake', 'Diet Quality', 'Smoking', 'Alcohol', 
        'Sleep Duration', 'Sleep Quality', 'Daily Activity', 'Exercise Type', 
        'Family History', 'Existing Conditions', 'Stress Score', 'Anxiety Level', 
        'Air Quality Index', 'Exposure', 'Urban/Rural', 'Work Hours', 'State', 'City', 
        'Life Expectancy'
    ]
    df = df[final_cols]
    
    print("Data generation complete.")
    return df

if __name__ == '__main__':
    new_data = generate_health_data(num_records=25000)
    output_path = '../data/wellwise_health_data_v5_final.csv'
    
    # --- NEW: Automatically create the directory before saving ---
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_data.to_csv(output_path, index=False)
    print(f"✅ New dataset with 25,000 records saved to '{output_path}'")

