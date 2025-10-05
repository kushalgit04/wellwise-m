import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
import os

# --- Get the absolute path of the directory where the script is located ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load Data using a robust relative path ---
print("Loading new, high-quality dataset...")
# Construct a path to the data file relative to the script's location
csv_path = os.path.join(script_dir, '..', 'data', 'wellwise_health_data_v5_final.csv')
df = pd.read_csv(csv_path)
print("Data loaded successfully.")

# --- 2. Data Cleaning: Process Blood Pressure ---
print("Cleaning 'Blood Pressure' column...")
# Split '120/80' into two columns
bp_split = df['Blood Pressure'].str.split('/', expand=True)

# Create new numeric columns for systolic and diastolic pressure
df['Systolic_Pressure'] = pd.to_numeric(bp_split[0])
df['Diastolic_Pressure'] = pd.to_numeric(bp_split[1])

# Drop the original text-based 'Blood Pressure' column
df = df.drop(columns=['Blood Pressure'])
print("Blood pressure processed successfully.")


# --- 3. Prepare Features and Target ---
X = df.drop(columns=['Life Expectancy'], axis=1) # The 'Impacts' column no longer exists
y = df['Life Expectancy']

# --- 4. Encode Categorical Features ---
print("Encoding categorical features...")
# Ensure all categorical columns are present
categorical_cols = [
    'Gender', 'Ethnicity', 'Diet Type', 'Protein Intake', 'Junk Food Frequency',
    'Sugar Intake', 'Diet Quality', 'Smoking', 'Alcohol', 'Sleep Quality',
    'Exercise Type', 'Family History', 'Existing Conditions', 'Exposure',
    'Urban/Rural', 'State', 'City'
]
encoder_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoder_dict[col] = le
print("Encoding complete.")

# --- 5. Split and Train Model ---
print("Splitting data and retraining the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMRegressor(objective='regression', metric='rmse', random_state=42)
model.fit(X_train, y_train)
print("Model retraining complete.")

# --- 6. Save the Final Model and Encoders ---
# Construct a path to the models directory relative to the script's location
output_dir = os.path.join(script_dir, '..', 'models')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(model, os.path.join(output_dir, 'life_expectancy_model.pkl'))
joblib.dump(encoder_dict, os.path.join(output_dir, 'label_encoders.pkl'))

print(f"\n✅ Success! New model and encoders have been saved to the '{output_dir}' folder.")

