# WellWise-M 🩺

**AI-powered life expectancy prediction, tailored to Indian health and lifestyle data.**

WellWise-M predicts a person's estimated life expectancy from health, lifestyle, and demographic inputs — age, BMI, blood pressure, smoking/alcohol habits, diet quality, sleep, stress, air quality exposure, and more — using a model trained on synthetic data built to reflect real state-by-state health patterns across India (life expectancy, air quality, and demographic baselines by state).

---

## Features
- **Custom synthetic dataset generator** — models realistic health/lifestyle profiles using state-specific baselines (life expectancy, AQI, median age) for all 28 Indian states, with logically-derived fields (e.g. Diet Quality computed from protein/junk food/sugar intake, rather than random).
- **LightGBM regression model** trained on 25,000+ synthetic records to predict life expectancy.
- **Flask JSON API** (`app1.py`) — send health data, get a prediction back, built for programmatic/frontend integration (CORS-enabled for use with React, ngrok, etc.).
- **Web form frontend** (`templates/`) for entering health data directly in the browser, including live Air Quality Index lookup via the OpenWeatherMap API based on the user's geolocation.
- **Separate environments for data science vs. the web app** — dedicated requirements files (`requirements_ds.txt`, `requirements_app.txt`) so model training and app serving don't share unnecessary dependencies.

---

## How It Works

```
scripts/generate_data.py
      │  generates synthetic health records using state-specific baselines
      ▼
data/wellwise_health_data_v5_final.csv
      │
      ▼
scripts/train_model.py
      │  cleans data, encodes categorical features, trains a LightGBM regressor
      ▼
models/life_expectancy_model.pkl + models/label_encoders.pkl
      │
      ▼
app1.py (Flask API)
      │  loads model + encoders, exposes POST /predict
      ▼
test_api.py  →  sends a sample health profile, prints the predicted life expectancy
```

---

## 📁 Project Structure

```
.
├── app1.py                    # Flask API — loads model, serves /predict
├── test_api.py                 # Script to test the API with sample data
├── instructions.md              # How to run and test the API locally
├── scripts/
│   ├── generate_data.py          # Synthetic dataset generator
│   └── train_model.py             # Trains the LightGBM model
├── data/
│   ├── wellwise_health_data_v5_final.csv    # Final training dataset (25k records)
│   └── synthetic_india_health_20000_quantified_le.csv  # Earlier dataset version
├── models/
│   ├── life_expectancy_model.pkl
│   └── label_encoders.pkl
├── templates/
│   ├── index.html               # Health data input form
│   └── result.html               # Prediction result display
├── static/css/style.css
├── requirements_app.txt          # Dependencies for running the Flask app
└── requirements_ds.txt            # Dependencies for data generation/training
```

---

## ⚙️ Setup & Usage

### 1. Train the model (optional — a trained model is already included in `models/`)
```bash
python -m venv venv-data-science
source venv-data-science/bin/activate      # Windows: venv-data-science\Scripts\activate
pip install -r requirements_ds.txt

python scripts/generate_data.py     # generates the dataset
python scripts/train_model.py        # trains and saves the model
```

### 2. Run the API
```bash
python -m venv venv-webapp
source venv-webapp/bin/activate      # Windows: venv-webapp\Scripts\activate
pip install -r requirements_app.txt

python app1.py
```
The API will be available at `http://127.0.0.1:5000`.

### 3. Test a prediction
With the Flask server running, in a second terminal:
```bash
pip install requests
python test_api.py
```
Edit the `user_data` dictionary in `test_api.py` to test different health profiles.

**Example request:**
```json
POST /predict
{
  "Age": 45,
  "Gender": "Male",
  "BMI": 29.4,
  "Blood Pressure": "140/90",
  "Smoking": "Daily",
  "Exercise Type": "Walking",
  "State": "Delhi",
  "City": "Delhi"
  ...
}
```
**Example response:**
```json
{
  "predicted_life_expectancy": 68.4,
  "status": "success"
}
```

---

##  Tech Stack
`Python` · `Flask` · `Flask-CORS` · `LightGBM` · `scikit-learn` · `Pandas` / `NumPy` · `joblib` · `HTML/CSS`

---
## Project Info
This is a part of a hackathon project which is a web app. 
