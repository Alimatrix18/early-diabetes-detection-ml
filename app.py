from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# ------------------------
# Load models
# ------------------------
ensemble_path = "models/final_ensemble.pkl"  # <-- Your model path
scaler_path = "models/unified_scaler.pkl"    # <-- Your scaler path

ensemble = joblib.load(ensemble_path)
scaler = joblib.load(scaler_path)

# ------------------------
# Features
# ------------------------
final_features = [
    "gender", "age", "bmi", "hypertension", "cholesterol", "physical_activity",
    "heart_disease", "smoker", "blood_glucose_level", "insulin", "skin_thickness",
    "family_history", "pregnancies", "polyuria", "polydipsia", "polyphagia"
]

numeric_features = ["age", "bmi", "blood_glucose_level", "insulin", "skin_thickness"]

def get_yes_no(val):
    return 1 if str(val).lower() in ["yes", "y"] else 0

def predict_risk(data):
    df = pd.DataFrame([data])
    df = df.reindex(columns=final_features, fill_value=0)
    df[numeric_features] = scaler.transform(df[numeric_features])
    prob = ensemble.predict_proba(df)[0][1]
    risk_percent = round(prob * 100, 2)

    if prob < 0.20:
        return risk_percent, "LOW RISK", "No immediate risk detected. Maintain a healthy lifestyle.", "#2e7d32"
    elif prob < 0.65:
        return risk_percent, "MEDIUM RISK", "Moderate risk detected. Consider lifestyle improvements and monitor health.", "#f9a825"
    else:
        return risk_percent, "HIGH RISK", "High risk of diabetes. Please consult a healthcare professional soon.", "#c62828"

# ------------------------
# Flask app
# ------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Convert yes/no to 0/1
        for key in ["hypertension", "cholesterol", "physical_activity", "heart_disease", "smoker",
                    "family_history", "polyuria", "polydipsia", "polyphagia"]:
            if key in data:
                data[key] = get_yes_no(data[key])

        # Gender numeric
        if "gender" in data:
            data["gender"] = 1 if str(data["gender"]).lower() in ["male", "m"] else 0

        # Pregnancies
        if data.get("gender") == 0 and "pregnancies" in data:
            data["pregnancies"] = int(data["pregnancies"])
        else:
            data["pregnancies"] = 0

        # Convert numeric fields
        for key in ["age", "bmi", "blood_glucose_level", "insulin", "skin_thickness"]:
            if key in data:
                data[key] = float(data[key])

        risk_percent, risk_level, recommendation, color = predict_risk(data)
        return jsonify({
            "risk_percent": risk_percent,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "color": color
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)