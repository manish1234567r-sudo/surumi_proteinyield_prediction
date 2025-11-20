# app.py
import os
from flask import Flask, request, render_template_string
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

app = Flask(__name__)

MODEL_FILE = "protein_model.pkl"
SCALER_FILE = "scaler.pkl"

# -------------------------------
# HIGHER ACCURACY TRAINING DATA
# -------------------------------
DATA = np.array([
    [4.0, 55, 20, 0.320, 0.43],
    [6.5, 55, 20, 0.320, 0.36],
    [4.0, 65, 20, 0.320, 1.00],
    [6.5, 65, 20, 0.320, 1.15],
    [4.0, 60, 15, 0.320, 1.17],
    [6.5, 60, 15, 0.320, 0.86],
    [4.0, 60, 25, 0.320, 1.06],
    [6.5, 60, 25, 0.320, 0.99],
    [5.5, 55, 15, 0.320, 0.95],
    [5.5, 65, 15, 0.320, 0.98],
    [5.5, 55, 25, 0.320, 1.30],
    [5.5, 65, 25, 0.320, 1.04],
])

BASE_VOLUME = 0.320

# -------------------------------
# TRAIN MODEL (HIGH ACCURACY)
# -------------------------------
def train_model():
    X = DATA[:, :-1]
    y = DATA[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        max_depth=12
    )
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_FILE)
    print("High-accuracy model trained and saved!")


def load_model_safe():
    if not Path(MODEL_FILE).exists() or not Path(SCALER_FILE).exists():
        train_model()

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler


MODEL, SCALER = load_model_safe()


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_protein(pH, temp, time_min, requested_volume):
    X_input = np.array([[pH, temp, time_min, BASE_VOLUME]])
    scaled = SCALER.transform(X_input)
    base_yield = float(MODEL.predict(scaled)[0])
    return base_yield * (requested_volume / BASE_VOLUME)


# -------------------------------
# PREMIUM GLASSMORPHISM UI
# -------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Surimi Protein Yield Predictor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
body {
    margin: 0;
    height: 100vh;
    background: linear-gradient(135deg, #0a0f24, #001531, #032436);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Poppins', sans-serif;
    color: white;
}

/* Glass Panel */
.card {
    width: 90%;
    max-width: 800px;
    padding: 35px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: fadeIn 1.2s ease-in-out;
}

/* Neon Header */
h2 {
    text-align: center;
    margin-bottom: 25px;
    font-size: 28px;
    text-shadow: 0 0 12px #00eaff;
}

/* Inputs */
input, select {
    width: 100%;
    padding: 14px;
    border-radius: 12px;
    border: none;
    margin-bottom: 16px;
    background: rgba(255,255,255,0.15);
    color: white;
    font-size: 16px;
    outline: none;
}

/* Button */
button {
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #00eaff, #0072ff);
    border: none;
    border-radius: 12px;
    color: #000;
    font-weight: 700;
    cursor: pointer;
    transition: 0.3s;
}

button:hover {
    transform: scale(1.03);
}

/* Result */
.result {
    margin-top: 20px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    color: #00ffcf;
    text-shadow: 0 0 10px #00ffc8;
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
</head>

<body>
<div class="card">

  <h2>Surimi Protein Yield Predictor</h2>

  <form method="POST">
    <label>Fish Type</label>
    <select name="fish">
      {% for f in ['Threadfin Bream','Lizardfish','Armor Croaker','Ribbonfish','Silver Carp','Tilapia','Catfish'] %}
        <option {{ 'selected' if fish==f else '' }}>{{ f }}</option>
      {% endfor %}
    </select>

    <label>pH</label>
    <input type="number" step="0.1" name="ph" required value="{{ ph or '' }}">

    <label>Temperature (°C)</label>
    <input type="number" name="temp" required value="{{ temp or '' }}">

    <label>Time (min)</label>
    <input type="number" name="time" required value="{{ time or '' }}">

    <label>Volume (L)</label>
    <input type="number" step="0.001" name="volume" required value="{{ volume or '' }}">

    <button type="submit">Predict Yield</button>
  </form>

  {% if result is not none %}
  <div class="result">
      Predicted Yield: {{ result }} g
  </div>
  {% endif %}

</div>
</body>
</html>
"""

# -------------------------------
# FLASK ROUTE
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    ph = temp = time_val = volume = fish = None

    if request.method == "POST":
        fish = request.form.get("fish")
        ph = float(request.form.get("ph"))
        temp = float(request.form.get("temp"))
        time_val = float(request.form.get("time"))
        volume = float(request.form.get("volume"))

        result = round(predict_protein(ph, temp, time_val, volume), 4)

    return render_template_string(
        TEMPLATE,
        result=result,
        ph=ph,
        temp=temp,
        time=time_val,
        volume=volume,
        fish=fish
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
