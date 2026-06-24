from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "irrigation_model_best_rf.pkl") 
SCALER_PATH = os.path.join(BASE_DIR, "models", "irrigation_scaler.pkl")

# Load your Scikit-Learn assets safely
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No payload provided'}), 400

        # 1. Parse dynamic inputs matching your ESP32 payload keys
        soil_moisture = float(data.get('soil_moisture', 50.0))
        temp = float(data.get('air_temperature_c', 25.0))
        humidity = float(data.get('air_humidity_pct', 60.0))
        nitrogen = float(data.get('nitrogen_ppm', 45.0))
        phosphorus = float(data.get('phosphorus_ppm', 45.0))
        potassium = float(data.get('potassium_ppm', 45.0))

        # 2. Reconstruct your 11-feature array in the exact training order:
        # [soil, temp, humidity, wind_speed, wind_gust, pressure, ph, rainfall, N, P, K]
        raw_features = np.array([[
            soil_moisture,
            temp,
            humidity,
            2.07,   # wind_speed_kmh baseline median
            8.38,   # wind_gust_kmh baseline median
            101.5,  # pressure_kpa baseline median
            6.75,   # ph baseline median
            214.8,  # rainfall baseline median
            nitrogen,
            phosphorus,
            potassium
        ]], dtype=np.float32)

        # 3. Standardize your input data using your scaler matrix
        scaled_features = scaler.transform(raw_features)

        # 4. Generate prediction from the Random Forest
        # For Scikit-Learn: predict() outputs the final 0 or 1 directly
        valve_trigger = int(model.predict(scaled_features)[0])
        
        # Calculate prediction probability/confidence if supported
        try:
            probabilities = model.predict_proba(scaled_features)[0]
            confidence = float(probabilities[valve_trigger])
        except AttributeError:
            confidence = 1.0  # Fallback if probability estimation was disabled during training

        return jsonify({
            'valve_trigger': valve_trigger,
            'confidence': round(confidence, 4)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)