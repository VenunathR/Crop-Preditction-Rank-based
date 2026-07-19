"""
app.py
Serves crop recommendations using a pre-trained XGBoost + Keras ensemble.
Models are trained once via train_models.py and loaded here (no retraining
on app startup).
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import shap
from tensorflow import keras

app = Flask(__name__)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ------------------------------------------------------------------
# Load everything ONCE at startup, not per-request
# ------------------------------------------------------------------
xgb_model = joblib.load("models/xgb_model.joblib")
keras_model = keras.models.load_model("models/keras_model.keras")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# SHAP explainer built once at startup (fast for tree models)
explainer = shap.TreeExplainer(xgb_model)


def predict_ensemble(features_dict):
    """Run both models, average their probabilities, return top-3 + explanation."""
    x = np.array([[features_dict[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)

    xgb_probs = xgb_model.predict_proba(x)[0]
    keras_probs = keras_model.predict(x_scaled, verbose=0)[0]
    ensemble_probs = (xgb_probs + keras_probs) / 2.0

    top3_idx = np.argsort(ensemble_probs)[::-1][:3]
    top3 = [
        {"crop": label_encoder.classes_[i], "probability": round(float(ensemble_probs[i]), 4)}
        for i in top3_idx
    ]

    # SHAP explanation for the top predicted class, from the XGBoost component
    top_class_idx = top3_idx[0]
    shap_exp = explainer(x)
    contributions = shap_exp.values[0, :, top_class_idx]
    explanation = sorted(
        [{"feature": f, "impact": round(float(c), 4)} for f, c in zip(FEATURES, contributions)],
        key=lambda d: abs(d["impact"]),
        reverse=True,
    )

    return {
        "top_prediction": top3[0]["crop"],
        "top3": top3,
        "explanation": explanation,
        "model_confidences": {
            "xgboost": round(float(np.max(xgb_probs)), 4),
            "keras_nn": round(float(np.max(keras_probs)), 4),
            "ensemble": round(float(np.max(ensemble_probs)), 4),
        },
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        features_dict = {f: float(data[f]) for f in FEATURES}
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": f"Provide numeric values for: {FEATURES}"}), 400

    result = predict_ensemble(features_dict)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
