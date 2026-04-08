from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

DATA_PATH = "data/Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

df = pd.read_csv(DATA_PATH)

# Basic validation (helps learning/debugging)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

X = df[FEATURES].astype(float)
y_str = df[TARGET].astype(str)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)  # int class ids

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42
        )),
    ]
)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# class order used by predict_proba
class_names = list(label_encoder.classes_)  # index aligns with proba columns


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}

    # Read inputs in the same order as FEATURES
    try:
        row = [float(data[f]) for f in FEATURES]
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e.args[0]}"}), 400
    except ValueError:
        return jsonify({"error": "All inputs must be numeric."}), 400

    X_new = np.array([row], dtype=float)
    proba = model.predict_proba(X_new)[0]  # shape: (n_classes,)

    top_3_idx = np.argsort(proba)[-3:][::-1]
    top_3_crops = [class_names[i] for i in top_3_idx]
    top_3_probabilities = [float(proba[i]) for i in top_3_idx]

    return jsonify({"top_3_crops": top_3_crops, "probabilities": top_3_probabilities})


if __name__ == "__main__":
    app.run(debug=True)