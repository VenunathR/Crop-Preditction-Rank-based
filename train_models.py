"""
train_models.py
Trains an ensemble crop recommendation system (XGBoost + Keras NN),
evaluates calibration, generates SHAP explanations, and saves everything
needed for app.py to serve predictions without retraining.

Run: python train_models.py
Outputs:
  models/xgb_model.joblib
  models/keras_model.keras
  models/scaler.joblib
  models/label_encoder.joblib
  results/*.png  (EDA + calibration + SHAP plots)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import shap

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

RANDOM_STATE = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# ============================================================
# STAGE 1: LOAD + EDA
# ============================================================
print("=" * 60)
print("STAGE 1: Load data & EDA")
print("=" * 60)

df = pd.read_csv("data/Crop_recommendation.csv")
print("Shape:", df.shape)
print(df.info())

# Class balance
class_counts = df[TARGET].value_counts()
print("\nNumber of classes:", df[TARGET].nunique())
print(class_counts)

plt.figure(figsize=(10, 6))
class_counts.plot(kind="bar")
plt.title("Sample count per crop")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("results/01_class_balance.png", dpi=150)
plt.close()

# Feature distributions
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for i, feat in enumerate(FEATURES):
    ax = axes[i // 3, i % 3]
    sns.histplot(df[feat], kde=True, ax=ax)
    ax.set_title(feat)
for j in range(len(FEATURES), 9):
    fig.delaxes(axes[j // 3, j % 3])
plt.tight_layout()
plt.savefig("results/02_feature_distributions.png", dpi=150)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig("results/03_correlation_heatmap.png", dpi=150)
plt.close()

# Mean feature values per crop (reference for later SHAP sanity-check)
crop_means = df.groupby(TARGET)[FEATURES].mean()
crop_means.to_csv("results/04_crop_feature_means.csv")
print("\nSaved per-crop feature means to results/04_crop_feature_means.csv")

# ============================================================
# STAGE 2: PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2: Preprocessing")
print("=" * 60)

X = df[FEATURES].values
y_raw = df[TARGET].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
n_classes = len(label_encoder.classes_)
print("Classes:", list(label_encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STAGE 3: XGBOOST
# ============================================================
print("\n" + "=" * 60)
print("STAGE 3: XGBoost")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
)
# XGBoost doesn't need scaling (tree splits are scale-invariant) — use raw X
xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)
xgb_preds = np.argmax(xgb_probs, axis=1)
print("XGBoost accuracy:", accuracy_score(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds, target_names=label_encoder.classes_, zero_division=0))

# ============================================================
# STAGE 4: KERAS NEURAL NETWORK
# ============================================================
print("\n" + "=" * 60)
print("STAGE 4: Keras Neural Network")
print("=" * 60)

keras_model = keras.Sequential([
    layers.Input(shape=(len(FEATURES),)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(n_classes, activation="softmax"),
])
keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = keras_model.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=60,
    batch_size=16,
    verbose=0,
)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/05_keras_training_curves.png", dpi=150)
plt.close()

keras_probs = keras_model.predict(X_test_scaled, verbose=0)
keras_preds = np.argmax(keras_probs, axis=1)
print("Keras NN accuracy:", accuracy_score(y_test, keras_preds))
print(classification_report(y_test, keras_preds, target_names=label_encoder.classes_, zero_division=0))

# ============================================================
# STAGE 5: ENSEMBLE (soft voting)
# ============================================================
print("\n" + "=" * 60)
print("STAGE 5: Ensemble")
print("=" * 60)

ensemble_probs = (xgb_probs + keras_probs) / 2.0
ensemble_preds = np.argmax(ensemble_probs, axis=1)
print("Ensemble accuracy:", accuracy_score(y_test, ensemble_preds))
print(classification_report(y_test, ensemble_preds, target_names=label_encoder.classes_, zero_division=0))

cm = confusion_matrix(y_test, ensemble_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.title("Ensemble confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/06_ensemble_confusion_matrix.png", dpi=150)
plt.close()

# ============================================================
# STAGE 6: CALIBRATION
# ============================================================
print("\n" + "=" * 60)
print("STAGE 6: Calibration")
print("=" * 60)

confidences = np.max(ensemble_probs, axis=1)
correct = (ensemble_preds == y_test).astype(int)

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_accs, bin_confs, bin_counts = [], [], []
ece = 0.0
for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (confidences > lo) & (confidences <= hi)
    count = mask.sum()
    if count > 0:
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (count / len(confidences)) * abs(acc - conf)
    else:
        acc, conf = 0, 0
    bin_accs.append(acc)
    bin_confs.append(conf)
    bin_counts.append(count)

print(f"Expected Calibration Error (ECE): {ece:.4f}")

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
plt.bar(bin_edges[:-1], bin_accs, width=0.1, align="edge", alpha=0.7, edgecolor="black", label="observed accuracy")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("results/07_reliability_diagram.png", dpi=150)
plt.close()

# ============================================================
# STAGE 7: SHAP (on XGBoost component)
# ============================================================
print("\n" + "=" * 60)
print("STAGE 7: SHAP explainability")
print("=" * 60)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(
    shap_values.values, X_test, feature_names=FEATURES,
    class_names=label_encoder.classes_, plot_type="bar", show=False
)
plt.title("Global feature importance (SHAP)")
plt.tight_layout()
plt.savefig("results/08_shap_global_importance.png", dpi=150)
plt.close()

# Local explanation for one example
sample_idx = 0
pred_class = xgb_preds[sample_idx]
waterfall_exp = shap.Explanation(
    values=shap_values.values[sample_idx, :, pred_class],
    base_values=shap_values.base_values[sample_idx, pred_class],
    data=X_test[sample_idx],
    feature_names=FEATURES,
)
plt.figure()
shap.plots.waterfall(waterfall_exp, show=False)
plt.title(f"Why sample {sample_idx} -> {label_encoder.inverse_transform([pred_class])[0]}")
plt.tight_layout()
plt.savefig("results/09_shap_local_example.png", dpi=150)
plt.close()

# ============================================================
# SAVE EVERYTHING FOR app.py
# ============================================================
print("\n" + "=" * 60)
print("Saving models")
print("=" * 60)

joblib.dump(xgb_model, "models/xgb_model.joblib")
keras_model.save("models/keras_model.keras")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(label_encoder, "models/label_encoder.joblib")

summary = {
    "xgb_accuracy": float(accuracy_score(y_test, xgb_preds)),
    "keras_accuracy": float(accuracy_score(y_test, keras_preds)),
    "ensemble_accuracy": float(accuracy_score(y_test, ensemble_preds)),
    "ece": float(ece),
    "n_classes": n_classes,
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done. Summary:", summary)
