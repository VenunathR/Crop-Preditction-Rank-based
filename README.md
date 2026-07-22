# Crop Prediction (Rank Based)

This is a small Flask app I built to recommend which crop to grow based on soil and weather conditions. Instead of just spitting out one answer, it gives you the top 3 crops ranked by probability, and also tells you *why* it picked the top one using SHAP.

Under the hood it's not just one model — I'm combining an XGBoost classifier and a Keras neural net, and averaging their predictions together. In my testing this ensemble approach gave more stable results than either model alone.

## What it actually does

You give it 7 numbers:
- N, P, K (nitrogen, phosphorous, potassium levels in the soil)
- temperature
- humidity
- ph
- rainfall

And it:
1. Scales/preps the input
2. Runs it through both the XGBoost model and the Keras model
3. Averages the two sets of probabilities
4. Picks the top 3 crops from that
5. Runs SHAP on the XGBoost side to show which features pushed the top prediction

The models are trained ahead of time and just loaded when the app starts — it's not retraining anything on every request (that would be way too slow).

## Files in here

```
app.py                     -> the Flask app itself
Crop_recommendation.csv     -> the dataset used for training
data/                        -> extra data stuff
templates/                   -> the HTML page for the UI
requirements.txt             -> deps
Procfile                     -> for deploying with gunicorn
runtime.txt                  -> python version pin
```

Heads up — `app.py` expects a `models/` folder with these files in it:
- `xgb_model.joblib`
- `keras_model.keras`
- `scaler.joblib`
- `label_encoder.joblib`

These aren't retrained live, so you need to have them saved there already (from whatever training script generated them) before running the app.

## Getting it running

```bash
git clone https://github.com/VenunathR/Crop-Preditction-Rank-based.git
cd Crop-Preditction-Rank-based

python -m venv venv
source venv/bin/activate      # windows: venv\Scripts\activate

pip install -r requirements.txt
```

Then just:

```bash
python app.py
```

and go to `http://127.0.0.1:5000/` in your browser.

## Using the API directly

If you don't want to use the web page and just want to hit the endpoint:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202.9}'
```

and you'll get back something like:

```json
{
  "top_prediction": "rice",
  "top3": [
    {"crop": "rice", "probability": 0.95},
    {"crop": "jute", "probability": 0.03},
    {"crop": "maize", "probability": 0.01}
  ],
  "explanation": [
    {"feature": "rainfall", "impact": 0.42},
    {"feature": "humidity", "impact": 0.31},
    {"feature": "N", "impact": -0.05}
  ],
  "model_confidences": {
    "xgboost": 0.97,
    "keras_nn": 0.93,
    "ensemble": 0.95
  }
}
```

`top_prediction` is the winner, `top3` gives you a broader picture in case you want to consider alternatives, and `explanation` tells you which features mattered most for that top pick (positive impact = pushed toward that crop, negative = pushed away from it).

## Stack

- Flask for the backend/API
- XGBoost + TensorFlow/Keras for the two models in the ensemble
- SHAP for explaining predictions
- scikit-learn for scaling and label encoding
- gunicorn for running it in production (see the Procfile)

## Deploying

There's a Procfile already set up (`web: gunicorn app:app`), so this should deploy fine on Heroku or anything similar that reads Procfiles. Just make sure the `models/` folder actually gets included in the deploy — it's easy to forget since it's not committed by default in a lot of setups.

## TODO / things to add later

- No license yet, should probably add one
- Could use a requirements pin for exact versions instead of unpinned (only Flask is pinned right now)
- Might be worth adding the training script to the repo so people can regenerate the models themselves instead of needing them handed over
