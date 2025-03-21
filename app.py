from flask import Flask, request, jsonify, render_template
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

np.random.seed(42)
n_samples = 500

temperature = np.random.uniform(15, 35, n_samples)
humidity = np.random.uniform(60, 90, n_samples)
ph = np.random.uniform(5.5, 8.5, n_samples)
rainfall = np.random.uniform(100, 300, n_samples)

crop_labels = ['rice', 'wheat', 'maize', 'barley', 'soybean', 'millet', 'sorghum', 'groundnut', 'cotton', 'sugarcane']
labels = np.random.choice(crop_labels, n_samples)

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall,
    'label': labels
})

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(df[['label']])

scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['label']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', max_iter=100, batch_size=8, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_sample = np.array([[data['temperature'], data['humidity'], data['ph'], data['rainfall']]])
    new_sample = scaler.transform(new_sample)
    predictions = model.predict_proba(new_sample)

    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_crops = [crop_labels[i] for i in top_3_indices]
    top_3_probabilities = [float(predictions[0][i]) for i in top_3_indices]

    return jsonify({'top_3_crops': top_3_crops, 'probabilities': top_3_probabilities})

if __name__ == '__main__':
    app.run(debug=True)