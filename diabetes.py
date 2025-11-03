from flask import Flask, request, jsonify
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Train a simple regression model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

@app.route("/", methods=["GET"])
def home():
    return "Diabetes regression model running."

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    features = np.array(payload["features"]).reshape(1, -1)
    pred = model.predict(features)[0]
    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
