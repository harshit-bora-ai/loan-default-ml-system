from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Create dataset directly (no CSV needed)
data = {
    "age": [25,45,35,50,28,40,30,55,32,48],
    "income": [50000,80000,60000,90000,52000,75000,58000,95000,62000,85000],
    "loan_amount": [20000,30000,25000,40000,22000,35000,24000,45000,26000,38000],
    "credit_score": [650,700,680,720,660,710,670,730,675,705],
    "default": [0,1,0,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)

# Train model
X = df.drop("default", axis=1)
y = df["default"]

model = RandomForestClassifier()
model.fit(X, y)

@app.route("/")
def home():
    return "✅ Loan Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
