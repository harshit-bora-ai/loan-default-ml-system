import os
import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

scaler = StandardScaler()
model = LogisticRegression(random_state=42)


def generate_training_data(n_samples=1000):
    np.random.seed(42)
    age = np.random.randint(20, 70, n_samples)
    income = np.random.randint(20000, 150000, n_samples)
    loan_amount = np.random.randint(5000, 200000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)

    default = (
        (credit_score < 580).astype(int) * 3
        + (loan_amount / (income + 1) > 3).astype(int) * 2
        + (age < 25).astype(int)
        + np.random.randint(0, 2, n_samples)
    ) >= 4

    X = np.column_stack([age, income, loan_amount, credit_score])
    y = default.astype(int)
    return X, y


X_train, y_train = generate_training_data()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
model.fit(X_train_scaled, y_train)


def validate_input(data):
    required_fields = ["age", "income", "loan_amount", "credit_score"]
    errors = []

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return None, errors

    try:
        age = float(data["age"])
        income = float(data["income"])
        loan_amount = float(data["loan_amount"])
        credit_score = float(data["credit_score"])
    except (ValueError, TypeError) as e:
        return None, [f"All fields must be numeric: {str(e)}"]

    validation_errors = []
    if not (18 <= age <= 120):
        validation_errors.append("'age' must be between 18 and 120")
    if income < 0:
        validation_errors.append("'income' must be non-negative")
    if loan_amount <= 0:
        validation_errors.append("'loan_amount' must be greater than 0")
    if not (300 <= credit_score <= 850):
        validation_errors.append("'credit_score' must be between 300 and 850")

    if validation_errors:
        return None, validation_errors

    return {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
    }, None


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()

    validated, errors = validate_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    features = np.array(
        [
            [
                validated["age"],
                validated["income"],
                validated["loan_amount"],
                validated["credit_score"],
            ]
        ]
    )
    features_scaled = scaler.transform(features)

    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    default_probability = float(probabilities[1])
    no_default_probability = float(probabilities[0])

    risk_level = (
        "low"
        if default_probability < 0.3
        else "medium"
        if default_probability < 0.6
        else "high"
    )

    return jsonify(
        {
            "prediction": prediction,
            "label": "default" if prediction == 1 else "no_default",
            "default_probability": round(default_probability, 4),
            "no_default_probability": round(no_default_probability, 4),
            "risk_level": risk_level,
            "input": validated,
        }
    )


@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "logistic_regression"})


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "Loan Default Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "POST /predict": "Predict loan default given age, income, loan_amount, credit_score",
                "GET /healthz": "Health check",
            },
            "example_request": {
                "age": 35,
                "income": 55000,
                "loan_amount": 15000,
                "credit_score": 680,
            },
        }
    )


if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0", port=port, debug=False)
