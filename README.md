# Loan Default Prediction System
This API predicts whether a loan applicant is likely to default based on financial features.
An end-to-end Machine Learning project that predicts loan default risk using a Logistic Regression model and exposes predictions via a Flask API.

## Features

* Logistic Regression model
* Feature scaling (StandardScaler)
* Input validation
* REST API with Flask
* Risk classification (low / medium / high)

## Input Parameters

* age
* income
* loan_amount
* credit_score

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

## API Endpoint

POST `/predict`

### Example Request

```json
{
  "age": 35,
  "income": 55000,
  "loan_amount": 15000,
  "credit_score": 680
}
```

### Example Response

```json
{
  "prediction": 0,
  "label": "no_default",
  "risk_level": "low"
}
```

## Tech Stack

* Python
* Scikit-learn
* Flask
* NumPy

## Future Improvements

* Deploy on cloud (Render / AWS)
* Add frontend UI
* Use real-world dataset
