# AI-Flight-Price-Predictor

# ✈️ Flight Price Predictor (Machine Learning Project)

## Overview

This project is an end-to-end machine learning system that predicts flight ticket prices based on airline, route, departure time, arrival time, duration, class, and number of days left before departure.

It demonstrates a full ML pipeline including data preprocessing, feature engineering, model training, evaluation, and real-time prediction

---

## 🚀 Features

- Predicts flight ticket prices based on user input
- Handles categorical and numerical feature engineering
- Trained on 300,000+ flight records
- High-performance regression model (Random Forest)
- CLI-based interactive prediction system
- Saved trained model for fast inference (`model.pkl`)
- Clean modular structure (train vs inference separated)

---

## 📊 Dataset

The dataset contains ~300,000 flight records with the following features:

- Airline (e.g., Vistara, AirAsia, SpiceJet)
- Source city
- Destination city
- Departure time
- Arrival time
- Flight duration
- Travel class (Economy / Business)
- Days left until departure
- Price (target variable)

---

## 🧠 Machine Learning Approach

### Data Preprocessing
- One-hot encoding for categorical variables
- Feature alignment between training and prediction
- Removal of unnecessary columns (e.g., flight ID)

### Model
- Random Forest Regressor
- Handles non-linear relationships effectively
- Robust to mixed data types

### Evaluation Metrics
- Mean Absolute Error (MAE)
- R² Score

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| MAE    | ~868  |
| R²     | ~0.988 |

The model explains ~98% of the variance in flight prices, indicating strong predictive performance.

---

## 📁 Project Structure

```text
AI-Flight-Price-Predictor/
│
├── data/
│   └── clean_database.csv
│
├── models/
│   ├── model.pkl
│   └── model_columns.pkl
│
├── src/
│   ├── train.py      # Model training script
│   ├── app.py        # CLI prediction system
│
├── requirements.txt
├── README.md
└── .gitignore