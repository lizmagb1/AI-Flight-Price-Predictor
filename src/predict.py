import joblib
import pandas as pd

# Load model + column structure
model = joblib.load("model.pkl")
columns = joblib.load("model_columns.pkl")

# Sample input
input_data = pd.DataFrame([{
    "duration": 2.5,
    "days_left": 10
}])

# One-hot encode + align columns
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=columns, fill_value=0)

# Predict
prediction = model.predict(input_data)

print("Predicted price:", prediction[0])