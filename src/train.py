import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# 1. Load dataset
df = pd.read_csv("data/clean_database.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Drop unnecessary column if exists
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Target variable
y = df["price"]

# Features
X = df.drop("price", axis=1)

# Handle categorical variables
categorical_cols = X.select_dtypes(include=["object", "string"]).columns

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Save feature columns for prediction time
model_columns = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = RandomForestRegressor(
    n_estimators=50,  
    random_state=42,
    n_jobs=-1          
)

model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nModel Performance:")
print("MAE:", mae)
print("R2 Score:", r2)

# Save model + columns
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

print("\nModel saved successfully!")