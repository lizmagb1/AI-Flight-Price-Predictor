import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data/clean_database.csv")

print("Dataset preview:")
print(df.head())

# Define target
y = df["price"]
X = df.drop("price", axis=1)

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Save column structure
feature_columns = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model + columns
joblib.dump(model, "model.pkl")
joblib.dump(feature_columns, "model_columns.pkl")

print("Model training complete")

