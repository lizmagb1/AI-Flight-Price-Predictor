import pandas as pd
import pickle

# Load model + columns
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

df = pd.read_csv("data/clean_database.csv")

# Helper function for menus
def select_option(options, title):
    print(f"\nSelect {title}:")
    for i, option in enumerate(options):
        print(f"{i + 1}. {option}")

    while True:
        try:
            choice = int(input("Enter number: ")) - 1
            if 0 <= choice < len(options):
                return options[choice]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")

# Build option lists from data
airlines = sorted(df["airline"].unique())
source_cities = sorted(df["source_city"].unique())
destination_cities = sorted(df["destination_city"].unique())
departure_times = sorted(df["departure_time"].unique())
arrival_times = sorted(df["arrival_time"].unique())
classes = sorted(df["class"].unique())

# User interface INPUT
print("=== Flight Price Predictor ===")

airline = select_option(airlines, "Airline")
source_city = select_option(source_cities, "Source City")
destination_city = select_option(destination_cities, "Destination City")
departure_time = select_option(departure_times, "Departure Time")
arrival_time = select_option(arrival_times, "Arrival Time")
flight_class = select_option(classes, "Class")

duration = float(input("\nEnter duration (hours): "))
days_left = int(input("Enter days left until departure: "))

# Create input dataframe
input_dict = {
    "airline": airline,
    "source_city": source_city,
    "destination_city": destination_city,
    "departure_time": departure_time,
    "arrival_time": arrival_time,
    "class": flight_class,
    "duration": duration,
    "days_left": days_left
}

input_df = pd.DataFrame([input_dict])

# One-hot encode same as training
input_df = pd.get_dummies(input_df)

# Align with training columns
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Prediction
prediction = model.predict(input_df)[0]

print("\n======================")
print(f"Estimated Flight Price: ${prediction:,.2f}")
print("======================")