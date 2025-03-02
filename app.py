import pandas as pd
import numpy as np
import streamlit as st
import joblib
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
def load_data():
    df = pd.read_csv("Flight_Price.csv")
    return df

# Data Preprocessing
def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time']).dt.hour
    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time']).dt.hour
    df['Duration'] = df['Duration'].str.replace('h', '*60').str.replace(' ', '+').str.replace('m', '*1')
    df['Duration'] = df['Duration'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    label_cols = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']
    label_encoders = {col: LabelEncoder() for col in label_cols}
    for col in label_cols:
        df[col] = label_encoders[col].fit_transform(df[col])
    
    df['Total_Stops'] = df['Total_Stops'].replace({
        'non-stop': 0,
        '1 stop': 1,
        '2 stops': 2,
        '3 stops': 3,
        '4 stops': 4
    })
    
    df.drop(columns=['Date_of_Journey'], inplace=True)
    
    X = df.drop(columns=['Price'])
    y = df['Price']
    return X, y, label_encoders

# Train Models
def train_models(X, y, label_encoders):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    mlflow.set_experiment("Flight_Price_Prediction")

    for name, model in models.items():
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
            mlflow.log_metric("MSE", mean_squared_error(y_test, y_pred))
            mlflow.log_metric("R2", r2_score(y_test, y_pred))
            mlflow.sklearn.log_model(model, name)

            results[name] = {"R2": r2_score(y_test, y_pred), "model": model}

        mlflow.end_run()

    best_model = max(results, key=lambda x: results[x]['R2'])
    joblib.dump(results[best_model]['model'], "best_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

    return results[best_model]['model']

# Load Model & Encoders Efficiently
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

@st.cache_resource
def load_label_encoders():
    return joblib.load("label_encoders.pkl")

# **✅ Fix: Initialize session state correctly**
if "predict_clicked" not in st.session_state:
    st.session_state["predict_clicked"] = False  # Properly initialized

def set_predict_false():
    st.session_state["predict_clicked"] = False  # Reset when filters change

# Streamlit App UI
def run_app():
    st.title("✈️ Flight Price Prediction")

    model = load_model()
    label_encoders = load_label_encoders()

    # Sidebar Input Widgets (with callbacks to prevent reload)
    st.sidebar.header("Select Flight Details")
    airline = st.sidebar.selectbox("Airline", label_encoders['Airline'].classes_, key="airline", on_change=set_predict_false)
    source = st.sidebar.selectbox("Source", label_encoders['Source'].classes_, key="source", on_change=set_predict_false)
    destination = st.sidebar.selectbox("Destination", label_encoders['Destination'].classes_, key="destination", on_change=set_predict_false)
    route = st.sidebar.selectbox("Route", label_encoders['Route'].classes_, key="route", on_change=set_predict_false)
    additional_info = st.sidebar.selectbox("Additional Info", label_encoders['Additional_Info'].classes_, key="additional_info", on_change=set_predict_false)

    dep_time = st.sidebar.slider("Departure Hour", 0, 23, 10, key="dep_time", on_change=set_predict_false)
    arrival_time = st.sidebar.slider("Arrival Hour", 0, 23, 18, key="arrival_time", on_change=set_predict_false)
    duration = st.sidebar.number_input("Flight Duration (minutes)", min_value=0, value=120, key="duration", on_change=set_predict_false)
    stops = st.sidebar.number_input("Total Stops", min_value=0, value=1, key="stops", on_change=set_predict_false)

    # Button to Trigger Prediction
    if st.sidebar.button("Predict Price"):
        st.session_state["predict_clicked"] = True  # ✅ Properly initialized

    # **✅ Check if session state key exists before using it**
    if "predict_clicked" in st.session_state and st.session_state["predict_clicked"]:
        input_data = np.array([
            label_encoders['Airline'].transform([airline])[0],
            label_encoders['Source'].transform([source])[0],
            label_encoders['Destination'].transform([destination])[0],
            label_encoders['Route'].transform([route])[0],
            dep_time, arrival_time, duration, stops,
            label_encoders['Additional_Info'].transform([additional_info])[0]
        ]).reshape(1, -1)

        price = model.predict(input_data)[0]
        st.success(f"💰 Predicted Flight Price: ₹{price:.2f}")

if __name__ == "__main__":
    df = load_data()
    X, y, label_encoders = preprocess_data(df)
    train_models(X, y, label_encoders)
    run_app()
