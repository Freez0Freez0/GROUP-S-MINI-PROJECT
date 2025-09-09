import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Prediction App")

# --- Load Dataset ---
st.subheader("Dataset Preview")
df = pd.read_csv("laptop_price.csv", encoding="ISO-8859-1")
st.write(df.head(10))

# --- Clean Data ---
df = df.dropna().drop_duplicates()

# Convert RAM and Weight to numeric
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# Drop columns that can't be directly converted or don't add much predictive value
for col in ['Product', 'ScreenResolution']:
    if col in df.columns:
        df = df.drop(columns=[col])

# One-hot encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- Train/Test Split ---
X = df.drop('Price_euros', axis=1)
y = df['Price_euros']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted")
ax.axline((0, 0), slope=1, color='r', linestyle="--")
st.pyplot(fig)

# --- Prediction Form ---
st.subheader("🔮 Predict Laptop Price")

ram = st.slider("RAM (GB)", 2, 64, 8)
weight = st.slider("Weight (kg)", 0.5, 5.0, 1.5)

# Select Company from available options in dataset
company_options = [col for col in X_train.columns if col.startswith("Company_")]
company_choice = st.selectbox("Company", company_options)

# Select CPU
cpu_options = [col for col in X_train.columns if col.startswith("Cpu_")]
cpu_choice = st.selectbox("CPU", cpu_options)

# Select GPU
gpu_options = [col for col in X_train.columns if col.startswith("Gpu_")]
gpu_choice = st.selectbox("GPU", gpu_options)

# Select Operating System
os_options = [col for col in X_train.columns if col.startswith("OpSys_")]
os_choice = st.selectbox("Operating System", os_options)

# Build input row
input_data = {col: 0 for col in X_train.columns}
input_data['Ram'] = ram
input_data['Weight'] = weight
if company_choice in input_data:
    input_data[company_choice] = 1
if cpu_choice in input_data:
    input_data[cpu_choice] = 1
if gpu_choice in input_data:
    input_data[gpu_choice] = 1
if os_choice in input_data:
    input_data[os_choice] = 1

df_input = pd.DataFrame([input_data])

if st.button("Predict"):
    predicted_price = model.predict(df_input)[0]
    st.success(f"💰 Predicted Price: {predicted_price:.2f} Euros")
