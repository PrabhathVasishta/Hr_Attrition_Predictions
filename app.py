import streamlit as st
import pandas as pd
import pickle

# ===== LOAD DATASET =====
df = pd.read_csv("HR-Employee-Attrition.csv")

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

# ===== TITLE =====
st.title("📊 Employee Attrition Prediction System")
st.write("Predict whether an employee is likely to leave the company.")

# ===== DATASET RANGES =====
age_min = int(df['Age'].min())
age_max = int(df['Age'].max())

income_min = int(df['MonthlyIncome'].min())
income_max = int(df['MonthlyIncome'].max())

years_min = int(df['YearsAtCompany'].min())
years_max = int(df['YearsAtCompany'].max())

# ===== INPUTS =====

Age = st.number_input(
    "Age",
    min_value=age_min,
    max_value=age_max,
    value=30
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=income_min,
    max_value=income_max,
    value=5000,
    step=1000
)

JobLevel = st.selectbox(
    "Job Level",
    [1, 2, 3, 4, 5]
)

JobSatisfaction = st.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4]
)

WorkLifeBalance = st.selectbox(
    "Work Life Balance",
    [1, 2, 3, 4]
)

YearsAtCompany = st.number_input(
    "Years At Company",
    min_value=years_min,
    max_value=years_max,
    value=5
)

OverTime_option = st.selectbox(
    "OverTime",
    ["No", "Yes"]
)

OverTime = 1 if OverTime_option == "Yes" else 0

# ===== PREDICT BUTTON =====

if st.button("Predict Attrition"):

    input_data = [[
        Age,
        MonthlyIncome,
        JobLevel,
        JobSatisfaction,
        WorkLifeBalance,
        YearsAtCompany,
        OverTime
    ]]

    # SCALE INPUT
    input_scaled = scaler.transform(input_data)

    # PREDICT
    prediction = model.predict(input_scaled)

    # OUTPUT
    if prediction[0] == 1:
        st.error("⚠ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay in the company.")