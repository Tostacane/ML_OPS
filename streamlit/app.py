import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd

import streamlit as st
from src.models.persistence import load_model


MODEL = load_model("hr_log_reg.pkl")

st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="centered"
)

st.title("Employee Attrition Prediction App")
st.write("Enter employee information to estimate the probability of attrition.")

st.subheader("Employee Data")

# TODO: add these parameters with streamlit and add also the bounds

"""
    ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
    'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager']
"""

age = st.number_input("Age", min_value=0, max_value=100, value=0)
income = st.number_input("Income", min_value=1000, max_value=50000, value=1000)
overtime = st.selectbox("Overtime", ["Yes", "No"])
jobrole = st.selectbox(
    "Job Role",
    [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director", "Human Resources"
    ]
)

input_data = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": income,
    "Overtime": overtime,
    "JobRole": jobrole
}])

if st.button("Predict Attrition"):
    pred = MODEL.predict(input_data)[0]
    proba = MODEL.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if pred == 1:
        st.error(f"The model predicts **Attrition = YES** with probability {proba:.2f}")
    else:
        st.success(f"The model predicts **Attrition = NO** with probability {proba:.2f}")

    st.write("Input data used for prediction:")
    st.dataframe(input_data)
