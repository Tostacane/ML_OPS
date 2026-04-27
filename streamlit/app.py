import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.models.persistence import load_model
from src.data.preprocess import cast_columns
from src.utils.config import load_config


MODEL = load_model("model.pkl")
_PREPROCESS_CONFIG = load_config("preprocess")["preprocess"]


st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="centered"
)

st.title("Employee Attrition Prediction App")
st.write("Enter employee information to estimate the probability of attrition.")

st.subheader("Employee Data")


age = st.number_input("Age", min_value=0, max_value=120, value=0)

business_travel = st.selectbox(
    "BusinessTravel",
    [
        "Travel_Rarely", "Travel_Frequently", "Non-Travel"
    ]
)

daily_rate = st.number_input("Daily Rate", min_value=0, max_value = 2000, value = 0)

department = st.selectbox(
    "Department",
    [
        "Sales", "Research & Development", "Human Resources"
    ]
)

distance_from_home = st.number_input("Distance From Home", min_value=0, max_value = 100, value = 0)

education = st.number_input("Education", min_value=0, max_value = 10, value = 0)

education_field = st.selectbox(
    "Educational Field",
    [
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
    ]
)

environment_satisfaction = st.number_input("Environment Satisfaction", min_value=0, max_value = 10, value = 0)

gender = st.selectbox(
    "Gender",
    [
        "Male", "Female"
    ]
)

hourly_rate = st.number_input("Hourly Rate", min_value=0, max_value = 200, value = 0)

job_involvement = st.number_input("Job Involvement", min_value=0, max_value = 10, value = 0)

job_level = st.number_input("Job Level", min_value=0, max_value = 10, value = 0)

job_role = st.selectbox(
    "Job Role",
    ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']
)

job_satisfaction = st.number_input("Job Satisfaction", min_value=0, max_value = 10, value = 0)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

monthly_income = st.number_input("Monthly Income", min_value=0, max_value = 30000, value = 0)

monthly_rate = st.number_input("Monthly Rate", min_value=0, max_value = 30000, value = 0)

num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value = 10, value = 0)

overtime = st.selectbox("Overtime", ["Yes", "No"])

percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value = 50, value = 0)

performance_rating = st.number_input("Performance Rating", min_value=0, max_value = 10, value = 0)

relationship_rating = st.number_input("Relationship Satisfaction", min_value=0, max_value = 10, value = 0)

stock_option_level = st.number_input("Stock Option Level", min_value=0, max_value = 10, value = 0)

total_working_years = st.number_input("Total working years", min_value=0, max_value = 50, value = 0)

training_times_last_year = st.number_input("Training times last year", min_value=0, max_value = 10, value = 0)

work_life_balance = st.number_input("Work-Life Balance", min_value=0, max_value = 10, value = 0)

years_at_company = st.number_input("Years at company", min_value=0, max_value = 50, value = 0)
years_in_current_role = st.number_input("Years in current role", min_value=0, max_value = 30, value = 0)
years_since_last_promotion = st.number_input("Years since last promotion", min_value=0, max_value = 30, value = 0)
years_with_current_manager = st.number_input("Years with current manager", min_value=0, max_value = 30, value = 0)


input_data = pd.DataFrame([{
    'Age': age,
    'BusinessTravel': business_travel,
    'DailyRate' : daily_rate,
    'Department' : department,
    'DistanceFromHome' : distance_from_home,
    'Education' : education,
    'EducationField' : education_field,
    'EnvironmentSatisfaction' : environment_satisfaction,
    'Gender' : gender,
    'HourlyRate' : hourly_rate,
    'JobInvolvement' : job_involvement,
    'JobLevel' : job_level,
    'JobRole' : job_role,
    'JobSatisfaction' : job_satisfaction,
    'MaritalStatus' : marital_status,
    'MonthlyIncome' : monthly_income,
    'MonthlyRate' : monthly_rate,
    'NumCompaniesWorked' : num_companies_worked,
    'OverTime' : overtime,
    'PercentSalaryHike' : percent_salary_hike,
    'PerformanceRating' : performance_rating,
    'RelationshipSatisfaction' : relationship_rating,
    'StockOptionLevel' : stock_option_level,
    'TotalWorkingYears' : total_working_years,
    'TrainingTimesLastYear' : training_times_last_year,
    'WorkLifeBalance' : work_life_balance,
    'YearsAtCompany' : years_at_company,
    'YearsInCurrentRole' : years_in_current_role,
    'YearsSinceLastPromotion' : years_since_last_promotion,
    'YearsWithCurrManager' : years_with_current_manager,
}])

input_data = cast_columns(input_data, _PREPROCESS_CONFIG.get("cast_columns", {}))


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
