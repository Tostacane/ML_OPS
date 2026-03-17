import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

df = pd.read_csv('../data/HR-Employee-Attrition.csv')

print(df.info())
print(df.sample(10))



# Preprocessing

# Drop these columns that are not useful for modeling and have no variance
df_filtered = df.drop(columns=["EmployeeNumber","EmployeeCount","StandardHours","Over18"]).dropna()

# Convert 'OverTime' to boolean and age to integer
df_filtered['OverTime'] = df_filtered['OverTime']=='Yes'
df_filtered['Age'] = df_filtered['Age'].astype('str').astype('int')


df_filtered = pd.get_dummies(df_filtered,columns=['Department','EducationField','Gender','MaritalStatus','JobRole','BusinessTravel'])


# Split the data into training and testing sets
X = df_filtered.drop(columns=["Attrition"])
y = df_filtered["Attrition"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

ct = ColumnTransformer(
    transformers=[
        ("standard",StandardScaler(),["HourlyRate","DailyRate","MonthlyRate","JobInvolvement","JobSatisfaction",'TrainingTimesLastYear','WorkLifeBalance','StockOptionLevel',"Age","Education","EnvironmentSatisfaction"]),
        ("robust",RobustScaler(),["DistanceFromHome","MonthlyIncome","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","TotalWorkingYears","PercentSalaryHike",'NumCompaniesWorked']) 
    ],
    remainder="passthrough"
)

X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

columns = [
    "HourlyRate","DailyRate","MonthlyRate","JobInvolvement","JobSatisfaction",'TrainingTimesLastYear','WorkLifeBalance','StockOptionLevel',"Age","Education","EnvironmentSatisfaction",
    "DistanceFromHome","MonthlyIncome","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","TotalWorkingYears","PercentSalaryHike",'NumCompaniesWorked',
       'OverTime',
       'PerformanceRating',
       'RelationshipSatisfaction','Department_Human Resources', 'Department_Research & Development',
       'Department_Sales', 'EducationField_Human Resources',
       'EducationField_Life Sciences', 'EducationField_Marketing',
       'EducationField_Medical', 'EducationField_Other',
       'EducationField_Technical Degree', 'Gender_Male',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'JobRole_Healthcare Representative',
       'JobRole_Human Resources', 'JobRole_Laboratory Technician',
       'JobRole_Manager', 'JobRole_Manufacturing Director',
       'JobRole_Research Director', 'JobRole_Research Scientist',
       'JobRole_Sales Executive', 'JobRole_Sales Representative',
       'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
       'BusinessTravel_Travel_Rarely'
]

X_train_scaled = pd.DataFrame(X_train_scaled,columns=columns)
X_test_scaled =  pd.DataFrame(X_test_scaled,columns=columns)

