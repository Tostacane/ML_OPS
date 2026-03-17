if __name__ == "__main__":
    pass



import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv('data/raw/HR-Employee-Attrition.csv')

# Drop these columns that are not useful for modeling and have no variance
df_filtered = df.drop(columns=["EmployeeNumber","EmployeeCount","StandardHours","Over18"]).dropna()

# Convert Attrition to integer (for target variable)
df_filtered["Attrition"] = (df_filtered["Attrition"] == "Yes").astype(int)

# Split the data into training and testing sets
X = df_filtered.drop(columns=["Attrition"])
y = df_filtered["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Define columns for transformation
categorical_cols = ['Department', 'EducationField', 'Gender', 'MaritalStatus', 'JobRole', 'BusinessTravel']
numeric_cols_standard = ["HourlyRate", "DailyRate", "MonthlyRate", "JobInvolvement", "JobSatisfaction",
                         'TrainingTimesLastYear', 'WorkLifeBalance', 'StockOptionLevel', "Age",
                         "Education", "EnvironmentSatisfaction"]
numeric_cols_robust = ["DistanceFromHome", "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole",
                       "YearsSinceLastPromotion", "YearsWithCurrManager", "TotalWorkingYears",
                       "PercentSalaryHike", 'NumCompaniesWorked']

# Function to convert OverTime and Age
def convert_features(X):
    X = X.copy()
    X['OverTime'] = (X['OverTime'] == 'Yes').astype(int)
    X['Age'] = X['Age'].astype('str').astype('int')
    return X

# Create full pipeline with classifier
full_pipeline = Pipeline([
    ('feature_conversion', FunctionTransformer(convert_features, validate=False)),
    ('column_transformer', ColumnTransformer(
        transformers=[
            ("standard", StandardScaler(), numeric_cols_standard),
            ("robust", RobustScaler(), numeric_cols_robust),
            ("onehot", OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=101))
])

# Train the full pipeline
full_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")