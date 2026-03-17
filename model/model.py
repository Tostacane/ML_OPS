import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn

df = pd.read_csv('../data/HR-Employee-Attrition.csv')

print(df.info())
print(df.sample(10))



# Preprocessing

# Drop these columns that are not useful for modeling and have no variance
df_filtered = df.drop(columns=["EmployeeNumber","EmployeeCount","StandardHours","Over18"]).dropna()

# Convert 'OverTime' to boolean and age to integer
df_filtered['OverTime'] = (df_filtered['OverTime']=='Yes').astype(int)
df_filtered["Attrition"] = (df_filtered["Attrition"] == "Yes").astype(int)
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

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=ct.get_feature_names_out()
).astype("float32")

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=ct.get_feature_names_out()
).astype("float32")

print(X_train_scaled.head())
print(X_test_scaled.head())


# Classification Neural Network

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = BinaryClassifier(input_dim=X_train_scaled.shape[1])
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X = torch.tensor(X_train_scaled.values, dtype=torch.float32)
y = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

for epoch in range(0, 101):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    logits = model(torch.tensor(X_test_scaled.values, dtype=torch.float32))
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()

    print("Preds:", preds[:30].flatten())
    print("Probs:", probs[:30].flatten())
    print("True labels:", y_test.values[:30])
