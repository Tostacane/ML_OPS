from networkx import display
import pandas as pd
import numpy as np

df = pd.read_csv('../data/HR-Employee-Attrition.csv')

print(df.info())
print(df.sample(10))


