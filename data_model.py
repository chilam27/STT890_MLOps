# Import Library & Dataset
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

df = pd.read_csv("sampregdata.csv", index_col = 0)
df.head()

# Feature Selection
y = df['y']
X = df.drop('y', axis = 1)

X = sm.add_constant(X)

model_ols = sm.OLS(y, X).fit()
model_ols.pvalues

# Model Training
X_1 = df[['x1']]

model = LinearRegression()
model.fit(X_1, y)
