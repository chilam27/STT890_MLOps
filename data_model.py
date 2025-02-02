# Import Library & Dataset
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("sampregdata.csv", index_col = 0)
df.head()

# Feature Selection
y = df['y']
X = df.drop('y', axis = 1)

X = sm.add_constant(X)

model_ols = sm.OLS(y, X).fit()
model_ols.pvalues

# Model Training
X_2 = df[['x1', 'x3']]

model = LinearRegression()
model.fit(X_2, y)

mean_squared_error(y, model.predict(X_2)) # 68.6
r2_score(y, model.predict(X_2)) # 0.40
mean_absolute_error(y, model.predict(X_2)) # 6.6
