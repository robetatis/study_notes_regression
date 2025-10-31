import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
import statsmodels.api as sm


df = load_diabetes()

X = pd.DataFrame(df.data, columns=df.feature_names)
y = pd.Series(df.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

sfs = SequentialFeatureSelector(
    model,
    k_features='best',
    forward=True,
    floating=True,
    scoring='r2',
    cv=5
)

sfs = sfs.fit(X_train, y_train)
print('Selected features:', list(sfs.k_feature_names_))