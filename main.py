import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

datas = pd.read_csv("satislar.csv")
print(datas)

months = datas[["Aylar"]]
print(months)

sales = datas[["Satislar"]]
print(sales)

# feature scaling
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size=0.33, random_state=0)
print("x train:", y_test)

"""standart_scaler = StandardScaler()

X_train = standart_scaler.fit_transform(x_train)
X_test = standart_scaler.fit_transform(x_test)

Y_train = standart_scaler.fit_transform(y_train)
Y_test = standart_scaler.fit_transform(y_test)
"""
# modelin oluşturulması
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

predict = linear_regression.predict(x_test)
print(predict)
