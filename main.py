import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

datas = pd.read_csv("satislar.csv")
print(datas)

months = datas[["Aylar"]]
print(months)

sales = datas[["Satislar"]]
print(sales)



