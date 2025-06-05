import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cdist

df_linear = pd.read_csv('datasets/linear_regdataset.csv')
df_lwr = pd.read_csv('datasets/LWRdataset.csv')
df_poly = pd.read_csv('datasets/poly_regdataset.csv')

def linear_regression(df):
    X,y = df[['X']],df['Y']
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    plt.scatter(X,y,label = 'Data')
    plt.plot(X,y_pred,color = 'red', label='Linear Regression')
    plt.legend()
    plt.title('Linear Regression')
    plt.show()
linear_regression(df_linear)

def polynomial_regression(df, degree):
    X, Y = df[['X']], df['Y']
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, Y)
    y_pred = model.predict(X)
    plt.scatter(X, Y, label='Data')
    plt.plot(X, y_pred, color='red', label=f'Polynomial Regression (deg = {degree})')
    plt.legend()
    plt.title('Polynomial Regression')
    plt.show()

polynomial_regression (df_poly, degree=3)
