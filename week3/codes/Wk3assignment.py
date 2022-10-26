from cProfile import label
from re import X
# from statistics import LinearRegression
from tkinter import Y
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

df = pd.read_csv('/Users/doimasanari/Desktop/week3.csv')

x = df.iloc[:, 0:2]
x1 = df.iloc[:, 0]
x2 = df.iloc[:, 1]
y = df.iloc[:, 2]


# Plot the data I downloaded as a 3D scatter
fig = plt.figure()
plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, y)
ax.set_xlabel("x1", color="green", size=15)
ax.set_ylabel("x2", color="green", size=15)
ax.set_zlabel("y", color="green", size=15)
ax.set_title("3d plot")

# x1Cube = x1[0]**3


poly = PolynomialFeatures(5)
xPoly = poly.fit_transform(x)
# print(xPoly)
Xtest = []
grid = np.linspace(-2, 2)
for i in grid:
    for j in grid:
        Xtest.append([i, j])
Xtest = np.array(Xtest)
# print(Xtest)
XtestPoly = poly.fit_transform(Xtest)

Xtest1 = Xtest[:, 0]
Xtest2 = Xtest[:, 1]

def lassoRegression(c):
    # model = linear_model.Lasso(alpha = 1/(2*c)) 1/c or 1/2c does not really matter
    model = linear_model.Lasso(alpha = 1/c) # train by Lasso Regression
    model.fit(xPoly, y)
    print("when c = ", c)
    print("slope = ", model.coef_)             # get a slope here
    print("intercept = ", model.intercept_)   # get an intercept here

    yPred = model.predict(XtestPoly)
    
    fig = plt.figure()
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y, color = "blue", label = "trainig data")
    ax.plot_trisurf(Xtest1, Xtest2, yPred, color = "green", label = "prediction")
    ax.set_xlabel("x1", color="green", size=15)
    ax.set_ylabel("x2", color="green", size=15)
    ax.set_zlabel("y predicted", color="green", size=15)
    ax.set_title("Lasso Regression. 3D plot when c = " + str(c))
    # ax.legend(loc= "upper right" , bbox_to_anchor=(0.6,0.5))



def ridgeRegression(c):
    model = linear_model.Ridge(alpha = 1/(2*c))     # train by Ridges Regression
    model.fit(xPoly, y)
    print("when c = ", c)
    print("slope = ", model.coef_)             # get a slope here
    print("intercept = ", model.intercept_)   # get an intercept here
    yPred = model.predict(XtestPoly)
    fig = plt.figure()
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y, color = "blue", label = "trainig data")
    ax.plot_trisurf(Xtest1, Xtest2, yPred, color = "green", label = "prediction")
    ax.set_xlabel("x1", color="green", size=15)
    ax.set_ylabel("x2", color="green", size=15)
    ax.set_zlabel("y predicted", color="green", size=15)
    ax.set_title("Ridge Regression 3D plot when c = " + str(c))
    # ax.legend(loc= "upper right" , bbox_to_anchor=(0.6,0.5))


# question (i)
print("When using Lasso Regression")
lassoRegression(0.001)
lassoRegression(1.0)
lassoRegression(10)
lassoRegression(100)
lassoRegression(1000)
lassoRegression(10000)

print("when using Ridge Regression")
ridgeRegression(0.001)
ridgeRegression(1.0)
ridgeRegression(10)
ridgeRegression(100)
ridgeRegression(1000)
ridgeRegression(10000)

plt.show()
