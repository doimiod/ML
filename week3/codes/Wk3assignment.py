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


df = pd.read_csv('/Users/doimasanari/Desktop/ugoke.csv')

# print(df)
x = df.iloc[:, 0:2]
x1 = df.iloc[:, 0]
x2 = df.iloc[:, 1]
print(x)
# print(x1)
# print(x2)
y = df.iloc[:, 2]
# y = np.array(y)
# print(y)

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


# q (i) (b)
poly = PolynomialFeatures(5)
xPoly = poly.fit_transform(x)
print(xPoly)

Xtest = []
grid = np.linspace(5, 5)
for i in grid:
    for j in grid:
        Xtest.append([i, j])
Xtest = np.array(Xtest)
print(Xtest)
XtestPoly = poly.fit_transform(Xtest)

Xtest1 = Xtest[:, 0]
print(Xtest1)
Xtest2 = Xtest[:, 1]


def lassoRegression(c):
    model = linear_model.Lasso(alpha=(1/c))
    model.fit(xPoly, y)
    print("when c = ", c)
    print("slope = ", model.coef_)             # get a slope here
    print("intercept = ", model.intercept_)   # get an intercept here

       
    yPred = model.predict(XtestPoly)
    
    fig = plt.figure()
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Xtest1, Xtest2, yPred)
    ax.plot_trisurf(x1, x2, yPred)
    ax.set_xlabel("x1", color="green", size=15)
    ax.set_ylabel("x2", color="green", size=15)
    ax.set_zlabel("y predicted", color="green", size=15)
    ax.set_title("3d plot when c = " + str(c))



def ridgeRegression(c):
    model = linear_model.Ridge(alpha = 1/(2*c))


lassoRegression(0.001)
lassoRegression(1.0)
lassoRegression(10)
lassoRegression(100)
lassoRegression(1000)

# xTrain, xTest, resultTrain, resultTest = train_test_split(x, y, test_size=0.2) # split the data for training and testing.

plt.show()
