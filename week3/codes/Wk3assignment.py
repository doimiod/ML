from cProfile import label
from re import X
# from statistics import LinearRegression
from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('/Users/doimasanari/Desktop/ugoke.csv')

print(df)
x = df.iloc[:,0:2]
x1 = df.iloc[:,0]
x2 = df.iloc[:,1]
print(x1)
print(x2)
y = df.iloc[:,2]
# y = np.array(y)
print(y)

fig = plt.figure()
plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, y)

plt.show()

# x1Cube = x1[0]**3

xTrain, xTest, resultTrain, resultTest = train_test_split(x, y, test_size=0.2) # split the data for training and testing.

