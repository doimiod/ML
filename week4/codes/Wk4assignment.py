from cProfile import label
from calendar import c
from re import X
# from statistics import LinearRegression
from tkinter import Y
from turtle import color
from xml.etree.ElementInclude import include
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
from sklearn.model_selection import cross_val_score 


# df = pd.read_csv('/Users/doimasanari/Desktop/ML/week4/codes/# id:18-36--18-0.csv')
df1 = pd.read_csv('/Users/doimasanari/Desktop/ML/week4/codes/# id:12--12-12-1.csv')

df2 = pd.read_csv('/Users/doimasanari/Desktop/ML/week4/codes/# id:12-24-12-1,,.csv')

# x = df.iloc[0:1455, 0:2]

data1X = df1.iloc[:, 0:2]
data1X1 = df1.iloc[:, 0]
data1X2 = df1.iloc[:, 1]
data1Y = df1.iloc[:, 2]

data2X = df2.iloc[:, 0:2]
data2X1 = df2.iloc[:, 0]
data2X2 = df2.iloc[:, 1]
data2Y = df2.iloc[:, 2]

# print(x)


# split the data into 2 here
# for row in range(len(df)):
    
#     if row != 0 and "#" in df.values[row][0]:
#         data1X = df.iloc[0:row, 0:2]
#         data1X1 = df.iloc[0:row, 0]
#         data1X2 = df.iloc[0:row, 1]
#         data1Y = df.iloc[0:row, 2]

#         data2X = df.iloc[row+1:len(df), 0:2]
#         data2X1 = df.iloc[row+1:len(df), 0]
#         data2X2 = df.iloc[row+1:len(df), 1]
#         data2Y = df.iloc[row+1:len(df), 2]
#         break

# print(data1X)
# print(data1X1)
# print(data1X2)
# print(data1Y)
# print(data2X1)
# print(data2X2)
# print(data2Y)


def getANormalGraph(x1, x2, y, onlyThisGraph):
    # it was really unclear if plotting actual graoh and prediction together in the same graoh.
    # the graph below is an actual graph
    plt.figure()
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.scatter(x1[y>0], x2[y>0], color="blue", label="actual +1")
    plt.scatter(x1[y<0], x2[y<0], color="green",label="actual -1")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    if onlyThisGraph == True:
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.title("Actual graph with no prediction")    

# getANormalGraph(data1X1, data1X2, data1Y)
# getANormalGraph(data2X1, data2X2, data2Y)

def Qa(x, x1, x2, y):

    # a(i)
    mean_error=[]
    std_error=[]

    polyPowers = range(0, 10)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2) # split the data for training and testing.
    xTest1 = xTest.iloc[:, 0]
    xTest2 = xTest.iloc[:, 1]
    for polyPower in polyPowers:

        poly = PolynomialFeatures(polyPower)
        xPoly = poly.fit_transform(x)
        xPolyTrain = poly.fit_transform(xTrain)
        xPolyTest = poly.fit_transform(xTest)
        model = LogisticRegression(penalty = "l2").fit(xPolyTrain, yTrain)
        yPred = model.predict(xPolyTest)
        getPredictionPlot(x1, x2, y, xTest1, xTest2, yPred, polyPower, 0)

        score = cross_val_score(model, xPoly, y, cv=5, scoring="f1")
        print("when power of polynomial = ", polyPower)
        print("f1 score = ", score)
        
        # score.append(model.score(xPoly, y))
        # error.append(model.score(xPoly, y))
        # error.append(mean_squared_error(y, yPred))
        mean_error.append(np.array(score).mean()) 
        std_error.append(np.array(score).std())

    plt.figure()
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True 
    plt.errorbar(polyPowers, mean_error, yerr=std_error, linewidth = 1, label = "f1 score")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

    # a (ii)
    print("(ii)")
    mean_error=[]
    std_error=[]
    Ci_range = [0.01, 0.1, 1, 5, 10, 25, 30, 35, 50, ]
    for Ci in Ci_range:
        # polyPowers = range(2)
        # for polyPower in polyPowers:
        poly = PolynomialFeatures(2)
        xPoly = poly.fit_transform(x)
        xPolyTrain = poly.fit_transform(xTrain)
        xPolyTest = poly.fit_transform(xTest) 
        model = LogisticRegression(C = Ci).fit(xPolyTrain, yTrain)
        yPred = model.predict(xPolyTest)
        getPredictionPlot(x1, x2, y, xTest1, xTest2, yPred, 0, Ci)

        score = cross_val_score(model, xPoly, y, cv=5, scoring="f1")
        print("when c = %d, and power of polynomial = %d" % (Ci, 2))
        print("f1 score = ", score)
        mean_error.append(np.array(score).mean()) 
        std_error.append(np.array(score).std())

    plt.figure()    
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=1, label = "f1 score")
    
    plt.xlabel("C value")
    plt.ylabel("F1 Score")
    plt.title("Cross validation of C")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()


def getPredictionPlot(x1, x2, y, xTest1, xTest2, yPred, polyPower, Ci):

    getANormalGraph(x1, x2, y, False)
    plt.scatter(xTest1[yPred>0], xTest2[yPred>0], color="red", marker="+", label = "predicted +1")
    plt.scatter(xTest1[yPred<0], xTest2[yPred<0], color="yellow", marker="+", label = "predicted -1")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    if Ci == 0:
        plt.title("Logistic Regression when power of polynomial = " + str(polyPower))
    else:
        plt.title("Logistic Regression model when C =  " + str(Ci))
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def Qb():
       print("a") 



# get a graph with just a plain data
getANormalGraph(data1X1, data1X2, data1Y, True)
getANormalGraph(data2X1, data2X2, data2Y, True)

Qa(data1X, data1X1, data1X2, data1Y)
Qa(data2X, data2X1, data2X2, data2Y)


# cross_val_score(model, X, y, cv=5, scoring="f1")

def aaa():
    data1X = np.array(data1X).reshape(-1,2)
    data1Y = np.array(data1Y).reshape(-1,1)
    print("x =============", data1X)
    print("y =============", data1Y)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    import matplotlib.pyplot as plt
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error=[]; std_error=[]
    q_range = [1,2,3,4,5,6]
    for q in q_range:
        from sklearn.preprocessing import PolynomialFeatures
        Xpoly = PolynomialFeatures(q).fit_transform(data1X)
        model = LogisticRegression()
        temp=[]; plotted = False
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], data1Y[train])
            ypred = model.predict(Xpoly[test])
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(data1Y[test],ypred))
            if ((q==1) or (q==2) or (q==6)) and not plotted:
                plt.scatter(data1X, data1Y, color="black")
                ypred = model.predict(Xpoly)
                plt.plot(data1X, ypred, color="blue", linewidth=3)
                plt.xlabel("input x")
                plt.ylabel("output y")
                plt.show()
                plotted = True
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
    plt.errorbar(q_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel("q")
    plt.ylabel("Mean square error")
    plt.show()


plt.show()





# data1X = []
# data1X1 = []
# data1X2 = []
# data1Y = []

# data2X = []
# data2X1 = []
# data2X2 = []
# data2Y = []

# found = False

# if row != 0 and isinstance(df.values[row][0], str) == True:
    #     # print(df.values[row][0])
    #     found = True

    # if(found == False):
        # data1X.append(df.values[row][0])
        # data1X.append(df.values[row][1])
        # data1X1.append(df.values[row][0])
        # data1X2.append(df.values[row][1])
        # data1Y.append(df.values[row][2])

    
    # if(found == True):
        
        # data2X.append(df.values[row][0])
        # data2X.append(df.values[row][1])
        # data2X1.append(df.values[row][0])
        # data2X2.append(df.values[row][1])
        # data2Y.append(df.values[row][2])



# data1X = np.array(data1X).reshape(-1,2)
# data2X = np.array(data2X).reshape(-1,2)
# 
# 
# 
# 
# # plt.figure()
        # plt.rc('font', size=18)
        # plt.rcParams["figure.constrained_layout.use"] = True
        # # ax = fig.add_subplot()
        # # ax.scatter(x1, x2, y, color = "blue", label = "trainig data")
        # plt.scatter(x1[y==1], x2[y==1], color = "blue", label = "actual +1")
        # plt.scatter(x1[y==-1], x2[y==-1], color = "green", label = "actual -1")
        # # ax.scatter(x1, x2, y, color = "blue", label = "trainig data")
        # # ax.scatter(xPoly1[yPred==1], xPoly2[yPred==1], yPred, marker="+", color = "red", label = "predicted +1")
        # # ax.scatter(xPoly1[yPred==-1], xPoly2[yPred==-1], yPred, marker="+", color = "yellow", label = "predicted -1")
        # plt.scatter(x1[yPred==1], x2[yPred==1], marker="+", color = "red", label = "predicted +1")
        # plt.scatter(x1[yPred==-1], x2[yPred==-1], marker="+", color = "yellow", label = "predicted -1")
        # plt.xlabel("x1")
        # plt.ylabel("x2")
        # # ax.set_zlabel("y predicted", color="green", size=15)
        # plt.title("Logistic Regression when polynomial degree is up to = "+ str(polyNum))
    # plt.figure()