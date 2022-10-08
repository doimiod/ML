# import numpy as np
# Xtrain = np.arange(0,1,0.01).reshape(-1, 1)
# ytrain = 10*Xtrain + np.random.normal(0.0,1.0,100).reshape(-1, 1)
# from sklearn.linear_model import LinearRegressio
# model = LinearRegression().fit(Xtrain.reshape(-1, 1), ytrain.reshape(-1, 1)) 
# print(model.intercept_, model.coef_)

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

df = pd.read_csv('/Users/doimasanari/Desktop/datasetMasanariDoi.csv')
# Xtrain = []
# Ytrain = []

x1RealPlus = []
x2RealPlus = []

x1RealMinus = []
x2RealMinus = []

# x1PredPlus = []
# x2PredPlus = []

# xPredMinus = []
# x2PredMinus = []

xData = df.iloc[:,0:2]
resultData = df.iloc[:,2]

model = LogisticRegression()
model.fit(xData, resultData)
print("regression intercept = ", model.intercept_)
print("slope = ", model.coef_)
print('correct percentage:', format(model.score(xData, resultData)))

# distinguish x1 and x2 according to its value, 1 or -1.
for row in range(len(df)):

    if df.values[row][2] == 1:
        x1RealPlus.append(df.values[row][0])
        x2RealPlus.append(df.values[row][1])
    else:
        x1RealMinus.append(df.values[row][0])
        x2RealMinus.append(df.values[row][1])

# split the data for training and testing.
xTrain, xTest, resultTrain, resultTest = train_test_split(xData, resultData, test_size=0.2) 
# get_testArray(xTest):
xTestArray = np.array(xTest)
# print(xTestArray)

def get_a_decisionBoundary(coefficient, intercept):
    coef = np.array(coefficient)
    coef = coef.reshape(-1,1)
    x = np.linspace(-1, 1, 5)
    y = -((coef[0]*x)+intercept)/coef[1]
    return x, y

def get_a_predData(model, xTestArray):
    # preData = model.predict(xTestArray)
    # print(preData)
    predData = np.array(model.predict(xTestArray))
    predData = predData.reshape(-1,1)
    return predData

def get_a_prediction_array(xTestArray, predData):

    # print(resultTest)
    # print(predData)
    # print(predData[185][0])

    # format(model.score(xData, resultTrain)) )

    xPred = []
    yPred = [] 

    x1PredPlus = []
    x2PredPlus = []
    x1PredMinus = []
    x2PredMinus = []   

    for row in range(len(xTestArray)):

        # xPred.append(xTestArray[row][0])
        # yPred.append(xTestArray[row][1])

        if predData[row][0] == 1:
            x1PredPlus.append(xTestArray[row][0])
            x2PredPlus.append(xTestArray[row][1])
        else:
            x1PredMinus.append(xTestArray[row][0])
            x2PredMinus.append(xTestArray[row][1])

    return x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus   

def get_a_graph(x, y, x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus, num):
    
    plt.figure(num)
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.scatter(x1RealPlus, x2RealPlus, color="blue", label="actual +1")
    plt.scatter(x1RealMinus, x2RealMinus, color="green", label="actual -1")
    plt.scatter(x1PredPlus, x2PredPlus, color="red", marker="+", label = "predicted +1")
    plt.scatter(x1PredMinus, x2PredMinus, color="yellow", marker="+", label = "predicted -1")
    plt.plot(x, y, color="black", linewidth = 2, label = "decision boundary")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def logisticRegression(xTrain, resultTrain, xTestArray, num, questionNum):
    model = LogisticRegression()
    model.fit(xTrain, resultTrain)
    # train_score = format(model.score(xTrain, resultTrain))
    print("slope got by a training = ", model.coef_)
    print("intercept by the training = ", model.intercept_)
    print("train score = ", format(model.score(xTrain, resultTrain)))
    if(questionNum==1):
        x, y = get_a_decisionBoundary(model.coef_ , model.intercept_)
    else :
        x = 0
        y = 0
    # print("xTestArray =", xTestArray)
    predData = get_a_predData(model, xTestArray)
    x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus = get_a_prediction_array(xTestArray, predData)
    get_a_graph(x, y, x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus, num)


def liner_SVC (c, xTrain, resultTrain, num):
    
    model = LinearSVC(C=c).fit(xTrain, resultTrain)
    print("when C =", c)
    print("slope = ", model.coef_)
    print("intercept = ", model.intercept_)
    x, y = get_a_decisionBoundary(model.coef_ , model.intercept_) 
    predData = get_a_predData(model, xTestArray)
    x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus = get_a_prediction_array(xTestArray, predData)
    get_a_graph(x, y, x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus, num)



# for question (a) and (b)

logisticRegression(xTrain, resultTrain, xTestArray,0, 1)
plt.show()
logisticRegression(xTrain, resultTrain, xTestArray, 1, 1)
# print(xTrain)
liner_SVC (0.001, xTrain, resultTrain, 2)
liner_SVC (1, xTrain, resultTrain, 3)
liner_SVC (100, xTrain, resultTrain, 4)
# print(xTrain)

# plt.show()

# print(xData)

def new_training(xData):

    xData = np.array(xData)
    tempXData = xData
    # print(tempXData)
    fCul = 2
    secCul = 3
    
    for row in range(len(xData)):
        newFeature1 = tempXData[row][0]*tempXData[row][0]
        newFeature2 = tempXData[row][1]*tempXData[row][1]
        xData = np.insert(xData, fCul, newFeature1)
        xData = np.insert(xData, secCul, newFeature2)
        fCul = fCul +4
        secCul = secCul + 4

    xData = xData.reshape(-1, 4)
    xTrain, xTest, resultTrain, resultTest = train_test_split(xData, resultData, test_size=0.5)
    # print(xData)
    return xData, xTrain, xTest, resultTrain, resultTest


xData, xTrain, xTest, resultTrain, resultTest = new_training(xData)
xTestArray = np.array(xTest)
logisticRegression(xTrain, resultTrain, xTestArray, 5, 2)
# print(xData)
# print(xTestArray)

plt.show()


















# model = LinearSVC(C=1.0).fit(xTrain, resultTrain)
# print("when C = 1.0")
# print("intercept = ", model.intercept_)
# print("slope = ", model.coef_)

# model = LinearSVC(C=100).fit(xTrain, resultTrain)
# print("when C = 100")
# print("intercept = ", model.intercept_)
# print("slope = ", model.coef_)



# # ["+1","-1", "predict +1", "predict -1"]
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.read_csv("/Users/doimasanari/Desktop/datasetMasanariDoi.csv")
# print(df.iloc[:,0:2])
# print(df.iloc[:,2])
# print(df.values[0][0])
# print(df.values[1][1])
# print(len(df))
# print(df.size)
# # X1=df.iloc[:,0]
# # X2=df.iloc[:,1] 
# # X=np.column_stack((X1,X2)) 
# # y=df.iloc [:,2]
# # print(X1)
