# import numpy as np
# Xtrain = np.arange(0,1,0.01).reshape(-1, 1)
# ytrain = 10*Xtrain + np.random.normal(0.0,1.0,100).reshape(-1, 1)
# from sklearn.linear_model import LinearRegressio
# model = LinearRegression().fit(Xtrain.reshape(-1, 1), ytrain.reshape(-1, 1)) 
# print(model.intercept_, model.coef_)

from cProfile import label
# from statistics import LinearRegression
from tkinter import Y
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/doimasanari/Desktop/datasetMasanariDoi.csv')
# Xtrain = []
# Ytrain = []

x1RealPlus = []
x2RealPlus = []

x1RealMinus = []
x2RealMinus = []

# xPredPlus = []
# yPredPlus = []

# xPredMinus = []
# yPredMinus = []


model = LogisticRegression()
# model.fit(xData, resultData)
# print("regression intercept = ", model.intercept_)
# print("slope = ", model.coef_)
# print('correct percentage:', format(model.score(xData, resultData)))

for row in range(len(df)):

    if df.values[row][2] == 1:
        x1RealPlus.append(df.values[row][0])
        x2RealPlus.append(df.values[row][1])
    else:
        x1RealMinus.append(df.values[row][0])
        x2RealMinus.append(df.values[row][1])


def setup(df):
    xData = []
    resultData = []
    xData = df.iloc[:,0:2]
    resultData = df.iloc[:,2]
    xTrain, xTest, resultTrain, resultTest = train_test_split(xData, resultData, test_size=0.2)
    xTestArray =  np.array(xTest)
    return xTrain, xTest, resultTrain, resultTest, xTestArray

# def get_testArray(xTest):
#     return np.array(xTest)

def get_a_decisionBoundary(coefficient, intercept):
    coef = []
    coef = np.array(coefficient)
    coef = coef.reshape(-1,1)
    x = np.linspace(-1, 1, 5)
    y = -((coef[0]*x)+intercept)/coef[1]
    return x, y

def get_a_predData(model, xTestArray):
    predData = []
    predData = np.array(model.predict(xTestArray))
    predData = predData.reshape(-1,1)
    return predData

def get_a_prediction_array(xTestArray, predData):

    # print(resultTest)
    # print(predData)
    # print(predData[185][0])

    # format(model.score(xData, resultTrain)) )

    # xPred = []
    # yPred = [] 

    xPredPlus = []
    yPredPlus = []
    xPredMinus = []
    yPredMinus = []   

    for row in range(len(xTestArray)):

        # xPred.append(xTestArray[row][0])
        # yPred.append(xTestArray[row][1])

        if predData[row][0] == 1:
            xPredPlus.append(xTestArray[row][0])
            yPredPlus.append(xTestArray[row][1])
        else:
            xPredMinus.append(xTestArray[row][0])
            yPredMinus.append(xTestArray[row][1])

    return xPredPlus, yPredPlus, xPredMinus, yPredMinus   

def get_a_graph(x, y, xPredPlus, yPredPlus, xPredMinus, yPredMinus, num):

    plt.figure(num)
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.scatter(x1RealPlus, x2RealPlus, color="blue", label="actual +1")
    plt.scatter(x1RealMinus, x2RealMinus, color="green", label="actual -1")
    plt.scatter(xPredPlus, yPredPlus, color="red", marker="+", label = "predicted +1")
    plt.scatter(xPredMinus, yPredMinus, color="yellow", marker="+", label = "predicted -1")
    plt.plot(x, y, color="black", linewidth = 2, label = "decision boundary")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # plot.legend(loc=2,prop={'size': 10} )
   
    
def logisticRegression(df, num):
    # train_score = format(model.score(xTrain, resultTrain))

    xTrain, xTest, resultTrain, resultTest, xTestArray = setup(df)
    model = LogisticRegression()
    model.fit(xTrain, resultTrain)
    print("slope got by a training = ", model.coef_)
    print("intercept by the training = ", model.intercept_)
    print("train score = ", format(model.score(xTrain, resultTrain)))
    x, y = get_a_decisionBoundary(model.coef_ , model.intercept_)
    # xTestArray = get_testArray(xTest)
    predData = get_a_predData(model, xTestArray)
    xPredPlus, yPredPlus, xPredMinus, yPredMinus = get_a_prediction_array(xTestArray, predData)
    get_a_graph(x, y, xPredPlus, yPredPlus, xPredMinus, yPredMinus, num)

def liner_SVC (c, df, num):

    xTrain, xTest, resultTrain, resultTest, xTestArray = setup(df)
    from sklearn.svm import LinearSVC
    model = LinearSVC(C=c).fit(xTrain, resultTrain)
    print("when C =", c)
    print("slope = ", model.coef_)
    print("intercept = ", model.intercept_)
    x, y = get_a_decisionBoundary(model.coef_ , model.intercept_) 
    # xTestArray = get_testArray(xTest)
    predData = get_a_predData(model, xTestArray)
    xPredPlus, yPredPlus, xPredMinus, yPredMinus = get_a_prediction_array(xTestArray, predData)
    get_a_graph(x, y, xPredPlus, yPredPlus, xPredMinus, yPredMinus, num)



logisticRegression(df, 1)
plt.show()
logisticRegression(df, 2)
liner_SVC (0.001, df, 3)
liner_SVC (1, df, 4)
liner_SVC (100, df, 5)


plt.show()

# def get_an_extended_data():
    
newNum1 = (df.values[0][0])*(df.values[0][0])

# df = np.array(df)
# df = df.reshape(-1,5)
df[0].insert(1,newNum1)
print(df)






# plt.show()

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