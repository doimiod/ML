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

x1RealPlus = []
x2RealPlus = []

x1RealMinus = []
x2RealMinus = []

xData = df.iloc[:,0:2]     # construct a 999x2 matrix which consists only x1 and x2
resultData = df.iloc[:,2]  # construct a 999x1 matrix which consists only values 1 or -1

for row in range(len(df)):    # this for-loop distinguishs x1 and x2 according to its value, 1 or -1.

    if df.values[row][2] == 1:
        x1RealPlus.append(df.values[row][0])
        x2RealPlus.append(df.values[row][1])
    else:
        x1RealMinus.append(df.values[row][0])
        x2RealMinus.append(df.values[row][1])

xTrain, xTest, resultTrain, resultTest = train_test_split(xData, resultData, test_size=0.2) # split the data for training and testing.

xTestArray = np.array(xTest) #make an array of x test data

def get_a_decisionBoundary(coefficient, intercept):
    coef = np.array(coefficient)            # 
    coef = coef.reshape(-1,1)               # make a coefficient array readable
    x = np.linspace(-1, 1, 5)               # 
    y = -((coef[0]*x)+intercept)/coef[1]    # x2 = - (θ1x1+ θ0)/θ1
    return x, y

def get_a_predData(model, xTestArray):               # get a prediction data in this function
    predData = np.array(model.predict(xTestArray))   
    predData = predData.reshape(-1,1)                # make a tidy 999x1 array of prediction data which contains values, 1 or -1
    return predData

def get_a_prediction_array(xTestArray, predData):  # distinguish x1 and x2 of test data according to its value, 1 or -1,
                                                   # in order to make an array of x prediced data
    x1PredPlus = []
    x2PredPlus = []
    x1PredMinus = []
    x2PredMinus = []   

    for row in range(len(xTestArray)):                   

        if predData[row][0] == 1:                       #if x1 and x2 are predicted to have +1
            x1PredPlus.append(xTestArray[row][0])     
            x2PredPlus.append(xTestArray[row][1])
        else:
            x1PredMinus.append(xTestArray[row][0])      #if x1 and x2 are predicted to have -1
            x2PredMinus.append(xTestArray[row][1])

    return x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus   

def get_a_graph(x, y, x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus, num): # make a graph here

    plt.figure(num)
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.scatter(x1RealPlus, x2RealPlus, color="blue", label="actual +1")
    plt.scatter(x1RealMinus, x2RealMinus, color="green", label="actual -1")
    plt.scatter(x1PredPlus, x2PredPlus, color="red", marker="+", label = "predicted +1")
    plt.scatter(x1PredMinus, x2PredMinus, color="yellow", marker="+", label = "predicted -1")
    if(num != 5):
        plt.plot(x, y, color="black", linewidth = 2, label = "decision boundary")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

def logisticRegression(xTrain, resultTrain, xTestArray, num, questionNum): # train data by logistic Regression

    model = LogisticRegression()                       
    model.fit(xTrain, resultTrain)                       # train data
    print("slope = ", model.coef_)                       # get a slope here
    print("intercept = ", model.intercept_)              # get an intercept here
    # print("train score = ", format(model.score(xTrain, resultTrain)))
    if(questionNum == 1):
        x, y = get_a_decisionBoundary(model.coef_ , model.intercept_)      # go and get a decision boundary
    else :
        x = 0
        y = 0
    predData = get_a_predData(model, xTestArray)                           # go and get a prediction data
    x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus = get_a_prediction_array(xTestArray, predData) # go and get arrays of predited x data
    get_a_graph(x, y, x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus, num) # go and get a graph


def liner_SVC (c, xTrain, resultTrain, num):                                # train data by logisticRegression

    model = LinearSVC(C=c).fit(xTrain, resultTrain)                         # train a data
    print("when C =", c)                                                    
    print("slope = ", model.coef_)                                          # get a slope here
    print("intercept = ", model.intercept_)                                 # get an intercept here
    x, y = get_a_decisionBoundary(model.coef_ , model.intercept_)           # go and get a decision boundary
    predData = get_a_predData(model, xTestArray)                            # go and get a prediction data
    x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus = get_a_prediction_array(xTestArray, predData)  # go and get arrays of predited x data
    get_a_graph(x, y, x1PredPlus, x2PredPlus, x1PredMinus, x2PredMinus, num)  # go and get a graph

def dummy_data ():    # dummy graph data. close the graph window then check an actual graphs

    plt.figure(0)
    plt.rc('font', size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.plot(0, 0, color="black", linewidth = 2, label = "figure 1 for q(a),2,3 and 4 for q(b) and 5 for q(c)")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


dummy_data() #this is dummy data. plese close this graph and see an actual graph

# for question (a) and (b)
print("Q(a) (ii)") 
logisticRegression(xTrain, resultTrain, xTestArray, 1, 1) #this is for question (a)

print("Q(b) (i)")
liner_SVC (0.001, xTrain, resultTrain, 2) # these three linear_SVC are answering question (b)
liner_SVC (1.0, xTrain, resultTrain, 3)
liner_SVC (100, xTrain, resultTrain, 4)

# below is for question (c)
def new_training(xData): # this function makes new training data, i.e. adding the square of each feature.

    xData = np.array(xData)
    tempXData = xData
    fCul = 2
    secCul = 3
    
    for row in range(len(xData)): 
        newFeature1 = tempXData[row][0]*tempXData[row][0] # making square of original two features
        newFeature2 = tempXData[row][1]*tempXData[row][1]
        xData = np.insert(xData, fCul, newFeature1)       # adding new squared features
        xData = np.insert(xData, secCul, newFeature2)
        fCul = fCul +4
        secCul = secCul + 4

    xData = xData.reshape(-1, 4)                          # make a 999x4 array of x data that includes new squared features.
    xTrain, xTest, resultTrain, resultTest = train_test_split(xData, resultData, test_size=0.2) # split new data for training and testing

    return xData, xTrain, xTest, resultTrain, resultTest

xData, xTrain, xTest, resultTrain, resultTest = new_training(xData)
xTestArray = np.array(xTest)
print("Q(c) (i)") 
logisticRegression(xTrain, resultTrain, xTestArray, 5, 2) # answering question (c)

plt.show()


