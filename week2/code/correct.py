from cProfile import label
from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/doimasanari/Desktop/datasetMasanariDoi.csv')
# Xtrain = []
# Ytrain = []

xPredPlus = []
yPredPlus = []

xPredMinus = []
yPredMinus = []

XtrainGreen = []
YtrainGreen = []
XtrainBlue = []
YtrainBlue = []
x_yData = df.iloc[:,0:2]
resultData = df.iloc[:,2]

model = LogisticRegression()
model.fit(x_yData, resultData)
print("regression intercept = ", model.intercept_)
print("slope = ", model.coef_)
print('correct percentage:', format(model.score(x_yData, resultData)))

# print(x_yData)
# print(resultData)

# x_yData = np.arange(0,1,0.05).reshape(-1, 1)
# resultData = 10*x_yData + np.random.normal(0.0,1.0,x_yData.size).reshape(-1, 1)

x_yTrain, x_yTest, resultTrain, resultTest = train_test_split(x_yData, resultData, test_size=0.2) 
model = LogisticRegression()
model.fit(x_yTrain, resultTrain)
# train_score = format(model.score(x_yTrain, resultTrain))
print("slope got by train = ", model.coef_)
print("train score = ", format(model.score(x_yTrain, resultTrain)))



# print(resultTest)
x_yTest = np.array(x_yTest)
# print(x_yTest)
predData = np.array(model.predict(x_yTest))
predData = predData.reshape(-1,1)
# print(predData)
# print(predData[185][0])

coef = np.array(model.coef_)
coef = coef.reshape(-1,1)
# intercept = np.array(model.intercept_)
x = np.linspace(-1, 1, 5)
y = -((coef[0]*x)+model.intercept_)/coef[1]

# format(model.score(x_yData, resultTrain)) )

for row in range(len(df)):

    if df.values[row][2] == 1:
        XtrainBlue.append(df.values[row][0])
        YtrainBlue.append(df.values[row][1])
    else:
        XtrainGreen.append(df.values[row][0])
        YtrainGreen.append(df.values[row][1])

xPredPlus = []
yPredPlus = []

xPredMinus = []
yPredMinus = []

for row in range(len(x_yTest)):
    if predData[row][0] == 1:
        xPredPlus.append(x_yTest[row][0])
        yPredPlus.append(x_yTest[row][1])
    else:
        xPredMinus.append(x_yTest[row][0])
        yPredMinus.append(x_yTest[row][1])

    
plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True
plt.scatter(XtrainBlue, YtrainBlue, color="blue", label="+1")
plt.scatter(XtrainGreen, YtrainGreen, color="green", label="-1")
plt.scatter(xPredPlus, yPredPlus, color="red", marker="+", label = "predict +1")
plt.scatter(xPredMinus, yPredMinus, color="yellow", marker="+", label = "predict -1")
plt.plot(x, y, color="black", linewidth = 2, label = "decision boundary")
# plt.plot(x_yTrain, predData, color="red", linewidth=3)
# plt.scatter(xPred, yPred, color="red")

# plt.plot(Xtrain, ypred, color="blue", linewidth=3) 
plt.xlabel("x_1"); 
plt.ylabel("y_2")
plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0 ) 
plt.show()
# ["+1","-1", "predict +1", "predict -1"]

from sklearn.svm import LinearSVC

model = LinearSVC(C=0.001).fit(x_yTrain, resultTrain)
print("when C = 0.001")
print("intercept = ", model.intercept_)
print("slope = ", model.coef_)

x_yTest = np.array(x_yTest)
# print(x_yTest)
predData = np.array(model.predict(x_yTest))
predData = predData.reshape(-1,1)

coef = np.array(model.coef_)
coef = coef.reshape(-1,1)
# intercept = np.array(model.intercept_)
x = np.linspace(-1, 1, 5)
y = -((coef[0]*x)+model.intercept_)/coef[1]


xPredPlus = []
yPredPlus = []

xPredMinus = []
yPredMinus = []

for row in range(len(x_yTest)):

    if predData[row][0] == 1:
        xPredPlus.append(x_yTest[row][0])
        yPredPlus.append(x_yTest[row][1])
    else:
        xPredMinus.append(x_yTest[row][0])
        yPredMinus.append(x_yTest[row][1])


plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True
plt.scatter(XtrainBlue, YtrainBlue, color="blue", label="+1")
plt.scatter(XtrainGreen, YtrainGreen, color="green", label="-1")
plt.scatter(xPredPlus, yPredPlus, color="red", marker="+", label = "predict +1")
plt.scatter(xPredMinus, yPredMinus, color="yellow", marker="+", label = "predict -1")
plt.plot(x, y, color="black", linewidth = 2, label = "decision boundary")

# plt.plot(x_yTrain, predData, color="red", linewidth=3)
# plt.scatter(xPred, yPred, color="red")

# plt.plot(Xtrain, ypred, color="blue", linewidth=3) 
plt.xlabel("x_1"); 
plt.ylabel("y_2")
plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0 ) 

plt.show()


model = LinearSVC(C=1.0).fit(x_yTrain, resultTrain)
print("when C = 1.0")
print("intercept = ", model.intercept_)
print("slope = ", model.coef_)


x_yTest = np.array(x_yTest)
# print(x_yTest)
predData = np.array(model.predict(x_yTest))
predData = predData.reshape(-1,1)

coef = np.array(model.coef_)
coef = coef.reshape(-1,1)
# intercept = np.array(model.intercept_)
x = np.linspace(-1, 1, 5)
y = -((coef[0]*x)+model.intercept_)/coef[1]

xPredPlus = []
yPredPlus = []

xPredMinus = []
yPredMinus = []

for row in range(len(x_yTest)):

    if predData[row][0] == 1:
        xPredPlus.append(x_yTest[row][0])
        yPredPlus.append(x_yTest[row][1])
    else:
        xPredMinus.append(x_yTest[row][0])
        yPredMinus.append(x_yTest[row][1])


plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True
plt.scatter(XtrainBlue, YtrainBlue, color="blue", label="+1")
plt.scatter(XtrainGreen, YtrainGreen, color="green", label="-1")
plt.scatter(xPredPlus, yPredPlus, color="red", marker="+", label = "predict +1")
plt.scatter(xPredMinus, yPredMinus, color="yellow", marker="+", label = "predict -1")
plt.plot(x, y, color="black", linewidth = 2, label = "decision boundary")

# plt.plot(x_yTrain, predData, color="red", linewidth=3)
# plt.scatter(xPred, yPred, color="red")

# plt.plot(Xtrain, ypred, color="blue", linewidth=3) 
plt.xlabel("x_1"); 
plt.ylabel("y_2")
plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0 ) 

plt.show()

model = LinearSVC(C=100).fit(x_yTrain, resultTrain)
print("when C = 100")
print("intercept = ", model.intercept_)
print("slope = ", model.coef_)

x_yTest = np.array(x_yTest)
# print(x_yTest)
predData = np.array(model.predict(x_yTest))
predData = predData.reshape(-1,1)

coef = np.array(model.coef_)
coef = coef.reshape(-1,1)
# intercept = np.array(model.intercept_)
x = np.linspace(-1, 1, 5)
y = -((coef[0]*x)+model.intercept_)/coef[1]

xPredPlus = []
yPredPlus = []

xPredMinus = []
yPredMinus = []

for row in range(len(x_yTest)):

    if predData[row][0] == 1:
        xPredPlus.append(x_yTest[row][0])
        yPredPlus.append(x_yTest[row][1])
    else:
        xPredMinus.append(x_yTest[row][0])
        yPredMinus.append(x_yTest[row][1])


plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True
plt.scatter(XtrainBlue, YtrainBlue, color="blue", label="+1")
plt.scatter(XtrainGreen, YtrainGreen, color="green", label="-1")
plt.scatter(xPredPlus, yPredPlus, color="red", marker="+", label = "predict +1")
plt.scatter(xPredMinus, yPredMinus, color="yellow", marker="+", label = "predict -1")
plt.plot(x, y, color="black", linewidth = 2, label = "decision boundary")

# plt.plot(x_yTrain, predData, color="red", linewidth=3)
# plt.scatter(xPred, yPred, color="red")

# plt.plot(Xtrain, ypred, color="blue", linewidth=3) 
plt.xlabel("x_1"); 
plt.ylabel("y_2")
plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0 ) 

plt.show()

# color = Color

# for row in range(len(predData)):
#     xPred.append(predData.values[row][0])
#     yPred.append(predData.values[row][1])

