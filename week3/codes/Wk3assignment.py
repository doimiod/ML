import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/doimasanari/Desktop/#id:23--23-23.csv')

# y = df.iloc[:,2]
# y = np.array(y)
# print(y)
x1 = df.iloc[0:,-1]
print(x1)
x2 = df.iloc[:,1]
print(x2)



fig = plt.figure()
plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, y)

plt.show()