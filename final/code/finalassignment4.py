from cProfile import label
from re import X
# from statistics import LinearRegression
from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string


from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.metrics  import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import PolynomialFeatures
# from yellowbrick.classifier import ROCAUC
from langdetect import detect
import re
from sklearn.feature_extraction.text import CountVectorizer

from array import array


"""#### text clean up function"""

def cleanup_texts(data, column):

  # print(data)
  
  data[column] = data[column].astype(str)
  
  def is_nonEnglish(text):
    # Check if the comments contain any non-English characters or emoji (non-ascii characters)
    if re.search(r'[^\x00-\x7F]', text):
      return True                       # True if the text contains any non-English characters
    return False
  
  # REmove non-english texts
  filters = data[column].apply(is_nonEnglish)
  data = data[filters == False]

  # print("----------------------------------------------------------------")
  # print(data) 

  # Stop words list in English
  stop_words = stopwords.words('english')

  def manipulate_texts(text):
    text = "".join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text 

 # Remove stop words and punctuations from the comments and make text lower case 
  data[column] = data[column].apply(lambda x: manipulate_texts(x)) 
  # data.dropna(axis=0, inplace=True)
  return data

"""#### Preprocessing in reviews.scv"""

reviewData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/reviews.csv')

# Sort the DataFrame by the values in column listing_id
# reviewData = reviewData.sort_values(by='listing_id')

# Remove the rows that have null
reviewData = reviewData[reviewData["comments"].notnull()]

reviewData = cleanup_texts(reviewData, "comments")

reviewData.to_csv('cleanedup_review.csv', index=False)

listingData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')
# Delete the row where listingData is null
listingData = listingData[listingData["review_scores_rating"].notnull()]

listingData.to_csv('reviewScoresOk_listing.csv', index=False)

"""#### super host"""

listingData = pd.read_csv('reviewScoresOk_listing.csv')

# Extract the x and y values from the data
x = listingData['review_scores_rating']
y = listingData['number_of_reviews']

# Extract the color values from the data
listingData = listingData[listingData["host_is_superhost"].notnull()]
colors = listingData['host_is_superhost']

# Create an empty list to store the color values
colors_list = []
label = []

# Iterate over the color values
for color in colors:
    # If the color value is "t", append "red" to the list
    if color == "t":
        colors_list.append("red")
    # If the color value is "f", append "blue" to the list
    elif color == "f":
        colors_list.append("blue")
    else: colors_list.append("blue")

colors_list.append("red")

plt.scatter(x, y, color=colors_list, label = "superhost")
plt.xlabel("Review scores")
plt.ylabel("The number of reviews")
plt.title("Reviews and the number of reviews")
plt.legend()
plt.show()

"""#### feature selection response rate


"""

import matplotlib.pyplot as plt

listingData = pd.read_csv('reviewScoresOk_listing.csv')
listingData = listingData[listingData["review_scores_communication"].notnull()]

# Extract data from columns
# x = listingData["review_scores_communication"]
# y = listingData['host_response_rate']

avg_rate = 0
# Get average
for row in range(len(listingData["review_scores_communication"])):
  if row in listingData["review_scores_communication"]:
    avg_rate = avg_rate + listingData["review_scores_communication"][row]

avg_rate = avg_rate / len(listingData["review_scores_communication"])
print(avg_rate)

def to_float(x):
    if pd.isnull(x):
        return 0
    else:
        return float(x.replace("%", ""))

# Remove $
listingData['host_response_rate'] = listingData['host_response_rate'].apply(to_float)

# Plot the data
plt.figure()
plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True

# Parameter for legend
legend_one = 0
legend_two = 0

# If the rate > average_rate, plot red colour, otherwise blue
for row in range(len(listingData["review_scores_communication"])):
  if row in listingData["review_scores_communication"]:
    # If the rate > average_rate, plot red colour, otherwise blue
    if listingData["review_scores_communication"][row] > avg_rate:
      plt.scatter(listingData["review_scores_communication"][row], listingData['host_response_rate'][row], color="red")
      if legend_one == 0:
        plt.scatter(listingData["review_scores_communication"][row], listingData['host_response_rate'][row], color="red", label = "high scores")
        plt.legend()
        legend_one = 1
    elif listingData["review_scores_communication"][row] <= avg_rate:
      plt.scatter(listingData["review_scores_communication"][row], listingData['host_response_rate'][row], color="blue")
      if legend_two == 0:
        plt.scatter(listingData["review_scores_communication"][row], listingData['host_response_rate'][row], color="blue", label = "low scores")
        plt.legend()
        legend_two = 1
      # colors_listll.append("blue")

plt.xlabel("Review scores communication")
plt.ylabel("Response Rate")
plt.title("Reviews and the response Rate")
# plt.legend()
plt.show()

"""#### location analysis"""

listingData = pd.read_csv('reviewScoresOk_listing.csv')
listingData = listingData[listingData["review_scores_location"].notnull()]

print(listingData["review_scores_location"])

avg_rate = 0

# Get average
for row in range(len(listingData["review_scores_location"])):
  if row in listingData["review_scores_location"]:
    avg_rate = avg_rate + listingData["review_scores_location"][row]

avg_rate = avg_rate / len(listingData["review_scores_location"])
print(avg_rate)

# x = listingData["latitude"]
# y = listingData['longitude']
# z = listingData["review_scores_location"]

# Plot the data
plt.figure()
plt.rc('font', size=18)
plt.rcParams["figure.constrained_layout.use"] = True

# Parameter for legend
legend_one = 0
legend_two = 0

# If the rate > average_rate, plot red colour, otherwise blue
for row in range(len(listingData["review_scores_location"])):
  if row in listingData["review_scores_location"]:
    # If the rate > average_rate, plot red colour, otherwise blue
    if listingData["review_scores_location"][row] > avg_rate:
      plt.scatter(listingData["latitude"][row], listingData['longitude'][row], color="red")
      if legend_one == 0:
        plt.scatter(listingData["latitude"][row], listingData['longitude'][row], color="red", label = "high rate")
        plt.legend()
        legend_one = 1
    elif listingData["review_scores_location"][row] <= avg_rate:
      plt.scatter(listingData["latitude"][row], listingData['longitude'][row], color="blue")
      if legend_two == 0:
        plt.scatter(listingData["latitude"][row], listingData['longitude'][row], color="blue", label = "low rate")
        plt.legend()
        legend_two = 1
      # colors_listll.append("blue")

plt.xlabel("latitude")
plt.ylabel("longitude")
plt.title("Reviews and lat and lon")
plt.show()

"""#### price analysis"""

import matplotlib.pyplot as plt

x = listingData["review_scores_value"]
y = listingData['price']

# Convert x data to numeric
y = [float(y.replace("$", "").replace(",", "")) for y in y]

av_price = 0

for row in range(len(y)):
    if y[row] > 75000:
      y[row] = 0
    av_price = av_price + y[row]

av_price = av_price / len(y)
print(av_price)

# Colur array
colors_list = []

for row in range(len(y)):
    # If price is higher than avg colour is red
    if y[row] > av_price:
        colors_list.append("red")
    # If price is lower than avg colour is blue
    else: colors_list.append("blue")

plt.scatter(x, y)
plt.scatter(x, y, color=colors_list, label = "low price")      
plt.xlabel("Score rate")
plt.ylabel("Price")
plt.title("Price and Rate")
plt.legend()
plt.show()

"""#### Preprocessing and feature selection in listings.scv """

listingData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')
temp = pd.read_csv('/content/drive/MyDrive/final/code/listings.csv')

print(listingData)

# Delete the columns that does not look important to the training.
listingData = listingData.drop(["listing_url", "scrape_id", "last_scraped", "picture_url","host_id" ,"host_url", "host_name", "host_neighbourhood", "host_thumbnail_url", 
                                "host_picture_url", "host_verifications", "neighbourhood", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "bathrooms", "calendar_last_scraped", 
                                "first_review","last_review","license", "calendar_updated", "calculated_host_listings_count_shared_rooms", "first_review", "last_review", "host_location",
                                "host_response_time", "price","host_is_superhost", "host_is_superhost", "host_response_rate","host_acceptance_rate", "host_since", "source", "longitude", "latitude"], axis = 1)

# print(listingData)

listingData['host_identity_verified'] = listingData['host_identity_verified'].map({"t": 1, "f": 0})
listingData['has_availability'] = listingData['has_availability'].map({"t": 1, "f": 0})
listingData['instant_bookable'] = listingData['instant_bookable'].map({"t": 1, "f": 0})
listingData['host_has_profile_pic'] = listingData['host_has_profile_pic'].map({"t": 1, "f": 0})



print("___________________________________________________________")
print(listingData)

# Choose the column that has sentences
text_columns = ["name", "description","neighborhood_overview", "host_about", ]

# listingData[text_columns] = listingData[text_columns].astype(str)

# Remove stop words and punctuations from the commentsmake and make text lower case 
for column in text_columns:
  listingData = cleanup_texts(listingData, column)
  # listingData[column] = listingData[column].apply(lambda x: cleanup_texts(x))

print("___________________________________________________________")
print(text_columns)
print(listingData)
print("___________________________________________________________")
# listingData = listingData[listingData["id"].notnull()]
# listingData.dropna(axis=0, inplace=True)
print(listingData)
# print(vectorised_texts)
# print(len(vectorised_texts))

listingData.to_csv('cleanedup_listing.csv', index=False)

"""#### add reviews to listings.csv"""

reviewData = pd.read_csv('cleanedup_review.csv')
# print(reviewData)
reviewData["listing_id"] = reviewData["listing_id"].astype(int)

listingData = pd.read_csv('cleanedup_listing.csv')
listingData["id"] = listingData["id"].astype(int)
print(listingData)

for row in range(len(listingData)):
  
  temp_array = []
  # print(listingData["id"][row])
  if row in listingData["id"]:
    print("___________________________________________________________________________________________________________________________________________________________")
    # print(listingData["id"][row])

    # Look for for rows that contain the same listing_id
    filters = reviewData["listing_id"].isin([listingData["id"][row]])
    result = reviewData[filters]
    print(result)
    temp_array = result["comments"].tolist()
    print("___________________________________________________________________________________________________________________________________________________________")
    # print(temp_array[119])
    # There is float so change it to strings
    temp_array = [str(x) for x in temp_array]
    temp_array = " ".join(temp_array)
    
    listingData.at[row, "reviews"] = temp_array

# print(listingData)

# listingData = listingData[listingData["reviews"].notnull()]
# listingData = listingData.dropna()
# listingData = listingData.drop(listingData.index[-1])
print("___________________________________________________________________________________________________________________________________________________________")
print(listingData)

# print(listingData)
listingData.to_csv('merged_listing.csv', index=False)

"""#### featuring"""

listingData = pd.read_csv('merged_listing.csv')
print("___________________________________________________________")
print(listingData)

# listingData = listingData.drop(["id","review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"  ], axis = 1)
listingData = listingData.drop(["id"], axis = 1)

# listingData = listingData.drop(["id"], axis = 1)

# Drop rows with any NaN values
listingData.to_csv('dropped_rating_listing.csv', index=False)
listingData = pd.read_csv('dropped_rating_listing.csv')
print("___________________________________________________________")
# print(listingData)

# Drop the row if the column has NaN
for column in listingData.columns:
  print("___________________________________________________________")
  print(column)
  print("___________________________________________________________")
  listingData = listingData.dropna(subset=[column])
  print(listingData)

listingData.to_csv('clean_completed_listing.csv', index=False)
listingData = pd.read_csv('clean_completed_listing.csv')
print("___________________________________________________________")
print(listingData)

# listingData = map(multiplyBy10, listingData["review_scores_rating"])
# listingData = map(multiplyBy10, listingData['review_scores_accuracy'])
# listingData = map(multiplyBy10, listingData['review_scores_cleanliness'])
# listingData = map(multiplyBy10, listingData['review_scores_checkin'])
# listingData = map(multiplyBy10, listingData['review_scores_communication'])
# listingData = map(multiplyBy10, listingData['review_scores_location'])
# listingData = map(multiplyBy10, listingData['review_scores_value'])

y = listingData.loc[:, "review_scores_rating":"review_scores_value"].values

# y = map(multiplyBy10, y)

# multiply all numbers in y by 100
y = [x * 100 for x in y]
print(y)
print("______________________________________________")
# print(listingData)

"""#### vectorarize"""

# why its not working??
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LinearRegression
# from scipy.sparse import hstack


# # Select the columns that contain text data
# text_columns = listingData.select_dtypes(include=['object'])
# print(text_columns.columns)

# # Initialize the CountVectorizer
# vectorizer = CountVectorizer()

# # Convert the text data to numerical vectors
# vectors = vectorizer.fit_transform(text_columns)

# vectors_array = []

# # Select the numerical columns from the input data
# numerical_columns = listingData.select_dtypes(exclude=['object'])

# print(pd.DataFrame(vectors.toarray()))

# X = hstack(pd.DataFrame(vectors.toarray()))

# X = hstack((X, numerical_columns))

# xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

"""#### vectorarizer"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack


# Select the columns that contain text data
text_columns = listingData.select_dtypes(include=['object'])
print(text_columns.columns)

vectors_array = []

for column in text_columns.columns:
  # Convert the text data into a matrix of token counts
  vectorizer = CountVectorizer()
  # print("__________________________________")
  print(listingData[column])
  vector = vectorizer.fit_transform(listingData[column])
  vectors_array.append(vector)

# text_matrix = vectorizer.fit_transform(text_columns)

# numeric_columns = []
# for column in listingData.columns:
#     if column not in text_columns:
#         numeric_columns.append(column)

numerical_columns = listingData.select_dtypes(exclude=['object'])

# print(numerical_columns)

# Convert the numeric data into dummy variables
numeric_dummies = pd.get_dummies(numerical_columns)

print("_____________________________________")
print(vectors_array)

# text_df = pd.DataFrame(text_matrix.todense())
# vectors_array = pd.to_numeric(vectors_array)

X = hstack(vectors_array)
X = hstack((X, numeric_dummies))
# print(text_matrix)

# Concatenate the dummy variables with the matrix of token counts
# X = pd.concat([text_df, numeric_dummies], axis=1)   
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
# print("______________________________")
# print(X)

"""#### train a model"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
LinearRegression,
model = LinearRegression()

# train data
model.fit(xTrain, yTrain)             
# get a slope here
print("slope = ", model.coef_)
# get an intercept here                       
print("intercept = ", model.intercept_)  

# Make predictions on the test set
y_pred = model.predict(xTest)

# Compute the MSE
mse = mean_squared_error(yTest, y_pred)
print("Mean Squared Error:", mse)

# Compute the R^2 score
r2 = r2_score(yTest, y_pred)
print("R^2 score:", r2)

# print(confusion_matrix(yTest,y_pred))

print("________________________________________________________________________________________________________________________________________________________________________________")

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(xTrain, yTrain)
# y_pred = model.predict(xTest)
# accuracy = accuracy_score(yTest, y_pred)
# print("Accuracy:", accuracy)

print("________________________________________________________________________________________________________________________________________________________________________________")

# from sklearn.svm import SVC
# model = SVC()
# model.fit(xTrain, yTrain)
# y_pred = model.predict(xTest)
# accuracy = accuracy_score(yTest, y_pred)
# print("Accuracy:", accuracy)

print("________________________________________________________________________________________________________________________________________________________________________________")

from sklearn.tree import DecisionTreeClassifier
model.fit(xTrain, yTrain)
y_pred = model.predict(xTest)
accuracy = accuracy_score(yTest, y_pred)
print("Accuracy:", accuracy)

"""#### trash"""

# def is_nonEnglish(text):
#   # Check if the comments contain any non-English characters or emoji (non-ascii characters)
#   if re.search(r'[^\x00-\x7F]', text):
#     return True                       # True if the text contains any non-English characters
#   return False

# # Convert the values in the comments column to strings
# reviewData['comments'] = reviewData['comments'].astype(str)

# # Apply the is_notEnglish function to each row of the comments columns and delete the row if it has non-english words
# filters = reviewData['comments'].apply(is_nonEnglish)
# reviewData = reviewData[filters == False]  

# Remove the rows that have null

# print("----------------------------------------------------------------")
# print(reviewData)

# # Stop words list in English
# stop_words = stopwords.words('english')

# def cleanup_texts(text):
#     text = "".join([word for word in text if word not in string.punctuation])
#     text = text.lower()
#     text = " ".join([word for word in text.split() if word not in stop_words])
#     return text

# # Remove stop words and punctuations from the commentsmake and make text lower case 
# reviewData['comments'] = reviewData['comments'].apply(lambda x: cleanup_texts(x))



# print("----------------------------------------------------------------")
# for row in range(len(reviewData)):
#   if row in reviewData["listing_id"] and reviewData["listing_id"][row] == 44077: 
#     print(reviewData["comments"][row])
# print(new_df)


# j = 0
# # listingData = listingData.assign(new_column=[])
# # listingData["new_column"] = []
# # print(listingData["new_column"])
# # listingData.dropna(inplace=True)

# review_array = [[] for _ in range(1)]
# for row in range(len(listingData)):
  
#   temp_array = []
#   # print(listingData["id"][row])
#   if row in listingData["id"]:
#     print("___________________________________________________________________________________________________________________________________________________________")
#     print(listingData["id"][row])
#     # for row2 in range(len(reviewData)):
#     #   if row2 in reviewData["listing_id"] and listingData["id"][row] == reviewData["listing_id"][row2]:
#     #     # print(reviewData["comments"][row2])
#     #     temp_array.append(reviewData["comments"][row2])
#     #     reviewData.dropna(inplace=True)

#     # Look for for rows that contain the same listing_id
#     filters = reviewData["listing_id"].isin([listingData["id"][row]])
#     result = reviewData[filters]
#     # print(result)
#     temp_array = result["comments"].tolist()
#     # yyy = row
#     # listingData.loc[row, "comments"] = temp_array[0]
#     temp_array = " ".join(temp_array)
#     # listingData[[row, "comments"]] = listingData.apply(lambda r: temp_array, axis=1, result_type="expand")
#     # listingData[["comments"]] = listingData.apply(lambda r: temp_array, axis=1, result_type="expand")
#     # listingData.at[row, "comments"] = temp_array[0]
#     # twmp_array = array("u", temp_array)
#     # temp_array = temp_array.toarray()
#     print(temp_array)
#     # review_array.append(temp_array)
#     # listingData.at[row, "id"] = None
#     # listingData.at[row, "id"] = temp_array[0]
#     listingData.at[row, "new_column"] = temp_array
#     # listingData.loc[row, "id"] = temp_array
#     # listingData[row]["new_column"].append(review_array)
#     # print()
#         # j += 1

#   # print(reviewData["listing_id"][j])
# print(len(review_array))
# # print(yyy)
#   # print(review_array)
   
#   # listingData.loc[row, "new_column"] = review_array
# # listingData.loc["new_column"] = review_array
#   # print("jfiejidnrvgigitm")
#   # print(listingData["new_column"])

# print(listingData)




# import math
# listingData = pd.read_csv('reviewScoresOk_listing.csv')
# resCount = 0
# resRateCount = 0
# acceptanceCount = 0
# print(listingData["host_response_time"])
# # print(listingData["host_response_time"][15])
# print("____________________________")
# # Count the number of N/A and change text to number. if its N/A its 0 otherwise 1
# for row in range(len(listingData["host_response_time"])):
  
#   if row in listingData["host_response_time"]:
#     resCount += 1
#     listingData["host_response_time"][row] = 1
#     # listingData["host_response_time"][row] = 0
#   else if listingData["host_response_time"].isna().any():
#     listingData["host_response_time"][row] == 0 
#     print(listingData["host_response_time"][row])
#   if row in listingData["host_response_rate"] and listingData["host_response_rate"][row] == "NaN": listingData["host_response_rate"][row] = 0
#   else :
#     resRateCount += 1
#     listingData["host_response_rate"][row] = 1
#   if row in listingData["host_acceptance_rate"] and listingData["host_acceptance_rate"][row] == "NaN": listingData["host_acceptance_rate"][row] = 0
#   else :
#     acceptanceCount += 1
#     listingData["host_acceptance_rate"][row] = 1

# print("____________________________")
# print(resCount, resRateCount,acceptanceCount)

# def to_int(cell):
#     if pd.isna(cell):
#         return 0
#     return 1

# listingData["host_response_time"] = listingData["host_response_time"].apply(to_int)  


# print(listingData["host_response_time"][15])


# print(listingData["host_response_rate"][14])


# plt.figure()
# plt.rc('font', size=18)
# plt.rcParams["figure.constrained_layout.use"] = True
# plt.scatter(x1RealPlus, x2RealPlus, color="blue", label="actual +1")
# plt.scatter(x1RealMinus, x2RealMinus, color="green", label="actual -1")
# plt.scatter(x1PredPlus, x2PredPlus, color="red", marker="+", label = "predicted +1")
# plt.scatter(x1PredMinus, x2PredMinus, color="yellow", marker="+", label = "predicted -1")
# plt.xlabel("x_1")
# plt.ylabel("x_2")
# plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

# plt.show()