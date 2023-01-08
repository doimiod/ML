# -*- coding: utf-8 -*-
"""finalAssignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Mz3zBXu5EiSi9jU5SCsaOGAgz8EgytCN

# Fianl Project

#### install *langdetect*
"""

# !pip install langdetect

"""#### Connecting Google Drive"""

# from google.colab import drive
# drive.mount('/content/drive')

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
from sklearn.metrics import confusion_matrix, classification_report

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
# nltk.download('stopwords')
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
reviewData = reviewData.sort_values(by='listing_id')

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

reviewData = reviewData[reviewData["comments"].notnull()]

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

reviewData = cleanup_texts(reviewData, "comments")

# print("----------------------------------------------------------------")
# for row in range(len(reviewData)):
#   if row in reviewData["listing_id"] and reviewData["listing_id"][row] == 44077: 
#     print(reviewData["comments"][row])
# print(new_df)

"""#### Preprocessing in listings.scv"""

listingData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')
temp = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')


# print(listingData)

# Delete the columns that are not relevant to the training.
listingData = listingData.drop(["listing_url", "scrape_id", "last_scraped", "picture_url", "host_id" ,"host_url", "host_name", "host_neighbourhood", "host_thumbnail_url", "host_picture_url",
                                "host_verifications", "neighbourhood", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "bathrooms", "calendar_last_scraped", "first_review", "last_review","license", "calendar_updated", "calculated_host_listings_count_shared_rooms" ], axis = 1)

# print(listingData)

listingData = listingData[listingData["review_scores_rating"].notnull()]
print("___________________________________________________________")
print(listingData)

# Select the text columns
text_columns = listingData.select_dtypes(include='object').columns

# listingData[text_columns] = listingData[text_columns].astype(str)

# Remove stop words and punctuations from the commentsmake and make text lower case 

for column in text_columns:
  listingData = cleanup_texts(listingData, column)
  # listingData[column] = listingData[column].apply(lambda x: cleanup_texts(x))

print("___________________________________________________________")
# print(vectorised_texts)
print(text_columns)
print(listingData)
print("___________________________________________________________")
# listingData = listingData[listingData["id"].notnull()]
# listingData.dropna(axis=0, inplace=True)
print(listingData)
# print(vectorised_texts)
# print(len(vectorised_texts))

"""#### add reviews to listings.csv"""

j = 0
# listingData = listingData.assign(new_column=[])
# listingData["new_column"] = []
# print(listingData["new_column"])
# listingData.dropna(inplace=True)

review_array = [[] for _ in range(1)]
for row in range(len(listingData)):
  
  temp_array = []
  # print(listingData["id"][row])
  if row in listingData["id"]:
    print("___________________________________________________________________________________________________________________________________________________________")
    print(listingData["id"][row])
    # for row2 in range(len(reviewData)):
    #   if row2 in reviewData["listing_id"] and listingData["id"][row] == reviewData["listing_id"][row2]:
    #     # print(reviewData["comments"][row2])
    #     temp_array.append(reviewData["comments"][row2])
    #     reviewData.dropna(inplace=True)

    # Look for for rows that contain the same listing_id
    filters = reviewData["listing_id"].isin([listingData["id"][row]])
    result = reviewData[filters]
    # print(result)
    temp_array = result["comments"].tolist()
    # yyy = row
    # listingData.loc[row, "comments"] = temp_array[0]
    temp_array = " ".join(temp_array)
    # listingData[[row, "comments"]] = listingData.apply(lambda r: temp_array, axis=1, result_type="expand")
    # listingData[["comments"]] = listingData.apply(lambda r: temp_array, axis=1, result_type="expand")
    # listingData.at[row, "comments"] = temp_array[0]
    # twmp_array = array("u", temp_array)
    # temp_array = temp_array.toarray()
    print(temp_array)
    # review_array.append(temp_array)
    # listingData.at[row, "id"] = None
    # listingData.at[row, "id"] = temp_array[0]
    listingData.at[row, "new_column"] = temp_array
    # listingData.loc[row, "id"] = temp_array
    # listingData[row]["new_column"].append(review_array)
    # print()
        # j += 1

  # print(reviewData["listing_id"][j])
print(len(review_array))
# print(yyy)
  # print(review_array)
   
  # listingData.loc[row, "new_column"] = review_array
# listingData.loc["new_column"] = review_array
  # print("jfiejidnrvgigitm")
  # print(listingData["new_column"])

print(listingData)

# # Convert the text columns to a matrix of token counts
# vectorizer = CountVectorizer()
# for column in text_columns:
#   listingData[column] = listingData[column].astype(str)
#   print(column)
#   vectorised_texts = vectorizer.fit_transform(listingData[column])
#   vectorised_texts = vectorised_texts.toarray()
# # vectorised_texts = vectorised_texts.reshape(-1, len(listingData['host_about']))

# # Replace the text columns with the resulting matrix
# # listingData[text_columns] = vectorised_texts

# print(vectorised_texts)
# print(len(listingData))
# # print(listingData[text_columns])

"""#### featuring"""

# Move the columns "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value" to the end of the columns
listingData = listingData.drop(["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"  ], axis = 1)
listingData[["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"]] = temp[["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"]]

print("___________________________________________________________")
print(listingData)

y = listingData.loc[:, "review_scores_rating":"review_scores_value"].values

print("___________________________________________________________")
print(len(y))
print("___________________________________________________________")
print(y[0][0])
print(y[2][2])

print("___________________________________________________________")

print(len(listingData["new_column"]))