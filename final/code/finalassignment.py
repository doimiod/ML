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

 # Remove stop words and punctuations from the commentsmake and make text lower case 
  data[column] = data[column].apply(lambda x: manipulate_texts(x))   
  return data

"""#### Preprocessing in reviews.scv"""

reviewData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/reviews.csv')

# Sort the DataFrame by the values in column listing_id
reviewData = reviewData.sort_values(by='listing_id') 

# Remove the rows that have null
reviewData = reviewData[reviewData["comments"].notnull()]

reviewData = cleanup_texts(reviewData, "comments")

# print("----------------------------------------------------------------")
print(reviewData)
# print(new_df)

"""#### Preprocessing in listings.scv"""

listingData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')
temp = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')


# print(listingData)

# Delete the columns that are not relevant to the training.
listingData = listingData.drop(["listing_url", "scrape_id", "last_scraped", "picture_url", "host_id" ,"host_url", "host_name", "host_neighbourhood", "host_thumbnail_url", "host_picture_url",
                                "host_verifications", "neighbourhood", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "bathrooms", "calendar_last_scraped", "first_review", "last_review","license", "calendar_updated", "calculated_host_listings_count_shared_rooms" ], axis = 1)

listingData = listingData[listingData["review_scores_rating"].notnull()]
print("___________________________________________________________")
print(listingData)

# Move the columns "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value" to the end of the columns
listingData = listingData.drop(["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"  ], axis = 1)
listingData[["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"]] = temp[["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"]]

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
# print(vectorised_texts)
# print(len(vectorised_texts))




# Convert the text columns to a matrix of token counts
# vectorizer = CountVectorizer()
# listingData['host_about'] = listingData['host_about'].astype(str)
# vectorised_texts = vectorizer.fit_transform(listingData['host_about'])
# vectorised_texts = vectorised_texts.toarray()
# # vectorised_texts = vectorised_texts.reshape(-1, len(listingData['host_about']))

# # Replace the text columns with the resulting matrix
# listingData['host_about'] = vectorised_texts



"""#### featuring"""

y = listingData.loc[:, "review_scores_rating":"review_scores_value"].values

print("___________________________________________________________")
print(y)
print("___________________________________________________________")
print(y[0][0])
print(y[2][2])