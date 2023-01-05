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
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
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

listingData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')
reviewData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/reviews.csv')

# Delete the columns that are not relevant to the training.
listingData = listingData.drop(["listing_url", "scrape_id", "last_scraped", "picture_url", "host_id" ,"host_url", "host_thumbnail_url", "host_picture_url", ], axis = 1)

# Sort the DataFrame by the values in column listing_id
reviewData = reviewData.sort_values(by='listing_id')

def is_notEnglish(text):
  # Check if the comments contain any non-English characters or emoji (non-ascii characters)
  if re.search(r'[^\x00-\x7F]', text):
    return True                       # True if the text contains any non-English characters
  return False

# Convert the values in the comments column to strings
reviewData['comments'] = reviewData['comments'].astype(str)

# Apply the is_notEnglish function to each row of the comments columns
filters = reviewData['comments'].apply(is_notEnglish)

# Use the filters to create a new dataframe containing only the rows where is_english returned False
reviewData = reviewData[filters == False]  

# print(listingData)
print("----------------------------------------------------------------")
# print(reviewData)
print(reviewData)



















# Define a function to detect the language of a sentence
# def is_english(comments):
#     try:
#         return detect(comments) == 'en'
#     except:
#         return False

# # Filter the DataFrame to only include rows with English sentences
# reviewData = reviewData[reviewData['comments'].apply(is_english)]