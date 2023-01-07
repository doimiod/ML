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
nltk.download('punkt')
nltk.download('stopwords')
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


listingData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/listings.csv')
reviewData = pd.read_csv('/Users/doimasanari/Documents/ML/final/code/reviews.csv')

# Delete the columns that are not relevant to the training.
listingData = listingData.drop(["listing_url", "scrape_id", "last_scraped", "picture_url", "host_id" ,"host_url", "host_thumbnail_url", "host_picture_url", "license", "calendar_updated", "calculated_host_listings_count_shared_rooms" ], axis = 1)

# Sort the DataFrame by the values in column listing_id
reviewData = reviewData.sort_values(by='listing_id')

def is_nonEnglish(text):
  # Check if the comments contain any non-English characters or emoji (non-ascii characters)
  if re.search(r'[^\x00-\x7F]', text):
    return True                       # True if the text contains any non-English characters
  return False

# Convert the values in the comments column to strings
reviewData['comments'] = reviewData['comments'].astype(str)

# Apply the is_notEnglish function to each row of the comments columns
filters = reviewData['comments'].apply(is_nonEnglish)

# Use the filters to create a new dataframe containing only the rows where is_nonEnglish returned False
reviewData = reviewData[filters == False]  

reviewData = reviewData[reviewData["comments"].notnull()]

# Initialize the CountVectorizer
vectorizer = CountVectorizer(stop_words="english")

# Fit the CountVectorizer to the list of strings
X = vectorizer.fit_transform(reviewData["comments"])

# Replace the text column with the resulting matrix
reviewData["comments"] = X.toarray()

print("----------------------------------------------------------------")
# print(common_words)

print("----------------------------------------------------------------")
# print(reviewData)
print(reviewData)

print(listingData)
print("----------------------------------------------------------------")
# print(reviewData)

avg_review_scores_rating = 0
avg_review_scores_accuracy = 0
avg_review_scores_cleanliness = 0
avg_review_scores_checkin = 0
avg_review_scores_communication = 0
avg_review_scores_location = 0
avg_review_scores_value = 0

# for row in range(len(listingData)):

#     if listingData[row]["review_scores_rating"] == None:
#         listingData = listingData.drop(listingData.index(row))

listingData = listingData[listingData["review_scores_rating"].notnull()]
print(listingData)

y = listingData.loc[:, "review_scores_rating":"review_scores_value"].values

print("___________________________________________________________")
# print(y)



















# Define a function to detect the language of a sentence
# def is_english(comments):
#     try:
#         return detect(comments) == 'en'
#     except:
#         return False

# # Filter the DataFrame to only include rows with English sentences
# reviewData = reviewData[reviewData['comments'].apply(is_english)]


# # Tokenize the text
# tokens = [nltk.word_tokenize(text) for text in reviewData["comments"]]

# # Remove stop words
# filtered_tokens = [[word for word in token_list if word.lower() not in stopwords.words("english")] for token_list in tokens]

# # Count the frequency of each word
# word_counts = {}
# for token_list in filtered_tokens:
#     for word in token_list:
#         if word in word_counts:
#             word_counts[word] += 1
#         else:
#             word_counts[word] = 1

# # Get the list of most common words
# common_words = sorted(word_counts, key=word_counts.get, reverse=True)

# # Delete highly repetitive words from the text data
# threshold = 5  # Set the threshold for what constitutes a highly repetitive word
# for word in common_words:
#     if word_counts[word] > threshold:
#         for i in range(len(filtered_tokens)):
#             filtered_tokens[i] = [token for token in filtered_tokens[i] if token != word]

# reviewData["comments"] = [" ".join(token_list) for token_list in filtered_tokens]