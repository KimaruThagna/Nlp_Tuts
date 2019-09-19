import re    # for regular expressions
import nltk  # for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# racist/ sexist tweets are labelled with 1 else, 0
train  = pd.read_csv('datasets/train_E6oV3lV.csv')
test = pd.read_csv('datasets/test_tweets_anuFYb8.csv')
train.set_index("id")
# check out some tweets from both classes
print(f' NON RACIST?SEXIST Tweets>>>>{train[train["label"] == 0].head()}')
print(f' Racist/sexist tweets {train[train["label"] == 1].head()}')
# inspect label distribution
print(train["label"].value_counts())

# visualize label distribution
sns.countplot(train["label"])
plt.title("Label Distribution")
plt.show()