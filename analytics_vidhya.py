import re    # for regular expressions
import nltk  # for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train  = pd.read_csv('datasets/train_E6oV3lV.csv')
test = pd.read_csv('datasets/test_tweets_anuFYb8.csv')