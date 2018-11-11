import json as j
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


json_data = None
with open('datasets/yelp_academic_dataset_review.json') as data_file:
    # file not available due to its large size but a link to the dataset is provided below
    lines = data_file.readlines()# returns a list containing read lines
    joined_lines = "[" + ",".join(lines) + "]"

    json_data = j.loads(joined_lines)

data = pd.DataFrame(json_data)
print(data.head())

stemmer = SnowballStemmer('english') # finding the root of a word. alternatives include brown stemmer
words = stopwords.words("english") # stopwords are the common non meaningful words eg the and for in
# if item is not a stop word, replace anything that doesnt start with a letter(eg number or punctuation with a blank
#performing a split where youll be left with purely the words...then stem them re.sub substitutes
data['cleaned'] = data['text'].apply(lambda x: " ".join( [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words] ).lower() )

X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.stars, test_size=0.2)

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)), # term frequency inverse document frequency. Used to determine how important a word is in a document
                     #ngram_range gives the number of words to consider as features eg, 1,1 states that each word is a feature 1,2 instructs the algm to consider 2 or 1 word each
                     ('chi',  SelectKBest(chi2, k=10000)), # select the best 10K features. Chi2 is used to calculate dependence and inedepndence of features
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))]) # classifier
#pipline ensures the input of one is from the output of another in the specified order


model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

target_names = ['1', '2', '3', '4', '5']
print("top 10 keywords per class:")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))

print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['i hate this place']))
'''
LINK TO YELP  MOVIE-REVIEW DATASET
https://github.com/coding-maniacs/text_classification/blob/master/data/yelp_academic_dataset_review.json.zip
'''
