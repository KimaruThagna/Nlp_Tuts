from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('datasets/Consumer_Complaints.csv')
#print(f'Consumer complaints dataset view\n{df.head()}')

relevant_col = ['Product', 'Consumer complaint narrative']
df = df[relevant_col]

df = df[pd.notnull(df['Consumer complaint narrative'])] # filer out missing values by getting rows where nconsumer narrative is not null
df.columns = ['Product', 'Consumer_complaint_narrative']
df['category_id'] = df['Product'].factorize()[0] # convert to categorical variables and obtain label. Label and unique value is returned
category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values) # convert list of lists to dictionary
id_to_category = dict(category_id_df[['category_id', 'Product']].values)
#print(f'Consumer complaints {df.head()}')

#visualize class distribution
sns.countplot(data=df, x='Product')
plt.title("Class Distribution")
plt.xticks(rotation=90)
plt.show()

# feature extraction
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.category_id
print(f' Extracted features {features.shape}')

# find correlated terms
N = 2
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])# return indices of indirectly sorted array
  feature_names = np.array(tfidf.get_feature_names())[indices] # obtain names
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print(f'# {Product}')
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# ML classification
X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
#Test this out
print(df[df['Consumer_complaint_narrative'] == "This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])