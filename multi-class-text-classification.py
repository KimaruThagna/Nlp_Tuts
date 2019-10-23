from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
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