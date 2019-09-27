import re    # for regular expressions
import nltk  # for text manipulation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from tqdm import tqdm
#tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import LabeledSentence

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

# check distribution in terms of length of tweets in the train and test set
length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
print(length_train)
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.title("Histogram on Word Length Distribution")
plt.show()

# combine train and test for data processing
combined = train.append(test, ignore_index=True)
print(combined.shape)

# data cleaning function to remove any pattern
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt) # substitute pattern with blank
    return input_txt

'''
Tweet cleaning steps
1. Remove twitter handles since they dont add to the sentiment of the tweet(denoted by @user due to privacy)
2. Remove punctuation numbers and special characters
3. Remove short words of size 3 or less. They dont have much lexical meaning.
(Maybe, will run model with and without this step to view difference)
4. Normalize text data. eg stemming
'''
combined['tidy_tweet'] = np.vectorize(remove_pattern)(combined['tweet'], "@[\w]*")
combined['tidy_tweet'] = combined['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") # replace everything except letters and #
combined['tidy_tweet'] = combined['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# Normalization
tokenized_tweet = combined['tidy_tweet'].apply(lambda x: x.split()) # tokenizing
print(tokenized_tweet.head())

stemmer = PorterStemmer()
lemmetizer = WordNetLemmatizer() # will consider in terms of accuracy of final model

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
# stitch items back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combined['tidy_tweet'] = tokenized_tweet
print(combined.head())

'''
STORY GENERATION AND VISUALIZATION OF CLEANED TWEEETS
'''
# all words
all_words = ' '.join([text for text in combined['tidy_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("WORDCLOUD FOR NON RACIST/SEXIST TWEETS")
plt.show()


# racist/sexist tweets
negative_words = ' '.join([text for text in combined['tidy_tweet'][combined['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("WORDCLOUD FOR RACIST/SEXIST TWEETS")
plt.show()

# impact of #(hastags) on tweets
#prepare a list of # for each class
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combined['tidy_tweet'][combined['label'] == 0])
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combined['tidy_tweet'][combined['label'] == 1])
# unnesting list

HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

# plot top n hashtags for positive class
freq_distribution = nltk.FreqDist(HT_regular) # key value pair
freq_dataframe = pd.DataFrame({'Hashtag': list(freq_distribution.keys()),'Count': list(freq_distribution.values())})
# selecting top 20 most frequent hashtags
data = freq_dataframe.nlargest(columns="Count", n = 20)

plt.figure(figsize=(16,5))
ax = sns.barplot(data=data, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Frequency Distribution for Non Racist/Sexist tweets")
plt.show()
# frequency distribution plot for racist/sexist tweets

distribution = nltk.FreqDist(HT_negative)
df = pd.DataFrame({'Hashtag': list(distribution.keys()), 'Count': list(distribution.values())})
# selecting top 20 most frequent hashtags
df = df.nlargest(columns="Count", n = 20)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=df, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Frequency Distribution for Racist/Sexist tweets")
plt.show()

'''
NLP Feature Engineering
'''
# bag of words
'''
A DXN matrix where D is the number of documents/sentences and N is the number of unique tokens from all the documents or sentences
Each row i contains frequency of tokens in document i
'''
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combined['tidy_tweet'])
print(bow)

print(f' bag of words shape{bow.shape}')

# TF-IDF
'''
Frequency based method that takes into account terms appearing in the documents and also in the whole corpus
Penalizes common words by assigning lower weights and assigns larger weights to terms that are rare in the whole corpus but appear
favourably in a number of documents
'''

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combined['tidy_tweet'])
print(f'TF-IDF shape{tfidf.shape}')

# Word2Vec
# Better than BOW of TF-IDF due to dimensionality reduction and can capture context
# train word2Vec model on our corpus

tokenized_tweet = combined['tidy_tweet'].apply(lambda x: x.split()) # tokenizing
model_w2v = gensim.models.Word2Vec(tokenized_tweet,
            size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)
# finding most similar word to a given word is by the cosine similarity method of the word vectors
model_w2v.train(tokenized_tweet, total_examples= len(combined['tidy_tweet']), epochs=20)
print(f'What the model looks like {model_w2v}')
print(f'Words from corpus similar to dinner {model_w2v.wv.most_similar(positive="dinner")}')
print(f'Words from corpus similar to trump {model_w2v.wv.most_similar(positive="trump")}')
print(f'Vector representation of a word like food {model_w2v["food"]}')

#use the below function to create a vector for each tweet
# by taking the average of the vectors of the words present in the tweet.
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count # find mean increamentally
    return vec

wordvec_arrays = np.zeros((len(tokenized_tweet), 200))
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    wordvec_df = pd.DataFrame(wordvec_arrays)

# Doc2Vec. Requires each tweet to be tagged to allow implementation
# Label each Tokenized Tweet
def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output
labeled_tweets = add_label(tokenized_tweet) # label all the tweets
print(labeled_tweets[:10])

model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model
                                  dm_mean=1, # dm = 1 for using mean of the context word vectors                                  size=200, # no. of desired features
                                  window=5, # width of the context window
                                  negative=7, # if > 0 then negative sampling will be used                                 min_count=5, # Ignores all words with total frequency lower than 2.
                                  workers=3, # no. of cores
                                  alpha=0.1, # learning rate
                                  seed = 23)
model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
model_d2v.train(labeled_tweets, total_examples= len(combined['tidy_tweet']), epochs=15)
# prepare doc2Vec featureset
docvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(combined)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))

docvec_df = pd.DataFrame(docvec_arrays)
print("DOCVEC DATAFRAME")
print(docvec_df.head())