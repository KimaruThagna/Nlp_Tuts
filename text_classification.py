import nltk,random
from nltk.corpus import movie_reviews

documents=[(list(movie_reviews.words(fileid)), category) #tuple containing a word and its category, whether +ve oe -ve
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
all_words=[]
for word in movie_reviews.words():
    all_words.append(word.lower())

all_words=nltk.FreqDist(all_words)
print(all_words.most_common(5)) # top 5 words with the highest frequency
print(all_words['love']) # how many times the word love appears

word_features=list(all_words.keys())[:3000]

def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets=[(find_features(rev),category) for (rev,category) in documents]
train=featuresets[:1900]
test=featuresets[1900:]

#https://www.youtube.com/watch?v=5xDE06RRMFk

#https://www.youtube.com/watch?v=6WpnxmmkYys
# preparing datasets
#https://www.youtube.com/watch?v=0xVqLJe9_CY