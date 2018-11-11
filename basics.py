import nltk,re,string
from textblob import TextBlob as tb
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize,word_tokenize,wordpunct_tokenize,ne_chunk,pos_tag
sample='Hello world. This is a sample text by Kimaru Thagana'
tokenized_sentence=sent_tokenize(sample) # extract sentences from a piece of text
tokenized_words=word_tokenize(sample) # extract words from a piece of text
words=wordpunct_tokenize(sample) #also seperates out punctuations
print(tokenized_sentence)
print(tokenized_words)
print(words)
print(nltk.pos_tag(tokenized_words))# parts of speech tagging
def entities(text):
    return ne_chunk((pos_tag(word_tokenize(text) ) ) )
comment=entities("When asked about the comments, Obama told the BBC: 'The UK will not be able to negotiate something with the US;")
comment.pprint()

# using textblob to capture sentiment
'''
Polarity- How positive or negative the sentiment is 1.0- +ve -1.0 -ve
Subjectivity- A measure of how subjective the text is, i.e, influenced by emotions and opinions and is subjected to intermpretation
'''
text1="The food at radison was not so good"
text2="I hate you"
print(tb(text1).sentiment)
print(tb(text2).sentiment)


####################
# Text Pre-Processing Techniques
####################

#The first process in text processing is normalization. This involves:
#1. converting text to lower or upper case using the .lower()/.upper() method on a string

#2. Removing numbers if they arent important in the analysis
input_str = "Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls."
result = re.sub(r"\d+", "", input_str)
print(result)
#3. Removing punctuation
input_str = "This &is [an] example? {of} string. with.? punctuation!!!!” # Sample string"
result = input_str.translate(string.maketrans("",""), string.punctuation)
print(result)
#4. Removing stop words(they dont convey meaning hence nit useful in processing
input_str = "NLTK is a leading platform for building Python programs to work with human language data."
stop_words = set(stopwords.words("english"))
from nltk.tokenize import word_tokenize
tokens = word_tokenize(input_str)
result = [i for i in tokens if not i in stop_words]
print (result)

#5. Stemming. Find example in stemming.py
#6. Lemmatization. Works the same way as stemming but uses lexical knowledge bases to find
#the correct root form of a word
lemmatizer=WordNetLemmatizer()
input_str="been had done languages cities mice"
input_str=word_tokenize(input_str)
for word in input_str:
    print(lemmatizer.lemmatize(word))

#7 Chunking. An example can be found in chunking.py

'''
text1="I love the food at radison but the waiters were not good"
text2="I do not hate you"
text1="I love the food at radison but the waiters were not good"
text2="I do not like you"
text1="I love the food at radison but the waiters were not good"
text2="I hate you"
text1="I love the food at radison but the waiters were rude"
text2="I hate you"
text1="I love the food at radison. It was nice"
text2="I hate you"
text1="The food at radison was awsome"
text2="I hate you"
text1="The food at radison was good"
text2="I hate you"
text1="The food at radison was really awsome"
text2="I hate you"
text1="The food at radison was not so good"
text2="I hate you
text1="The food at radison was not so bad"
text2="I hate you
'''