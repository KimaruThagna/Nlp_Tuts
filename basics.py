import nltk
from textblob import TextBlob as tb
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