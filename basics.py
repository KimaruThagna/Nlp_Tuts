import nltk
import textblob as tb
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
