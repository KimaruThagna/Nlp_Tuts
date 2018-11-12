import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk import ne_chunk

text=state_union.raw('2006- GWBush.txt')
custom_tnzr=PunktSentenceTokenizer(text)# this is synonimous to training the tokenizer using the corpus provided
tokenized=custom_tnzr.tokenize(text)

def process_content():

    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            namedEntity=ne_chunk(tagged,binary=True )
            namedEntity.draw()
            print(tagged)

    except Exception as e:
        print(str(e))

process_content()