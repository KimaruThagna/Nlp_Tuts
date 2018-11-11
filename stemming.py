'''
Purpose of stemming. To ensure variations of a verb in terms of tense all go back to the sme thing to help in decoding the meaning of a sentence
example of stemmers...brown stemmer, porter stemmer, snowball stemmer
'''
from nltk.stem import PorterStemmer,SnowballStemmer
from nltk.tokenize import word_tokenize
ps=PorterStemmer()
example_words=['python','pythoner','pythoning','pythoned']
for w in example_words:
    print(ps.stem(w))

# use ps stemmer to stem a sentence after it being tokenized. Ensure the sentence has varied tense verbs

#compare with snowball stemmer
sb=SnowballStemmer('english')
print('=================++++++=============++=\n Snowball stemmer')
for w in example_words:
    print(sb.stem(w))
# Comparison with lematization