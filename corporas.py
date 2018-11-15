from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize`

sample_text=gutenberg.raw('bible-kjv.txt')
tokenized_sentense=sent_tokenize(sample_text)