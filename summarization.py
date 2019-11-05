from gensim.summarization.summarizer import summarize
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel
#document summary
model = LsiModel(common_corpus, id2word=common_dictionary)
vectorized_corpus = model[common_corpus]

text = 'Today was a good day. Started off with a good morning run, went out for a jog and later, when i came back,' \
       'I went to work feeling fresh'
print(summarize(text))