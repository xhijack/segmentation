

from gensim.models import Doc2Vec
# text = ''
# with open('wiki.id.non_case.text') as file:
#     text = file.read()
#     file.close()
# # import pdb
# # pdb.set_trace()
# sentences = text
from gensim.models.doc2vec import LabeledSentence

filename = 'wiki.id.non_case.text'


class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for uid, line in enumerate(open(filename)):
            # import pdb
            # pdb.set_trace()
            # yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
            yield LabeledSentence(words=line.split(), tags=['SENT_%s' % uid])

sentences = LabeledLineSentence('wiki.id.non_case.text')

# import pdb
# pdb.set_trace()

model = Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.build_vocab(sentences)
for epoch in range(10):
    model.train(sentences, epochs=model.iter, total_examples=model.corpus_count)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

model.save('my_model.doc2vec')