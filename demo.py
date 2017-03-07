import nltk
import sys
import pickle
from nltk.corpus import conll2000
print(sys.argv)

def save_classifier(classifier):
   f = open('my_classifier.pickle', 'wb')
   pickle.dump(classifier, f, -1)
   f.close()

def load_classifier():
   f = open('my_classifier.pickle', 'rb')
   classifier = pickle.load(f)
   f.close()
   return classifier

# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     return {"pos": pos}

def tags_since_dt(sentence, i):
     tags = set()
     for word, pos in sentence[:i]:
         if pos == 'DT':
             tags = set()
         else:
             tags.add(pos)
     return '+'.join(sorted(tags))

def npchunk_features(sentence, i, history):
     word, pos = sentence[i]
     if i == 0:
         prevword, prevpos = "<START>", "<START>"
     else:
         prevword, prevpos = sentence[i-1]
     if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
     else:
         nextword, nextpos = sentence[i+1]
     return {"pos": pos,
             "word": word,
             "prevpos": prevpos,
             "nextpos": nextpos,
             "prevpos+pos": "%s+%s" % (prevpos, pos),
             "pos+nextpos": "%s+%s" % (pos, nextpos),
             "tags-since-dt": tags_since_dt(sentence, i)}

class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        #         TRAIN CLASSIFIER
        # self.classifier = nltk.MaxentClassifier.train(
        #     train_set, algorithm='GIS', trace=0)
        # LOAD CLASSIFIER
        self.classifier = load_classifier()
        # SAVE CLASSIFIER
        # save_classifier(self.classifier)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def printTags(tag, t):
    def filt(x):
        return x.label() == tag

    for subtree in t.subtrees(filter=filt):  # Generate all subtrees
        ls = []
        print(subtree)
        ls.append(subtree)
    return ls



test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

# start pickling the classifier to save time in training

while 1:

    inpt = input('Enter sentence:')
    if inpt.lower() == 'q':
        break
    print('results:')
    tags = nltk.pos_tag(nltk.word_tokenize(inpt))
    print(chunker.parse(tags))
    printTags('NP', )





