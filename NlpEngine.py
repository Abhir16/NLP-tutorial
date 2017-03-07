import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.classify.scikitlearn import SklearnClassifier
import sys

# reviewData = sys.stdin.read() # review text data passed in from node server
# TOKENIZE THE REVIEWS MIGHT NEED FOR LOOP, WAIT FOR INFO ON FORMATTING OF DATA
tokenized = nltk.word_tokenize("The quick brown fox jumps over the lazy dog.")
# PART OF SPEECH TAGGING
tagged = nltk.pos_tag(tokenized)
# print(tagged)

# ADD PATTERNS FOR NAME ENTITY RECOGNITION
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
results = cp.parse(tagged) # will be final name entity pairs, we need to hand


# le this now
print(results)


# wordnet use

word1 = wordnet.synset("word.n.01")
word2 = wordnet.synset("word.n.01")
print(word1.wup_similarity(word2))



