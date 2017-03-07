import nltk
import sys
print(sys.argv)

def printTags(tag, t):
    def filt(x):
        return x.label() == tag

    for subtree in t.subtrees(filter=filt):  # Generate all subtrees
        ls = []
        print(subtree)
        ls.append(subtree)
    return ls


while 1:
    inpt = input("enter: ")
    if inpt == 'q':
        break
    tokens = nltk.word_tokenize(inpt)
    pos = nltk.pos_tag(tokens)
    print(pos)

    # grammar = "NP: {<DT>?<JJ>*<NN>* | <jj>?<NN> | <NN>*<JJ>?}"
    # cp = nltk.RegexpParser(grammar)
    # results = cp.parse(pos)
    # print(results)
    #
    # grammar = "Desires: {<MD>?<VB>*<JJ>*<TO>?<NN>*}"
    # print('')
    # cp = nltk.RegexpParser(grammar)
    # results = cp.parse(pos)
    # print(results)

    # grammar = "feature: {<JJ>*<NNP>* | <JJ>*<NNS>* | <NNS>*<VBZ>*<JJ>*}"
    grammar = r"""
      NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
      PP: {<IN><NP>}               # Chunk prepositions followed by NP
      VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
      CLAUSE: {<NP><VP>}           # Chunk NP, VP
      """
    print('')
    cp = nltk.RegexpParser(grammar)
    results = cp.parse(pos)
    print(results)

    NPchks = printTags('NP', results)
    # NPchks = a[0].leaves()
    ls = []








