import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
print(custom_sent_tokenizer)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        # for i in tokenized:
            # words = nltk.word_tokenize(i)
            # tagged = nltk.pos_tag(words)
            # chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            # chunkParser = nltk.RegexpParser(chunkGram)
            # chunked = chunkParser.parse(tagged)
            # chunked.draw()
        words = nltk.word_tokenize("The quick brown fox jumps over the lazy dog.")
        words = nltk.word_tokenize("Happy to have discovered this app. So easy to fund our account. The checks deposit quickly. Will soon be wondering how we lived without it!")
        # words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        brown = nltk.pos_tag("brown")
        grammar = "NP: {<DT>?<JJ>*<NN> | <jj>?<NN> | <NN><JJ>?}"
        # grammar = r"""
  # NP:
  #   {<.*>+}          # Chunk everything
  #   }<VBD|IN>+{      # Chink sequences of VBD and IN
  # """
        cp = nltk.RegexpParser(grammar)
        chunked_sentences = nltk.ne_chunk(tagged, binary=True)
        nameEnt = nltk.ne_chunk(tagged, binary=True)
        print(tagged)
        print(nameEnt)
        results = cp.parse(tagged)
        print("final ne:", chunked_sentences)
        # chunked_sentences.draw()
        results.draw()

    except Exception as e:
        print(str(e))

process_content()

