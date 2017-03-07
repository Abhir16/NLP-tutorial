import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

train = state_union.raw("2005-GWBush.txt")
sample = state_union.raw("2006-GWBush.txt")
print(train)

custom_sent_tokenizer = PunktSentenceTokenizer(train)

tokenized = custom_sent_tokenizer.tokenize(sample)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as ex:
        print(str(ex))



process_content()
# run function to print tags from document



import re

exStr = '''
Jessica is 15 years old, and Daniel is 27 years old.
Edward is 97, and his grandfather, Oscar, is 102.
'''

# print(exStr)

ages = re.findall(r'\d{1,3}', exStr)
names = re.findall(r'[A-Z][a-z]*', exStr)

# print('results:')
# print(ages)
# print(names)
