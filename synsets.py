from nltk.corpus import wordnet

syns = wordnet.synsets("seconds")

print(syns)


print("finding similarities in words")
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2))


stg = "12345"

stg = stg[1:-1]
print(stg)
