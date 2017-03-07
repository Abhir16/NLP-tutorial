from nltk.stem import WordNetLemmatizer
import nltk

lemmetizer = WordNetLemmatizer()

print(lemmetizer.lemmatize("better", pos='a'))
print(lemmetizer.lemmatize("rocks"))
print(lemmetizer.lemmatize("stone"))
print(lemmetizer.lemmatize("cacti"))


print("path: " + nltk.__file__)# path to your python packages