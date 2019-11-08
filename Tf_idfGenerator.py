from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json

ps = PorterStemmer()
inverted_index = {}
DocCount=0
with open('dataTest.json') as json_file:
    data = json.load(json_file)
    for entry in data:
        DocCount+=1
        docName = entry["ID"]
        text = entry["Title"] + " " + entry["Author"] +" "+ entry["Type"]+ " "+entry["Platform"]
        terms = word_tokenize(text)
        for term in terms:
            term = ps.stem(term)
            if term not in inverted_index:
                inverted_index[term] = {}
                inverted_index[term][docName]= 1
            elif docName not in inverted_index[term]:
                inverted_index[term][docName] = 1
            elif docName in inverted_index[term]:
                inverted_index[term][docName] += 1
print(inverted_index)