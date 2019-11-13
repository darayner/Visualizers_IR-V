from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import json

ps = PorterStemmer()
inverted_index = {}

def main():
    with open('dataTest.json') as json_file:
        data = json.load(json_file)
        for entry in data:
            docName = entry["ID"]
            text = entry["Title"] + " " + entry["Author"] +" "+ entry["Type"]+ " "+entry["Platform"]
            terms = word_tokenize(text)
            for term in terms:
                if ps.stem(term) not in inverted_index:
                    inverted_index[ps.stem(term)] = "d" + str(docName)
                elif str(docName) not in inverted_index[ps.stem(term)]:
                    inverted_index[ps.stem(term)] = inverted_index[ps.stem(term)] + ", d" + str(docName)
            print(docName)
            print(text)
