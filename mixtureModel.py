from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import math
from fractions import Fraction

def main():
    lam = float(Fraction(1, 2))
    query_to_test = "nostromo"
    query_to_test = prepare_query(query_to_test)
    mixtureModule, term_count = create_mixtureModule(query_to_test)
    mixtureModelResult = calculate_mixtureModel(mixtureModule, query_to_test, term_count, lam)

def prepare_query(query_to_test):
    ps = PorterStemmer()
    query_to_test = word_tokenize(query_to_test)
    query = []
    for word in query_to_test:
        query.append(ps.stem(word))
    
    return query

def create_mixtureModule(query_to_test):
    ps = PorterStemmer()    
    mixtureModule_index = {}
    doc_count = 0
    
    with open('data.json') as json_file:
        data = json.load(json_file)
        
        term_count = 0

        for entry in data:
            termList = []
            flag = False
            doc_count += 1          
            doc_name = entry["ID"]
            text = entry["Title"] + " " + entry["Author"] + " " + entry["Type"] + " " + entry["Platform"]            
            terms = word_tokenize(text)

            for term in terms:
                term = ps.stem(term)
                if term in query_to_test:
                    flag = True                    
                    term_count = len(terms) + term_count
                    break      

            if flag == True:
                 for term in terms:
                     term = ps.stem(term)
                     termList.append(term)
                 mixtureModule_index[doc_name] = termList
                 
    return mixtureModule_index, term_count

def calculate_mixtureModel(mixture_module, query_to_test, total_terms, lam):
    ps = PorterStemmer()    
    calMixtureModule = {}    
    doc_count = 0

    for doc, value in mixture_module.items():
        calc = 1
        term_counter = 0
        for term in query_to_test:  
            termsInDoc = value.count(term)
            for doc_, value_ in mixture_module.items():
                term_counter = term_counter + value_.count(term)
            calc = calc + (float(Fraction(termsInDoc, len(value)+1)) + float(Fraction(term_counter, total_terms)))/lam
        calMixtureModule[doc] = calc

    return calMixtureModule

if __name__ == "__main__":
    main()
