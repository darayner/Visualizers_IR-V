from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import math


def main():
    queries_to_test = {}
    inverted, doc_count = create_inverted_index()
    tf_idf = create_tf_idf_matrix(inverted, doc_count)
    normalized = create_normalized_matrix(tf_idf)
    matrices_to_output = {'invertedIndex': inverted, 'tfIdfMatrix': tf_idf, 'normalizedMatrix': normalized}
    output_json(matrices_to_output)


def create_inverted_index():
    ps = PorterStemmer()
    inverted_index = {}
    doc_count = 0
    with open('data.json') as json_file:
        data = json.load(json_file)
        for entry in data:
            doc_count += 1
            doc_name = entry["ID"]
            text = entry["Title"] + " " + entry["Author"] + " " + entry["Type"] + " " + entry["Platform"]
            terms = word_tokenize(text)
            for term in terms:
                term = ps.stem(term)
                if term not in inverted_index:
                    inverted_index[term] = {}
                    inverted_index[term][doc_name] = 1
                elif doc_name not in inverted_index[term]:
                    inverted_index[term][doc_name] = 1
                elif doc_name in inverted_index[term]:
                    inverted_index[term][doc_name] += 1
    return inverted_index, doc_count


def create_tf_idf_matrix(inverted_index, doc_count):
    idfDic = {}
    for term in inverted_index:
        tfSum = 0
        for doc, value in inverted_index[term].items():
            tfSum += value
        idf = doc_count/tfSum
        idfDic[term] = idf
    for term in inverted_index:
        for doc, value in inverted_index[term].items():
            inverted_index[term][doc] = round(math.log(1 + value, 10) * math.log(idfDic[term], 10), 2)
    return inverted_index


def create_normalized_matrix(tf_idf_matrix):
    normalized_doc_matrix = {}
    for term in tf_idf_matrix:
        for doc, value in tf_idf_matrix[term].items():
            if doc not in normalized_doc_matrix:
                normalized_doc_matrix[doc] = {}
                normalized_doc_matrix[doc][term] = value
            else:
                normalized_doc_matrix[doc][term] = value
    for doc in normalized_doc_matrix:
        summation = 0
        for term, value in normalized_doc_matrix[doc].items():
            summation += math.pow(value, 2)
        l2_norm = math.sqrt(summation)
        for term, value in normalized_doc_matrix[doc].items():
            normalized_doc_matrix[doc][term] = round(value / l2_norm, 2)
    return normalized_doc_matrix


def cos_sim_matrix():
    pass


def output_json(matrices_to_output):
    for filename, matrix in matrices_to_output.items():
        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(matrix, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()