from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import math


def main():
    query_to_test = {'Query': "Adobe Windows security team"}
    inverted, doc_count = create_inverted_index()
    inverted = add_query_to_inverted(inverted, query_to_test)
    tf_idf = create_tf_idf_matrix(inverted, doc_count)
    normalized = create_normalized_matrix(tf_idf)
    cos_sim = cos_sim_matrix(normalized, 'Query')
    matrices_to_output = {'normalizedMatrix': normalized, 'cosSim': cos_sim}
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


def add_query_to_inverted(inverted_index, query_to_test):
    ps = PorterStemmer()
    for query_id, query in query_to_test.items():
        terms = word_tokenize(query)
        for term in terms:
            term = ps.stem(term)
            if term in inverted_index:
                inverted_index[term][query_id] = 1
            elif query_id in inverted_index[term]:
                inverted_index[term][query_id] += 1
    return inverted_index


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


def cos_sim_matrix(normalized_matrix, query_id):
    cos_sim = {}
    sorted_cos_sim = {}
    query_summation = 0
    query_vector = normalized_matrix[query_id]
    for term, value in query_vector.items():
        query_summation += math.pow(value, 2)
    query_l2_norm = math.sqrt(query_summation)
    for doc in normalized_matrix:
        doc_summation = 0
        dot_product = 0
        for term, value in normalized_matrix[doc].items():
            if term in query_vector:
                dot_product += (value * query_vector[term])
            doc_summation += math.pow(value, 2)
        doc_l2_norm = math.sqrt(doc_summation)
        cos_sim[doc] = round(dot_product/(query_l2_norm * doc_l2_norm), 2)
    for doc, sim in sorted(cos_sim.items(), key=lambda item: item[1], reverse=True):
       sorted_cos_sim[doc] = sim
    return sorted_cos_sim


def output_json(matrices_to_output):
    for filename, matrix in matrices_to_output.items():
        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(matrix, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()