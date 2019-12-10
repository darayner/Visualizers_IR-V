import textmining

import json

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, PorterStemmer

from nltk.tokenize import sent_tokenize , word_tokenize

import glob

import re

import os

import numpy as np

import sys



def termdocumentmatrix_example():

    # Create some very short sample documents

    f = open("C:/Users/sujey/Desktop/FALL 2019/IR/tfIdfMatrix.json","r", encoding = 'utf-8')

    message = f.read()

    tdm = textmining.TermDocumentMatrix()

    # Add the documents

    tdm.add_doc(message)

    # Write out the matrix to a csv file. Note that setting cutoff=1 means

    # that words which appear in 1 or more documents will be included in

    # the output (i.e. every word will appear in the output). The default

    # for cutoff is 2, since we usually aren't interested in words which

    # appear in a single document. For this example we want to see all

    # words however, hence cutoff=1.

    tdm.write_csv('matrix.csv', cutoff=1)

    # Instead of writing out the matrix you can also access its rows directly.

    # Let's print them to the screen.

    for row in tdm.rows(cutoff=1):

        print (row)



termdocumentmatrix_example()