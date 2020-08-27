from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import string
import os
import json
import ast
import numpy as np
import scipy.cluster.hierarchy as sch
import codecs
import time
from scipy.spatial import distance
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import average, dendrogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import sparse

# Stop words from the nltk library
stops = set(stopwords.words('english'))
# nltk stemmer
stemmer = SnowballStemmer('english')


# Stem and tokenize words. Takes a list of strings, and uses nltk to tokenize and stem
def stem_tk(plain):
    # Create a list of tokenized words
    tks = [word for sentence in nltk.sent_tokenize(plain) for word in nltk.word_tokenize(sentence)]
    tks_filter = []

    # For loop iterates through list, filters, and creates list of stems
    for tk in tks:
        if re.search('[a-zA-Z]', tk):
            tks_filter.append(tk)
    stems = [stemmer.stem(i) for i in tks_filter]
    return stems


# Normal tokenization algorithm that doesn't implement stemming. Used before stemming
# def tokenize(plain):
#     tks = [word.lower() for sentence in nltk.sent_tokenize(plain) for word in nltk.word_tokenize(sentence)]
#     tks_filtered = []
#     for tk in tks:
#         if re.search('[a-zA-Z]', tk):
#             tks_filtered.append(tk)
#     return tks_filtered

# Fancy dendrogram plotting. Used this function to answer questions
# Modified from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram', fontsize=70)
        plt.xlabel('Similarity', fontsize=30)
        plt.ylabel('Index', fontsize=30)
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def main():
    # Default input path. File expected to be in same directory as program
    # Save current working directory path for dendrogram output later
    path = 'input-dir'
    wd = os.getcwd()
    os.chdir(path)

    # Initializing variables
    page_names = []
    documents = []
    doc_count = 0
    # Time
    start_time = time.time()

    # For loop that iterates over the input directory
    # Concatenates 2d matrix of strings
    for page in os.listdir(os.getcwd()):
        # Check for hidden files
        if not page.startswith('.') and os.path.isfile(os.path.join(os.getcwd(), page)):
            # Keeping track of pages that the strings belong to
            page_names.append(str(page))
            print('Processing ' + str(page) + '.....')

            # Text extraction
            html = open(page, encoding='latin1')
            soup = BeautifulSoup(html, 'lxml')
            plain_str = soup.get_text()
            html.close()
            plain_str = ast.literal_eval(json.dumps(plain_str))

            # Add to documents list
            documents.append(plain_str)
            doc_count += 1

    print('-------------------------------')
    print('Creating similarity matrix.....')

    # sklean TfidfVectorizer function.
    tfidf_vectorizor = TfidfVectorizer(max_df=0.4, use_idf=True, analyzer='word', tokenizer=stem_tk)
    tfidf_matrix = tfidf_vectorizor.fit_transform(documents)

    # Returns matrix of negative cosine similarity. 1 - gives us the matrix we want.
    similarity = 1 - cosine_similarity(tfidf_matrix)

    print('Similarity matrix finished.')
    print('-------------------------------')
    print('Clustering similarity matrix.....')

    # Scipy method for average clustering. Input similarity matrix
    matrix = average(similarity)
    # Time
    end_time = time.time()
    seconds = end_time - start_time
    print('Clustering finished.')
    print('-------------------------------')
    print(str(doc_count) + ' documents processed and clustered in ' + str(seconds) + ' seconds. ' + str(
        doc_count / seconds) + ' documents/second.')
    print('-------------------------------')
    print('Plotting data to avg_clusters.png.....')

    # Create dendrogram, and plot
    fig, ax = plt.subplots(figsize=(75, 100))  # set size
    ax = fancy_dendrogram(matrix, labels=page_names, leaf_font_size=12, show_contracted=True, orientation='right')

    # Save figure to avg_clusters.png
    os.chdir(wd)
    plt.savefig('avg_clusters.png', dpi=200)
    plt.close()
    print('avg_clusters.png created. Program finished.')


if __name__ == "__main__":
    main()
