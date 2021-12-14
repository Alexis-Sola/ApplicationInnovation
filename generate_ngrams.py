import os
from nltk.probability import FreqDist
import numpy as np
import re
from nltk.util import ngrams
import csv
import pandas as pd

def combine_dicts(A, B):
    return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}


class NGramsGenerator():
    def __init__(self, N, char_to_remove):
        self.char_to_remove = char_to_remove
        self.N = N # N = nombre de grams
        self.lines = []

    def open_file(self, path):
        with open(path) as f:
            self.lines = f.readlines()

    def compute_ngrams(self):
        try:
            self.lines.index(0)
        except:
            tokens = []
            lst_n_grams = []
            lst_begin = []
            for val in self.lines:
                val = val.lower()
                tmp = ""
                tmp = re.sub(self.char_to_remove, tmp, val)
                tmp = tmp.replace(' ...', '.')
                tmp = tmp.replace('...', '.')
                tmp = tmp.replace('.', ' .')
                tmp = tmp.replace('!', ' !')
                #tmp = tmp.replace('’', ' ')
                # tmp = tmp.replace('\'', ' ')
                # tmp = tmp.replace('`', ' ')             
                 # remove unwanted char
                tmp = re.split('\t', tmp) # split string into word              
                tmp = ' '.join(tmp).split()
                #tokens.extend(tmp)
                tmp_ngrams = list(ngrams(tmp, self.N, pad_left=False, pad_right=False))

                if len(tmp_ngrams) != 0:
                    lst_begin.extend((tmp_ngrams[0],))

                lst_n_grams.extend(tmp_ngrams)
            return lst_n_grams, tokens, lst_begin
    
    def generate_file(self, lst_n_grams):       
        tmp = []
        for i in range(self.N):
            tmp.append('Partie' + str(i + 1))
        tmp.append('Support')
        col = [tmp]

        val = 0
        title = f"{self.N}_grams{val}.csv"
        f = open(title, 'w', encoding="utf-8")

        with f:
            writer = csv.writer(f)
            writer.writerows(col)

            print('---------------- PHASE 2 : WRITING DATA ----------------')
            format_data = [val + (lst_n_grams[val],) for val in lst_n_grams.keys()]
            writer.writerows(format_data)
        f.close()

    def generate_begin_grams(self, lst_begin):       
        tmp = []
        for i in range(self.N):
            tmp.append('Partie' + str(i + 1))
        tmp.append('Support')
        col = [tmp]
        title = f"{self.N}_grams_begin.csv"
        f = open(title, 'w', encoding="utf-8")

        with f:
            writer = csv.writer(f)
            writer.writerows(col)

            print('---------------- PHASE 3 : WRITING N GRAMS BEGINNING ----------------')
            format_data = [val + (lst_begin[val],) for val in lst_begin.keys()]
            writer.writerows(format_data)
        f.close()

def compute_tf():
    df = pd.read_csv("2_grams0.csv")
    partie1 = list(df["Partie1"].values)
    partie2 = list(df["Partie2"].values)
    sup = list(df["Support"])

    sum_bigrams = sum(sup)

    tf_dictionary = {}

    for i, val in enumerate(partie1):
        current_tuple = (val, partie2[i])
        tf = round(sup[i] / sum_bigrams, 6)
        tf_dictionary.update({current_tuple: tf})

    return tf_dictionary

def nb_tot_document():
    totfile = 0
    path = 'data/MEGALITE_FR/'
    directory = [i for i in os.listdir(path)]

    for num, d in enumerate(directory):
        pathd = path + d + '/'
        filenames = [i for i in os.listdir(pathd) if i.endswith("seg")]
        for i, file_name in enumerate(filenames):
            totfile += 1
    return totfile    

def compute_tf_idf():
    tf = compute_tf()
    totfile = nb_tot_document()
    tfidf = tf.copy()
    tfidf = dict(zip(list(tfidf.keys()), [0 for i in range(len(tfidf.values()))]))

    char_to_remove = '[(),:"–\-_;?—|+\#\[\])0-9«»]+'
    N = 2
    generator = NGramsGenerator(N, char_to_remove)

    path = 'data/MEGALITE_FR/'
    directory = [i for i in os.listdir(path)]

    processing = 0

    for num, d in enumerate(directory):
        pathd = path + d + '/'
        filenames = [i for i in os.listdir(pathd) if i.endswith("seg")]
        for i, file_name in enumerate(filenames):
            generator.open_file(pathd + file_name)
            ngrams_tmp, _, __ = generator.compute_ngrams()
            process_idf = 0
            for tuple in tf.keys():               
                if tuple in ngrams_tmp:
                    tfidf[tuple] += 1
                process_idf += 1
                print(f'Processing tuple in file {round((process_idf * 100) / len(tf.keys()), 2)} %', end='\r')
            processing += 1
            print(f'Processing tfidf {round((processing * 100) / totfile, 2)} %', end='\r')

    vals = list(tfidf.values())
    for i in range(vals):
        vals[i] = tf.values[i] * np.log(totfile / vals[i] + 1)       
    tfidf = dict(zip(list(tfidf.keys()), vals))

    return totfile    




def main():
    char_to_remove = '[(),:"–\-_;?—|+\#\[\])0-9«»]+'
    #char_to_remove = ''
    path = 'data/MEGALITE_FR/'
    directory = [i for i in os.listdir(path)]
    N = 3
    generator = NGramsGenerator(N, char_to_remove)
    n_grams = {}
    begin = {}
    totfile = nb_tot_document()

    print('---------------- PHASE 1 : SEARCH N GRAMS ----------------')
    processing = 0
    for num, d in enumerate(directory):
        pathd = path + d + '/'
        filenames = [i for i in os.listdir(pathd) if i.endswith("seg")]
        for i, file_name in enumerate(filenames):    
            #ngrams_tmp = []
            generator.open_file(pathd + file_name)
            ngrams_tmp, _, lst_begin = generator.compute_ngrams()

            freq_ngrams_tmp = FreqDist(ngrams_tmp)
            n_grams = combine_dicts(n_grams, freq_ngrams_tmp)

            freq_ngrams_begin_tmp = FreqDist(lst_begin)
            begin = combine_dicts(n_grams, freq_ngrams_begin_tmp)
            processing += 1

            print(f'CURRENT DIRECTORY : {d} {num + 1}/{len(directory)} CURRENT FILE : {i}/{len(filenames)} LOADING : {round((processing * 100) / totfile, 2)} %', end='\r')

    generator.generate_file(n_grams)
    generator.generate_begin_grams(begin)


if(__name__ == "__main__"):
    main()
   #compute_tf_idf()
   
