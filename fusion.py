import os
import pandas as pd
import csv
from tqdm import tqdm

def generate_begin_grams(N, lst_begin):       
        tmp = []
        for i in range(N):
            tmp.append('Partie' + str(i + 1))
        tmp.append('Support')
        col = [tmp]
        title = f"{N}_grams_begin.csv"
        f = open(title, 'w', encoding="utf-8")

        with f:
            writer = csv.writer(f)
            writer.writerows(col)

            format_data = [val + (lst_begin[val],) for val in lst_begin.keys()]
            writer.writerows(format_data)
        f.close()

def combine_dicts(A, B):
    return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}

def csvfusion(N):
    path = "all/"
    filenames = [i for i in os.listdir(path) if i.endswith("csv")]
    dictionaries = []

    for i, filename in tqdm(enumerate(filenames)):
        current_dict = {}  
        df = pd.read_csv(path+filename).values
        pandas_to_dict(current_dict, df)
        dictionaries.append(current_dict)
    
    finaldict = dictionaries[0]
    cpt = 1
    for i in tqdm(range(len(dictionaries) - 1)):
        finaldict = combine_dicts(finaldict, dictionaries[cpt])
        cpt += 1
    generate_begin_grams(N, finaldict)

def pandas_to_dict(current_dict, df):
    for tab in df:
        ngrams = tuple()
        for i in range(len(tab) - 1):
            ngrams += (tab[i], )
        current_dict.update({ngrams: int(tab[len(tab) - 1])})



csvfusion(3)

