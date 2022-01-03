import numpy as np
import re
from scipy.spatial.distance import cosine
import random
import pandas as pd
import nltk

INF = 999999999999

def open_file(path):
    with open(path) as f:
        lines = f.readlines()
    return lines

# retorune la table associative à partir d'un fichier
def read_table_asso(path):
    lines = open_file(path)
    assos = {}
    for line in lines:
        splited = re.split('\t', line)
        key = splited[0]
        splited.pop(0)
        assos.update({key : tuple(splited)})
    return assos

def rm_char(intial_list, list_to_rm):
    result = []
    for i in intial_list:
        if(i not in list_to_rm):
            result.append(i)

    for i in range(len(result)):
        for j in list_to_rm:
            result[i] = result[i].replace(j, "")

    return result

# charge les embeddings à partir d'un fichier
def load_embeddings(File):
    print("Loading Model")
    model = {}
    with open(File, 'r') as f:
        for line in f:
            split_line = line.split()
            split_line = rm_char(split_line, ["[", ",", "]"])

            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            model[word] = embedding
    print(f"{len(model)} words loaded!")
    return model

# retorune la valeurs de chaque par rapport au paramètre choisit.
def get_calcul_function(model, query, word, remoteness_word, list_pred_word = [], euclidean = True):
    try:
        v1 = model[query]
        v2 = model[word]
        v3 = model[remoteness_word]
        if(euclidean):
            dist = np.linalg.norm(v1 - v2)**2 - np.linalg.norm(v2 - v3)
        else:
            dist = cosine(v1, v2)
            remote = cosine(v2, v3)
            if (remote > dist):
                return None
            else:
                dist = -dist
        for i in list_pred_word:
            if(euclidean):
                dist = dist + np.linalg.norm(model[i] - v2)
            else:
                dist = dist - cosine(model[i], v2)

        return dist
    except KeyError:
        return None

# retorune le nombre d'occurences de chaque mot
def get_occurences(pred_word, word, bigrams):
    try:
        result = bigrams[(pred_word, word)]
        return result
    except KeyError:
        return None

# détermine le mot choisi parmis la liste de bigram
def get_closer2(model, query, pred, words_list, remoteness_word, bigrams, forbidden_words = [], euclidean = True):
    distances = []
    words = []
    closer = ""
    for i in words_list:
        bigram_occurence = get_occurences(pred, i, bigrams)
        if(bigram_occurence != None):
            dist = get_calcul_function(model, query, i, remoteness_word, forbidden_words, euclidean = euclidean)
        else:
            dist = INF

        if(dist == None):
            continue
        distances.append(dist)
        words.append(i)

    no_bigrams = distances.count(None) == len(distances)
    if(no_bigrams):
        return get_closer(model, query, words_list, remoteness_word, forbidden_words, euclidean = euclidean)

    min_distance = INF
    if(not euclidean):
        for i in range(len(distances)):
            if(distances[i] < 0 and distances[i] != INF):
                distances[i] = -distances[i]

    for i in range(len(distances)):
        if(distances[i] < min_distance and words[i] != query and words[i] not in forbidden_words):
            min_distance = distances[i]
            closer = words[i]

    if(closer == ""):
        closer = random.choice(words_list)
    return closer

# détermine le mot choisi
def get_closer(model, query, words_list, remoteness_word, forbidden_words = [], euclidean = True):
    distances = []
    words = []
    closer = ""
    for i in words_list:
        dist = get_calcul_function(model, query, i, remoteness_word, forbidden_words, euclidean = euclidean)
        if(dist == None):
            continue
        distances.append(dist)
        words.append(i)
    min_distance = INF
    if(not euclidean):
        for i in range(len(distances)):
            if(distances[i] < 0):
                distances[i] = -distances[i]

    for i in range(len(distances)):
        if(distances[i] < min_distance and words[i] != query and words[i] not in forbidden_words):
            min_distance = distances[i]
            closer = words[i]

    if(closer == ""):
        closer = random.choice(words_list)
    return closer

# retourne pour une phrase, la liste des élément associatif, les mots initiuax, les lemms de ces mots, les mots précédants l'élément de la table associative et les position de ces mots dans la phrase
def extract_info(sentence):
    assoc = []
    words = []
    lems = []
    preds = []

    end = 0
    tokens = nltk.word_tokenize(sentence)
    for i in range(len(tokens)):
        pred = ""
        if(tokens[i] == "*"):
            pred = (tokens[i-1])
        if(pred != ""):
            if(tokens[i-1] == "'"):
                pred = tokens[i-2] + pred
            preds.append(pred)

    for i in range(sentence.count('*')):

        start = sentence.find("*", end)
        end = sentence.find("/", start)
        assoc.append(sentence[start + 1: end])

        start = end + 1
        end = sentence.find("/", start)
        words.append(sentence[start: end])

        start = end + 1
        end = sentence.find(" ", start)
        lems.append(sentence[start: end])

        preds_index = []
        count = 0
        for i in range(len(sentence)):
            if(sentence[i] == "*"):
                preds_index.append(i - len(preds[count])+1)
                count = count + 1

    return assoc, words, lems, preds, preds_index

# retourne tout les mots à placer dans la phrase
def words_generation(model, dictionnary, query, sentences, iterations=1, euclidean = True, bigrams_path = None):
    if(bigrams_path != None):
        df = pd.read_csv(bigrams_path)
        partie1 = df["Partie1"].values
        partie2 = df["Partie2"].values
        sup = df["Support"]

        dictionnary = {}

        for i in range(len(partie1)):
            dictionnary.update({(partie1[i], partie2[i]): sup[i]})

    count = 0
    result = []
    for sentence in sentences:
        count = count + 1
        assoc, words, lems, preds, index = extract_info(sentence)

        forbidden_words = []
        tmp = []
        for i in range(iterations):
            #print("Step ", i + 1, " for sentence ", count)
            w = []
            for j in range(len(assoc)):
                if(bigrams_path != None):
                    closer = (get_closer2(model, query, preds[i], dictionnary[assoc[j]], words[j], dictionnary, forbidden_words, euclidean=euclidean))
                else:
                    closer = (get_closer(model, query, dictionnary[assoc[j]], words[j], forbidden_words, euclidean = euclidean))

                w.append(closer)
                forbidden_words.append(closer)
            tmp.append(w)
        result.append(tmp)

    return result

# génère la phrase à partir des mots choisi précédamment
def sentences_generations(generated_words, sentences):
    generations = []
    for i in range(len(sentences)):
        assoc, words, lems, preds, index = extract_info(sentences[i])
        for j in generated_words[i]:
            generation = sentences[i]
            for k in range(len(j)):
                generation = generation.replace("*" + assoc[k] + "/" + words[k] + "/" + lems[k], j[k])

            generations.append(generation)

    return generations

print("###############################################################################################################")
print("########################################## MAIN ###############################################################")
print("###############################################################################################################")


path_asso = "/home/ubuntu/M2S2/defi/Ressources Neuronales+Table associative+Structures grammaticales -20211115/TableAssociative"
model = load_embeddings("/home/ubuntu/M2S2/defi/Ressources Neuronales+Table associative+Structures grammaticales -20211115/embeddings-Fr.txt")
dic = read_table_asso(path_asso)
# doit contenir le chemin vers le fichier contenant les bigrams avec les règles suivantes:
# la première colonne doit s'appeller Partie1 et elle doit contenir la première partie des bigrams
# la deuxième colonne doit s'appeller Partie2 et elle doit contnire la seconde partie des bigrams
# la troisième colonne doit s'appeller Support et elle doit contenir les occurences des bigrams de la ligne
bigrams_path = "/home/ubuntu/M2S2/defi/Fichiers csv/2_grams0.csv"
iterations = 1
queries = ["tristesse", "amour", "joie", "haine", "bleu"]
sentences = [
    "Il n' y a pas_de *NCFS000/littérature/littérature sans *NCMS000/péché/péché .",
    "Il n' y a ni *AQ0FS00/morale/moral ni *NCFS000/responsabilité/responsabilité en *NCFS000/littérature/littérature .",
    "En *NCFS000/littérature/littérature , la *AQ0FS00/première/premier *NCFS000/impression/impression *VMIP3S0/est/être la plus *AQ0FS00/forte/fort",
    "La *NCFS000/littérature/littérature *VMIP3S0/est/être une *NCFS000/maladie/maladie .",
    "L' *NCMS000/enseignement/enseignement de les *NCFP000/lettres/lettre *VMIP3S0/est/être à la *NCFS000/littérature/littérature ce_que la *NCFS000/gynécologie/gynécologie *VMIP3S0/est/être à l' *NCMS000/érotisme/érotisme .",
    "Les *NCMP000/professeurs/professeur de *NCFP000/lettres/lettre *VMIP3P0/connaissent/connaître de la *NCFS000/littérature/littérature ce_que les *NCFP000/prostituées/prostitué *VMIP3P0/connaissent/connaître de l' *NCMS000/amour/amour .'"
]


print("__________________ Euclidean distance without bigrams __________________")
for query in queries:
    words = (words_generation(model, dic, query, sentences, iterations=iterations, euclidean = True))
    generations = sentences_generations(words, sentences)

    for i in range(len(generations)):
        print(generations[i] + "\t" + query)

print("__________________ Euclidean distance with bigrams __________________")
for query in queries:
    words = (words_generation(model, dic, query, sentences, iterations=iterations, euclidean = True, bigrams_path=bigrams_path))
    generations = sentences_generations(words, sentences)

    for i in range(len(generations)):
        print(generations[i] + "\t" + query)
        
print("__________________ Cosine Similarity without bigrams __________________")
for query in queries:
    words = (words_generation(model, dic, query, sentences, iterations=iterations, euclidean = False))
    generations = sentences_generations(words, sentences)

    for i in range(len(generations)):
        print(generations[i] + "\t" + query)

print("__________________ Cosine Similarity with bigrams __________________")
for query in queries:
    words = (words_generation(model, dic, query, sentences, iterations=iterations, euclidean=False, bigrams_path=bigrams_path))
    generations = sentences_generations(words, sentences)

    for i in range(len(generations)):
        print(generations[i] + "\t" + query)
