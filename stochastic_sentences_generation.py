import pandas as pd
from random import choices
from random import choice
from nltk.tokenize import word_tokenize

# choisi un mot dans une liste de possibilité. Cette sélectino ce fait en fonction d'une répartition de probabilité
# la probabilité est calculée en fonction du nombre d'occurence de chaque n grams élevée à un certain poids pour favoriser les mots avec plus d'occurences
def get_predict_words(possible, weight = 1):
    words = []
    proba = []

    for i in possible:
        words.append(i[0])
        proba.append(i[1]**weight)
    if(proba == []):
        return -1
    else:
        return choices(words, k = 1, weights = proba)[0]

# retourne l'index du mot dans la liste de vocubulaire
def get_index(voc, word):
    result = -1
    for i in range(len(voc)):
        if(voc[i] == word):
            result = i
            break

    return result

# retourne la probabilité de génération de chaque mot
def get_proba(possible):
    tot = 0
    for i in possible:
        tot = tot + i[1]

    result = []
    for i in possible:
        result.append([i[0], i[1]/tot])

    return result

# retourne le prochain mot de la phrase en cours de génération
def predict_next(words, data, display= False):
    tmp = words.copy()
    words = []
    for i in tmp:
        words.append(i.lower())
    if(display):
        print("--------------")
        print(words)
        if(len(words) == 1):
            print("bigram")
        if(len(words) == 2):
            print("trigram")
    possible = []
    possible_voc = []
    for i in data:
        tmp = []
        for j in range(len(words)):
            tmp.append(i[j])

        if(tmp == words):
            if(i[len(i)-2] not in possible_voc):
                possible.append([i[len(i)-2], i[len(i)-1]])
                possible_voc.append(i[len(i)-2])

    if(display):
        print(possible)
    return possible

# génère le début de la phrase en utilisant des trigrams ou des bigrams qui ont déjà éé employé en début de phrases
def sentence_generation_n_grams_start(data_start, start, display = False, weight = 1):
    if(display):
        print("START")
    result = start
    currents = [start.lower()]

    if(len(data_start[0])-2 == len(word_tokenize(start))):
        current = get_predict_words(get_proba(predict_next(currents, data_start)), weight = weight)
    else:
        current = -1

    if (current != -1):
        currents.append(current)
        result = result + " " + current

    return result

# génère toutes les phrases en utilisant des brigams et trigams (la fontione cherche dans les trigrams en premiers,et si il n'y a pas de trigrams compatile, elle cherche parmis les bograms)
def sentence_generation_n_grams(data, start, length = -1, display = False, weight = 1):
    stop = [".", "!"]
    result = start

    currents = word_tokenize(start)
    if(display):
        print(currents)

    current = ""
    end_sentence = False

    if (length != -1):
        for i in range(length - 2):
            for j in range(len(data)):
                if(len(currents) > len(data)):
                    del currents[0]
                current = get_predict_words(get_proba(predict_next(currents, data[j])), weight = weight)
                if(current in stop):
                    result = result + " " + current
                    end_sentence = True
                    break

                if(current != -1):
                    currents.append(current)
                    result = result + " " + current
                    break
                elif(len(currents) == (len(data)-j)):
                    del currents[0]

            if(end_sentence):
                break

    else:
        while(current not in stop):
            for j in range(len(data)):
                if(len(currents) > len(data)):
                    del currents[0]
                current = get_predict_words(get_proba(predict_next(currents, data[j])), weight=weight)
                if(current != -1):
                    currents.append(current)
                    result = result + " " + current
                    break
                else:
                    del currents[0]

    return result

def sentence_generation(data, start_data, start, length = -1, display = False, weight = 1):
    return sentence_generation_n_grams(data, sentence_generation_n_grams_start(start_data, start, display = display, weight=weight), length, display = display, weight=weight)

print("###############################################################################################################")
print("##################################################### MAIN ####################################################")
print("###############################################################################################################")

print("Extraction des bigrams étape 1")

# charge les bigrams sur tout le corpus de données
df = pd.read_csv("/home/ubuntu/M2S2/defi/Fichiers csv/2_grams0.csv")
partie1 = df["Partie1"].values
partie2 = df["Partie2"].values
sup = df["Support"]

data_bi = []

for i in range(len(partie1)):
    data_bi.append([partie1[i], partie2[i], sup[i]])

print("Extraction des bigrams étape 2")

# charge les bigrams de début de phrases sur tout le corpus de données
df = pd.read_csv("/home/ubuntu/M2S2/defi/Fichiers csv/2_grams_begin.csv")
partie1 = df["Partie1"].values
partie2 = df["Partie2"].values
sup = df["Support"]

data_bi_start = []

for i in range(len(partie1)):
    data_bi_start.append([partie1[i], partie2[i], sup[i]])

print("Extraction des trigrams étape 1")

# charge les trigrams de la motié des fichiers disponibles
df = pd.read_csv("/home/ubuntu/M2S2/defi/Fichiers csv/fusion_0_1.csv")
partie1 = df["Partie1"].values
partie2 = df["Partie2"].values
partie3 = df["Partie3"].values
sup = df["Support"]

data_tri = []

for i in range(len(partie1)):
    data_tri.append([partie1[i], partie2[i], partie3[i], sup[i]])

print("Extraction des trigrams étape 2")

# charge les trigrams de début de phrases de la motié des fichiers disponibles
df = pd.read_csv("/home/ubuntu/M2S2/defi/Fichiers csv/fusion_begin_0_1.csv")
partie1 = df["Partie1"].values
partie2 = df["Partie2"].values
partie3 = df["Partie3"].values
sup = df["Support"]

data_tri_start = []

for i in range(len(partie1)):
    data_tri_start.append([partie1[i], partie2[i], partie3[i], sup[i]])

print("Fin des extractions")

# liste des mots possible en tant que début de phrase
starts = ["Je", "Tu", "Il", "Elle", "On", "Nous", "Vous", "Ils", "Elles", "Le", "La", "Les", "Un", "Une", "Des", "Qui", "Que", "Quoi", "Où"]

weights = [1, 3, 5, 10]
num_generation = 20
lengths = [5, 10, 15]

for l in lengths:
    print("Version avec taille fixe ", l)
    print("")
    for j in weights:
        print("poids de ", j)
        for i in range(num_generation):
            print(sentence_generation([data_tri, data_bi],data_bi_start, choice(starts),  length=l, weight = j))
    print("")
print("Version sans taille fixe")
print("")
for j in weights:
    print("poids de ", j)
    for i in range(num_generation*len(lengths)):
        print(sentence_generation([data_tri, data_bi], data_bi_start, choice(starts), weight = j))
