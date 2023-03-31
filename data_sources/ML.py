import operator
import pickle
import re
from collections import Counter
from collections import defaultdict
from itertools import islice
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
def clean_text(text):
    no_special_car = re.sub("[^\w\s]", ' ', text)
    no_special_car = re.sub("\s+", " ", no_special_car)
    no_special_car = no_special_car.strip()
    return no_special_car


def cbf_language_model(profile, query):
    cleaned_profile = clean_text(profile)
    cleaned_query = clean_text(query)
    token_profile = word_tokenize(cleaned_profile)
    token_query = word_tokenize(cleaned_query)
    stem_profile = [w for w in token_profile if w not in stopwords.words("english")]
    stem_query = [w for w in token_query if w not in stopwords.words("english")]
    wordDict = Counter(stem_profile)
    profile = []
    for k in wordDict.keys():
        wordDict[k] /= len(stem_profile)
        profile.append(wordDict[k])
    vect = []
    # vect size is the same as the profile
    for el in wordDict.keys():
        if el in stem_query:
            vect.append(wordDict[el])
        else:
            vect.append(0)
    return vect, profile


def get_relevant_items(username, data_test):
    list_art = data_test[data_test.username == username]
    return list(set(list(list_art.id)))


def get_training_data(username, data):
    list_art = data[data.username == username]
    list_articles = list(list_art.title)
    return list(set(list_articles))


def ml(chemin_data_train, chemin_data_test, nb):
    user_recommendations = defaultdict(list)
    data_train = pd.DataFrame(pd.read_csv(chemin_data_train))
    data_test = pd.DataFrame(pd.read_csv(chemin_data_test))
    # with open(chemin_data_test, "rb") as file:
    #     data_test = pickle.load(file)
    list_titles = []
    article_tag = defaultdict(list)
    for index, element in data_test.iterrows():
        list_titles.append((element[2], element[4]))
    list_titles = list(set(list_titles))
    article_ids = list(data_test.id)
    for a in article_ids:
        d = data_test.loc[data_test["id"] == a]
        article_tag[a] = list(d.tags)
    precision_d = defaultdict(float)
    recall_d = defaultdict(float)
    list_titles = list(set(list_titles))
    set_users = set(list(data_test.username))
    co = 0
    for u in set_users:
        print(co)
        if co == nb:
            break
        user_profile = get_training_data(u, data_train)
        score_dictionary = defaultdict(float)
        relevant_items = get_relevant_items(u, data_test)
        string_profile = " ".join(user_profile)
        for e in list_titles:
            Id_article = e[0]
            article = e[1]
            article += " ".join(article_tag[Id_article])
            vect, profile = cbf_language_model(string_profile, str(article))
            x = np.array(vect)
            profile = np.array(profile)
            profile = profile.reshape(1, -1)
            x = x.reshape(1, -1)
            score_dictionary[Id_article] = cosine_similarity(profile, x)
        top_n = dict(sorted(score_dictionary.items(), key=operator.itemgetter(1), reverse=True))
        # recommend and compute recall and precision
        for n in np.arange(10, 30, 5):
            recommendations = dict(islice(top_n.items(), int(n)))
            if n == 25:
                user_recommendations[u] = recommendations.keys()
            truly_recommended = [e for e, v in recommendations.items() if e in relevant_items]
            precision_d[n] += len(truly_recommended) / n
            recall_d[n] += len(truly_recommended) / len(relevant_items)
        co += 1
    cpt = 0
    for ko in np.arange(10, 30, 5):
        recall_d[ko] = recall_d[ko] / co
        precision_d[ko] = precision_d[ko] / co
        cpt += 1
    return user_recommendations, recall_d, precision_d