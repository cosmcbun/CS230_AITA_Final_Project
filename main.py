import pandas as pd
import numpy as np
import datetime, re
import nltk
import neural_network
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import pickle
import os
import random
import string

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

def addPostObjectToList(row, posts):
    post = Post(row)
    if post.allText: posts.append(post)
    if len(posts)%10000 == 0: print(f"Post loading {len(posts)/1000}% complete")

class Post:
    def __init__(self, row):
        self.id, self.timestamp, self.title, self.body, self.edited, self.verdict, self.score, self.num_comments, self.is_asshole = row

        if pd.isnull(self.body): self.allText = self.title.lower()
        else: self.allText = self.title.lower() + " " + self.body.lower().replace("\n"," ")

        self.time = datetime.datetime.fromtimestamp(self.timestamp)
        self.extractDemographics()

        self.tokens = []
        self.tokenize()

    def extractDemographics(self):
        regex_strs = [
            "\[[0-9]+(M|F|NB|m|f|nb)\]",
            "\([0-9]+(M|F|NB|m|f|nb)\)"
            "\[[0-9]+ (M|F|NB|m|f|nb)\]",
            "\([0-9]+ (M|F|NB|m|f|nb)\)",
            " [0-9]+(M|F|NB|m|f|nb) ",
            "\[(M|F|NB|m|f|nb)[0-9]+\]",
            "\((M|F|NB|m|f|nb)[0-9]+\)"
            "\[(M|F|NB|m|f|nb) [0-9]+\]",
            "\((M|F|NB|m|f|nb) [0-9]+\)",
            " (M|F|NB|m|f|nb)[0-9]+ "
        ]
        self.age, self.gender = None, None
        for regex_str in regex_strs:
            results = re.finditer(regex_str, self.allText)
            for result in results:
                self.age = result.group(1)
                self.gender = re.findall("[0-9]+", result.group(0))[0]
                return

    def tokenize(self):
        stop_words = set(stopwords.words('english'))
        printable = set(string.ascii_letters)
        printable.add("'")
        printable.add(" ")
        lemmatizer = WordNetLemmatizer()

        for word in ''.join(filter(lambda x: x in printable, self.allText)).split(" "):
            if word and word not in stop_words and "www" not in word:
                self.tokens.append(lemmatizer.lemmatize(word))

    def __repr__(self):
        return self.title

def concatAllPosts(posts):
    return " ".join([p.allText for p in posts])

def trainWord2Vec(posts):
    tokenized_data = []
    for i in nltk.tokenize.sent_tokenize(concatAllPosts(posts)):
        temp = []
        for j in nltk.tokenize.word_tokenize(i):
            temp.append(j)
        tokenized_data.append(temp)

    model = gensim.models.Word2Vec(tokenized_data, min_count=1, vector_size=100, window=5, sg=1)
    return model

def getAllPosts():
    if os.path.isfile("posts.pickle"):
        return pickle.load(open("posts.pickle", "rb"))
    else:
        data = pd.read_csv("./aita_clean.csv")
        all_posts = []
        data.apply(lambda row: addPostObjectToList(row, all_posts), axis=1)
        random.seed(42)
        random.shuffle(all_posts)
        pickle.dump(all_posts, open("posts.pickle", "wb"))
        return all_posts

def getWord2Vec(word_set):
    if os.path.isfile("word2vec_model.pickle"):
        return pickle.load(open("word2vec_model.pickle", "rb"))
    else:
        model = trainWord2Vec(word_set)
        pickle.dump(model, open("word2vec_model.pickle", "wb"))
        return model

def tuneHyperparameters(word2Vec, training_set, validation_set, testing_set):
    nodes_per_layer = [10, 15, 50, 100]
    dropouts = [0.2, 0.4, 0.6]
    balancing_technique = ["smote", "undersample"]
    results = []
    for n in nodes_per_layer:
        for d in dropouts:
            for b in balancing_technique:
                acc = neural_network.modelOne(word2Vec, training_set, validation_set, testing_set, 50, n, d, b)
                print("FINAL", n, d, b, acc)
                results.append([n, d, b, acc])
    print(results)
    return(results)

def getAholeValues(posts, word2Vec, neural_network):
    avg_vecs = neural_network.postsToAverageVectors(posts, word2Vec)[0]
    ahole_vals = neural_network.predict(x=avg_vecs)[:,0]
    return ahole_vals
    #fun_vals = {posts[i]:ahole_vals[i] for i in range(len(posts))}
    #posts.sort(key=lambda p: posts[p])


VECTOR_SIZE = 100

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    all_posts = getAllPosts()
    training_set = all_posts[:80000]
    validation_set = all_posts[80000:90000]
    testing_set = all_posts[90000:]
    print("Post compilation complete")
    print(datetime.datetime.now()-start_time)

    word2Vec = getWord2Vec(training_set)
    print("Word2vec model compilation complete")
    print(datetime.datetime.now()-start_time)

    #neural_network.modelOne(word2Vec, training_set, validation_set, testing_set, 200, 100, 0.4)
    #ahole_vals = get_ahole_values(all_posts, word2Vec, pickle.load(open("neural_network.pickle", "rb")))

    fun_set = all_posts
    #avg_vecs = neural_network.postsToAverageVectors(fun_set, word2Vec)[0]
    #ahole_vals = network.predict(x=avg_vecs)[:,0]
    ahole_vals = pickle.load(open("sorted_ahole_confidences.pickle", "rb"))



    # with open("export_confidences_to_r.txt", "w") as file:
    #     file.write("confidences\n")
    #     for val in ahole_vals:
    #         file.write(str(val)+"\n")

    # ALL OF THE TRIALS:
    # wordset1 = neural_network.get_self_selected_words()
    # wordset2 = neural_network.get_highest_magnitude_words(word2Vec, 100)
    # wordset3 = neural_network.get_highest_magnitude_words(word2Vec, 500)
    # neural_network.modelTwoRevised(wordset1, training_set, validation_set, testing_set, 50, 100, 0.4, "undersample")
    # neural_network.modelTwoRevised(wordset2, training_set, validation_set, testing_set, 50, 100, 0.4, "undersample")
    # neural_network.modelTwoRevised(wordset3, training_set, validation_set, testing_set, 50, 100, 0.4, "undersample")

    # neural_network.modelOne(word2Vec, training_set, validation_set, testing_set, 200, 100, 0.4, "undersample")
    # tuneHyperparameters(word2Vec, training_set, validation_set, testing_set)
    print(datetime.datetime.now()-start_time)