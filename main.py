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
        #print([w for w in self.allText.split()])
        #print([w for w in ''.join(filter(lambda x: x in printable, self.allText)).split(" ")])
        #print([w for w in nltk.tokenize.word_tokenize(''.join(filter(lambda x: x in printable, self.allText)))])
        lemmatizer = WordNetLemmatizer()

        for word in ''.join(filter(lambda x: x in printable, self.allText)).split(" "):
            if word and word not in stop_words and "www" not in word:
                self.tokens.append(lemmatizer.lemmatize(word))
        #print(self.tokens)
        #print()


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
    if os.path.isfile("model.pickle"):
        return pickle.load(open("model.pickle", "rb"))
    else:
        model = trainWord2Vec(word_set)
        pickle.dump(model, open("model.pickle", "wb"))
        return model

VECTOR_SIZE = 100

start_time = datetime.datetime.now()
all_posts = getAllPosts()
training_set = all_posts[:80000]
validation_set = all_posts[80000:90000]
testing_set = all_posts[90000:]
print("Post compilation complete")
print(datetime.datetime.now()-start_time)

word2Vec = getWord2Vec(training_set)
print("Model compilation complete")
print(datetime.datetime.now() - start_time)

neural_network.modelOne(word2Vec, training_set, validation_set, testing_set, 50)
# neural_network.modelTwo(training_set, validation_set, testing_set, 25)
# do lemmatization for these models

# Highest magnitude
# Selected by us
# Difference in frequency
# Highest impacts words of certain frequency