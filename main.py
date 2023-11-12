import pandas as pd
import datetime, re
import nltk
import warnings
import pickle
import os

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

def createPost(row):
    global posts
    posts.append(Post(row))

class Post:
    def __init__(self, row):
        self.id, self.timestamp, self.title, self.body, self.edited, self.verdict, self.score, self.num_comments, self.is_asshole = row

        self.convertTimestamp()
        self.removeAITAFromTitle()
        self.fillEmptyFields()
        self.extractDemographics()
        self.stripFunctionalWords()
        self.cleanBody()
        self.tokenize()

    def convertTimestamp(self):
        self.time = datetime.datetime.fromtimestamp(self.timestamp)

    def removeAITAFromTitle(self):
        if " " not in self.title: return
        firstWord, remainder = self.title.split(" ", 1)
        if "AITA" in firstWord.upper(): self.title = remainder

    def fillEmptyFields(self):
        if type(self.body) != str:
            self.body = ""

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
            results = re.finditer(regex_str, self.title + " " + self.body)
            for result in results:
                self.age = result.group(1)
                self.gender = re.findall("[0-9]+", result.group(0))[0]
                return
        
    def stripFunctionalWords(self):
        pass

    def cleanBody(self):
        self.body = self.body.lower().replace("\n"," ")

    def tokenize(self):
        #wnl = nltk.stem.WordNetLemmatizer()
        #tokens = nltk.tokenize.word_tokenize(self.body)
        #for t in tokens:
        #    print(wnl.lemmatize(t, pos="v"))
        pass

    def __repr__(self):
        return self.title

def concatAllPosts(posts):
    return "".join([p.body for p in posts])

def trainWord2Vec(posts):
    tokenized_data = []
    for i in nltk.tokenize.sent_tokenize(concatAllPosts(posts)):
        temp = []
        for j in nltk.tokenize.word_tokenize(i):
            temp.append(j)
        tokenized_data.append(temp)

    model = gensim.models.Word2Vec(tokenized_data, min_count=1, vector_size=100, window=5, sg=1)
    return model

start_time = datetime.datetime.now()
if os.path.isfile("posts.pickle"):
    posts = pickle.load(open("posts.pickle", "rb"))
    print("Post loading complete")
else:
    data = pd.read_csv("./aita_clean.csv")
    posts = []
    data.apply(createPost, axis=1)
    print("Post compilation complete")
    pickle.dump(posts, open("posts.pickle", "wb"))

print(datetime.datetime.now()-start_time)
print(len(posts))

if os.path.isfile("model.pickle"):
    model = pickle.load(open("model.pickle", "rb"))
    print("Model loading complete")
else:
    model = trainWord2Vec(posts)
    print("Model compilation complete")
    pickle.dump(model, open("model.pickle", "wb"))

print(datetime.datetime.now()-start_time)

print("Cosine similarity between 'h' " +
      "and 'w' - Skip Gram : ",
      model.wv.similarity('lose', 'lifestyle'))
print("Cosine similarity between 'h' " +
      "and 'p' - Skip Gram : ",
      model.wv.similarity('the', 'a'))
print(model.wv)
