from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
import pandas as pd
import numpy as np
import datetime, re
import nltk
import warnings
import pickle
import os
import random

VECTOR_SIZE = 100

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

def createPost(row):
    global all_posts
    post = Post(row)
    if post.body: all_posts.append(post)

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
        self.tokens = nltk.tokenize.word_tokenize(self.body)
        #for t in tokens:
        #    print(wnl.lemmatize(t, pos="v"))

    def __repr__(self):
        return self.title

def concatAllPosts(posts):
    return " ".join([p.body for p in posts])

def trainWord2Vec(posts):
    tokenized_data = []
    for i in nltk.tokenize.sent_tokenize(concatAllPosts(posts)):
        temp = []
        for j in nltk.tokenize.word_tokenize(i):
            temp.append(j)
        tokenized_data.append(temp)

    model = gensim.models.Word2Vec(tokenized_data, min_count=1, vector_size=VECTOR_SIZE, window=5, sg=1)
    return model

def getAverageTokenVector(post, model):
    tokens = post.tokens
    # tokens = ["my", "girlfriend", "hates", "me"]#nltk.tokenize.word_tokenize(post.body)
    vectors = np.asarray([(model.wv[word] if word in model.wv else np.zeros(VECTOR_SIZE)) for word in tokens])
    return np.mean(vectors, axis=0)

start_time = datetime.datetime.now()
if os.path.isfile("posts.pickle"):
    all_posts = pickle.load(open("posts.pickle", "rb"))
    print("Post loading complete")
else:
    data = pd.read_csv("./aita_clean.csv")
    all_posts = []
    data.apply(createPost, axis=1)
    random.seed(42)
    random.shuffle(all_posts)
    print("Post compilation complete")
    pickle.dump(all_posts, open("posts.pickle", "wb"))
    print(datetime.datetime.now()-start_time)

for i in range(len(all_posts) - 1, -1, -1):
    if len(all_posts[i].tokens) < 10:
        all_posts.pop(i)

training_set = all_posts[:80000]
validation_set = all_posts[80000:90000]
testing_set = all_posts[90000:]


if os.path.isfile("model.pickle"):
    model = pickle.load(open("model.pickle", "rb"))
    print("Model loading complete")
else:
    model = trainWord2Vec(training_set)
    print("Model compilation complete")
    pickle.dump(model, open("model.pickle", "wb"))
    print(datetime.datetime.now()-start_time)


def create_neural_network():
  model = keras.Sequential()
  model.add(keras.Input(shape=(VECTOR_SIZE)))

  # three dense layers
  model.add(Dense(128, name = "layer1", activation="relu"))
  model.add(Dense(128, name = "layer2", activation="relu"))

  # add the output layer
  model.add(Dense(1, name = "output_layer", activation="sigmoid"))

  # let's set up the optimizer!
  model.compile(optimizer = 'adam', loss = BinaryCrossentropy(), metrics=['accuracy'])

  return model

def posts_to_dataset(posts, model):
    inputs = np.ndarray((len(posts), VECTOR_SIZE))
    for i, post in enumerate(posts):
        inputs[i] = getAverageTokenVector(post, model)
    # inputs = np.array([getAverageTokenVector(post, model) for post in posts])
    outputs = np.array([post.is_asshole for post in posts])
    return inputs, outputs

neural_net = create_neural_network()
print("created neural network!")
train_ds = posts_to_dataset(training_set, model)
validation_ds = posts_to_dataset(validation_set, model)
test_ds = posts_to_dataset(testing_set, model)

print("We got the sets!")

history_basic_model = neural_net.fit(train_ds[0], train_ds[1], epochs=25, validation_data = validation_ds)
print(history_basic_model.history)
# results = neural_net.evaluate(test_ds[0], test_ds[1])
# print(len(results))
# print("Test Loss:", results[0], "Test Accuracy:", results[1])
