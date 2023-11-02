import pandas as pd
import datetime

def createPost(row):
    global posts
    posts.append(Post(row))

class Post:
    def __init__(self, row):
        self.id, self.timestamp, self.title, self.body, self.edited, self.verdict, self.score, self.num_comments, self.is_asshole = row

        self.convertTimestamp()
        self.removeAITAFromTitle()
        self.extractDemographics()
        self.stripFunctionalWords()

    def convertTimestamp(self):
        self.time = datetime.datetime.fromtimestamp(self.timestamp)

    def removeAITAFromTitle(self):
        if " " not in self.title: return
        firstWord, remainder = self.title.split(" ", 1)
        if "AITA" in firstWord.upper(): self.title = remainder

    def extractDemographics(self):
        self.age, self.gender = None, None
        pass

    def stripFunctionalWords(self):
        pass

    def __repr__(self):
        return self.id

data = pd.read_csv("aita_clean.csv")
posts = []
data.apply(createPost, axis=1)
print(posts[:5])
