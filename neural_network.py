from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from nltk.stem import WordNetLemmatizer
import numpy as np

def postsToAverageVectors(posts, model):
    inputs = np.ndarray((len(posts), 100))
    for i, post in enumerate(posts):
        inputs[i] = getAverageTokenVector(post, model)
    # inputs = np.array([getAverageTokenVector(post, model) for post in posts])
    outputs = np.array([post.is_asshole for post in posts])
    return inputs, outputs

def postsToBooleans(posts, keywords):
    vector_size = len(keywords)
    inputs = np.ndarray((len(posts), vector_size))
    for i, post in enumerate(posts):
        boolean_vector = np.array([word in post.tokens for word in keywords])
        inputs[i] = boolean_vector        
    outputs = np.array([post.is_asshole for post in posts])
    return inputs, outputs

def getAverageTokenVector(post, model):
    tokens = post.tokens
    vectors = np.asarray([(model.wv[word] if word in model.wv else np.zeros(100)) for word in tokens])
    return np.mean(vectors, axis=0)

def createNeuralNetwork(vector_size, nodes_per_layer):
    model = keras.Sequential()
    model.add(keras.Input(shape=(vector_size)))

    # three dense layers
    model.add(Dense(nodes_per_layer, name = "layer1", activation="relu"))
    model.add(Dense(nodes_per_layer, name = "layer2", activation="relu"))
    model.add(Dense(nodes_per_layer, name = "layer3", activation="relu"))

    # add the output layer
    model.add(Dense(1, name = "output_layer", activation="sigmoid"))

    # let's set up the optimizer!
    model.compile(optimizer = 'adam', loss = BinaryCrossentropy(), metrics=['accuracy'])

    return model

# tests by taking the average of all vectors in the body
def modelOne(word2vec, training, validation, testing, nodes_per_layer):
    print("MODEL ONE")
    training = balanceOutcomes(training)
    # validation = balanceOutcomes(validation)
    # testing = balanceOutcomes(testing)

    train_ds = postsToAverageVectors(training, word2vec)
    validation_ds = postsToAverageVectors(validation, word2vec)
    test_ds = postsToAverageVectors(testing, word2vec)
    
    neural_net = createNeuralNetwork(100, nodes_per_layer)

    history_basic_model = neural_net.fit(train_ds[0], train_ds[1], epochs=50, validation_data = validation_ds)
    # print(history_basic_model.history)
    eval = neural_net.evaluate(test_ds[0], test_ds[1])
    print(eval)

# tests against specific identified words
def modelTwo(training, validation, testing, nodes_per_layer):
    print("MODEL TWO")
    training = balanceOutcomes(training)

    # keywords = ["mom", "girlfriend", "mother", "dad", "edit"]
    keywords = get_self_selected_words()
    # keywords = get_select_selected_high_freq_words(training, 500)

    train_ds = postsToBooleans(training, keywords)
    validation_ds = postsToBooleans(validation, keywords)
    test_ds = postsToBooleans(testing, keywords)
    
    neural_net = createNeuralNetwork(len(keywords), nodes_per_layer)

    history_basic_model = neural_net.fit(train_ds[0], train_ds[1], epochs=50, validation_data = validation_ds)
    eval = neural_net.evaluate(test_ds[0], test_ds[1])
    print(eval)
    # print(history_basic_model.history)

# tests against highest magnitude words
def modelThree(word2vec, training, validation, testing, nodes_per_layer):
    print("MODEL THREE")
    training = balanceOutcomes(training)

    keywords = get_highest_magnitude_words(word2vec)

    train_ds = postsToBooleans(training, keywords)
    validation_ds = postsToBooleans(validation, keywords)
    test_ds = postsToBooleans(testing, keywords)
    
    neural_net = createNeuralNetwork(len(keywords), nodes_per_layer)

    history_basic_model = neural_net.fit(train_ds[0], train_ds[1], epochs=50, validation_data = validation_ds)
    eval = neural_net.evaluate(test_ds[0], test_ds[1])
    print(eval)
    # print(history_basic_model.history)


# assumes there are more NTA than YTA, and that posts are already randomly ordered
# in the future, duplicate rather than delete
def balanceOutcomes(posts):
    balanced_posts = []
    spare_yta_posts = []
    more_nta = 0
    for post in posts:
        if post.is_asshole:
            spare_yta_posts.append(post)
            more_nta += 1
        elif more_nta > 0:
            balanced_posts.append(post)
            balanced_posts.append(spare_yta_posts.pop(0))
            more_nta -= 1
    return balanced_posts

def get_self_selected_words():
    #change to lemmatized versions
    lemmatizer = WordNetLemmatizer()
    politics = ["war", "politics", "democrat", "republican", "liberal", "conservative", "progressive", "socialist", "communist", "fascist", "trigger", "sensitive", "cancel", "immigrant", "problematic", "vegetarian", "vegan", "vote", "election", "trump", "obama", "clinton", "biden"]
    relationship_and_gender = ["wife", "husband", "boyfriend", "girlfriend", "gay", "straight", "man", "woman", "boy", "girl", "sex", "lesbian", "lgbt", "divorce", "marry", "orphan", "adopt", "cheat", "mom", "mother", "dad", "father", "son", "daughter", "uncle", "aunt", "cousin", "brother", "sister"]
    social_dynamic = ["beer", "alcohol", "drunk", "high", "marijuana", "weed", "drug", "birthday", "party", "nerd", "jock", "school", "college", "university", "friend", "bully", "text", "instagram", "facebook", "snapchat", "twitter", "game", "geek"]
    emotions = ["love", "hate", "emotion", "anger", "happy", "sad", "yell", "scream", "shout", "cry", "tear", "afraid", "fear", "lonely", "guilt", "shame"]
    misc = ["work", "job", "fired", "dog", "cat"]
    all_words = politics + relationship_and_gender + social_dynamic + emotions + misc
    return [lemmatizer.lemmatize(word) for word in all_words]

def get_select_selected_high_freq_words(posts, threshold):
    all_words = get_self_selected_words()
    all_tokens = []
    words_to_remove = []
    for post in posts:
        all_tokens += post.tokens
    for word in all_words:
        count = all_tokens.count(word)
        if count < threshold:
            words_to_remove.append(word)
    for word in words_to_remove:
        all_words.remove(word)
    return all_words

def word_to_magnitude(word, model):
    return np.linalg.norm(model.wv[word])

def get_highest_magnitude_words(model):
    words = model.wv.index_to_key
    words = sorted(words, key=lambda word: word_to_magnitude(word, model), reverse=True)
    return words[0:100]