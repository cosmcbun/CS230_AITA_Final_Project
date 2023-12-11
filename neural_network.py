from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from imblearn.over_sampling import SMOTE
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

def normalizeVectors(sample, data):
    average_vector = np.average(data, axis = 0)
    std_vector = np.std(data, axis = 0)
    # l = len(average_vector)
    # average_vector = np.reshape(average_vector, (l,1))
    # std_vector = np.reshape(std_vector, (l,1))
    # print(sample.shape, average_vector.shape, std_vector.shape)
    for i in range(sample.shape[0]): #wasnt working when i tried using np in one-line. oh well
        sample[i] = (sample[i] - average_vector)/std_vector 
    return sample
    # return (sample - average_vector)/std_vector

def createNeuralNetwork(vector_size, nodes_per_layer = 50, dropout = 0.4):
    model = keras.Sequential()
    model.add(keras.Input(shape=(vector_size)))

    # three dense layers
    model.add(Dense(nodes_per_layer, name = "layer1", activation="relu"))
    model.add(Dropout(dropout, seed=15))
    model.add(Dense(nodes_per_layer, name = "layer2", activation="relu"))
    model.add(Dropout(dropout, seed=16))
    model.add(Dense(nodes_per_layer, name = "layer3", activation="relu"))
    model.add(Dropout(dropout, seed=17))

    # add the output layer
    model.add(Dense(1, name = "output_layer", activation="sigmoid"))

    # let's set up the optimizer!
    model.compile(optimizer = 'adam', loss = BinaryCrossentropy(), metrics=['accuracy'])

    return model

# tests by taking the average of all vectors in the body
def createModelOneInstantiation(wantOtherModelData, word2Vec, training, validation, testing, epochs = 200, nodes_per_layer = 50, dropout = 0.4, balancing = "undersample"):
    if balancing == "undersample":
        training = balanceOutcomes(training)

    train_ds = postsToAverageVectors(training, word2Vec)
    validation_ds = postsToAverageVectors(validation, word2Vec)
    test_ds = postsToAverageVectors(testing, word2Vec)

    if balancing == "smote":
        train_ds = smoteBalancing(train_ds)

    neural_net = createNeuralNetwork(100, nodes_per_layer, dropout)
    history_basic_model = neural_net.fit(train_ds[0], train_ds[1], epochs = epochs, validation_data = validation_ds)
    if wantOtherModelData: return neural_net, history_basic_model, train_ds, validation_ds, test_ds
    else: return neural_net

def modelOne(word2Vec, training, validation, testing, epochs=200, nodes_per_layer=50, dropout=0.4, balancing="undersample"):
    print("MODEL ONE")
    neural_net, history_basic_model, train_ds, validation_ds, test_ds = createModelOneInstantiation(True, word2Vec, training, validation, testing, epochs, nodes_per_layer, dropout, balancing)

    print("TRAIN")
    custom_confusion_matrix(neural_net, train_ds)
    print("VALID")
    min_valid_accuracy = custom_confusion_matrix(neural_net, validation_ds)
    print("TEST")
    custom_confusion_matrix(neural_net, test_ds)

    # print(history_basic_model.history)
    eval = neural_net.evaluate(test_ds[0], test_ds[1])
    print(eval)
    return min_valid_accuracy

# tests against specific identified words
def modelTwoRevised(keywords, training, validation, testing, epochs = 200, nodes_per_layer = 50, dropout = 0.4, balancing = "undersample"):
    if balancing == "undersample":
        training = balanceOutcomes(training)
    
    train_ds = postsToBooleans(training, keywords)
    validation_ds = postsToBooleans(validation, keywords)
    test_ds = postsToBooleans(testing, keywords)

    if balancing == "smote":
        train_ds = smoteBalancing(train_ds)

    neural_net = createNeuralNetwork(len(keywords), nodes_per_layer, dropout)
    neural_net.fit(train_ds[0], train_ds[1], epochs = epochs, validation_data = validation_ds)

    print("TRAIN")
    custom_confusion_matrix(neural_net, train_ds)
    print("VALID")
    min_valid_accuracy = custom_confusion_matrix(neural_net, validation_ds)
    print("TEST")
    custom_confusion_matrix(neural_net, test_ds)

    # print(history_basic_model.history)
    eval = neural_net.evaluate(test_ds[0], test_ds[1])
    print(eval)
    return min_valid_accuracy



# tests against specific identified words
def modelTwo(training, validation, testing, nodes_per_layer):
    print("MODEL TWO")
    training = balanceOutcomes(training)
    validation = balanceOutcomes(validation)

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
def modelThree(word2Vec, training, validation, testing, nodes_per_layer, num_words):
    print("MODEL THREE")
    training = balanceOutcomes(training)

    keywords = get_highest_magnitude_words(word2Vec, num_words)

    train_ds = postsToBooleans(training, keywords)
    validation_ds = postsToBooleans(validation, keywords)
    test_ds = postsToBooleans(testing, keywords)
    
    neural_net = createNeuralNetwork(len(keywords), nodes_per_layer)

    history_basic_model = neural_net.fit(train_ds[0], train_ds[1], epochs=5, validation_data = validation_ds)
    eval = neural_net.evaluate(test_ds[0], test_ds[1])
    print(eval)

    # testing_y = [label.numpy() for _, label in test_ds.unbatch().take(-1)]
    # newresults = model.predict_on_batch(test_ds)
    results = np.argmax(neural_net.predict(test_ds[0]), axis=1)
    cm = tf.math.confusion_matrix(test_ds[1], results)
    print(cm)
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

# inspired by https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
def smoteBalancing(training_data):
    oversample = SMOTE()
    training_data = oversample.fit_resample(training_data[0], training_data[1])
    return training_data

def get_self_selected_words():
    #change to lemmatized versions
    lemmatizer = WordNetLemmatizer()
    politics = ["war", "politics", "democrat", "republican", "liberal", "conservative", "progressive", "socialist", "communist", "fascist", "trigger", "sensitive", "cancel", "immigrant", "problematic", "vegetarian", "vegan", "vote", "election", "trump", "obama", "clinton", "biden"]
    relationship_and_gender = ["wife", "husband", "boyfriend", "girlfriend", "gay", "straight", "man", "woman", "boy", "girl", "sex", "lesbian", "lgbt", "divorce", "marry", "orphan", "adopt", "cheat", "mom", "mother", "dad", "father", "son", "daughter", "uncle", "aunt", "cousin", "brother", "sister"]
    social_dynamic = ["beer", "alcohol", "drunk", "high", "marijuana", "weed", "drug", "birthday", "party", "nerd", "jock", "school", "college", "university", "friend", "bully", "text", "instagram", "facebook", "snapchat", "twitter", "game", "geek", "joke", "prank"]
    emotions = ["love", "hate", "emotion", "anger", "happy", "sad", "yell", "scream", "shout", "cry", "tear", "afraid", "fear", "lonely", "guilt", "shame", "laugh"]
    misc = ["work", "job", "fired", "dog", "cat", "edit"]
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

def get_highest_magnitude_words(model, num_words):
    words = model.wv.index_to_key
    words = sorted(words, key=lambda word: word_to_magnitude(word, model), reverse=True)
    return words[0:num_words]

def split_test_data_by_outcome(test_ds):
    yta_posts = []
    yta_outputs = []
    nta_posts = []
    nta_outputs = []
    for i in range(len(test_ds[0])):
        if test_ds[1][i]:
            yta_posts.append(test_ds[0][i])
            yta_outputs.append(test_ds[1][i])
        else:
            nta_posts.append(test_ds[0][i])
            nta_outputs.append(test_ds[1][i])
    test_yta = (np.array(yta_posts), np.array(yta_outputs))
    test_nta = (np.array(nta_posts), np.array(nta_outputs))
    return test_yta, test_nta

# for some odd reason, model.predict isn't working while model.evaluate does
# so I'm coding my own confusion matrix; we get the same result ultimately
def custom_confusion_matrix(model, test_ds):
    test_yta, test_nta = split_test_data_by_outcome(test_ds)
    len_yta = len(test_yta[0])
    len_nta = len(test_nta[0])
    accuracy = model.evaluate(test_ds[0], test_ds[1])
    yta_accuracy = model.evaluate(test_yta[0], test_yta[1])[1]
    nta_accuracy = model.evaluate(test_nta[0], test_nta[1])[1]
    print("Predict Y, True Y:", round(len_yta * yta_accuracy))
    print("Predict N, True Y:", round(len_yta - (len_yta * yta_accuracy)))
    print("Predict N, True N:", round(len_nta * nta_accuracy))
    print("Predict Y, True N:", round(len_nta - (len_nta * nta_accuracy)))
    print("General Accuracy:", accuracy[1])

    print(yta_accuracy, nta_accuracy)
    return min(yta_accuracy, nta_accuracy)

# making sure the initial model doesn't heavily favor YTA or NTA. not actually needed
def tune_weight_initialization(model, train_ds, vector_size, nodes_per_layer):
    accuracy = model.evaluate(train_ds[0], train_ds[1])[1]
    print("ACCURACY", accuracy)
    if accuracy > 0.55 or accuracy < 0.45:
        model = createNeuralNetwork(vector_size, nodes_per_layer)
        return tune_weight_initialization(model, train_ds)
    return model
