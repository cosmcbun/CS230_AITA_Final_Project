from main import getAllPosts, getWord2Vec
from neural_network import createModelOneInstantiation
import os, pickle, datetime

def getNeuralNetwork():
    if os.path.isfile("neural_network.pickle"):
        return pickle.load(open("neural_network.pickle", "rb"))
    elif os.path.isfile("aita_clean.csv"):
        start_time = datetime.datetime.now()
        print("Alright, got the data. We will first read the data and save the resulting data structure.")
        all_posts = getAllPosts()
        training_set = all_posts[:80000]
        validation_set = all_posts[80000:90000]
        testing_set = all_posts[90000:]
        print(f"Post compilation complete after {datetime.datetime.now() - start_time} seconds")
        print("Now we will compile the word vector model.")
        word2Vec = getWord2Vec(training_set)
        print(f"Word2vec model compilation complete after {datetime.datetime.now() - start_time} seconds")
        print("Lastly, we compile the neural network.")
        neural_network = createModelOneInstantiation(False, word2Vec, training_set, validation_set, testing_set, 200,
                                                     100, 0.4, "undersample")
        pickle.dump(neural_network, open("neural_network.pickle", "wb"))
        return neural_network
    else:
        print("It looks like this is the first time you've run this program on your computer. You'll first need the data set in the same directory, which can be found here: https://iterative.ai/blog/a-public-reddit-dataset#how-to-get-the-dataset")
        print("Note that this program may take a while to run for the first time, and will fill the directory up to around 1 gigabyte with cached data.")
        quit()

    neural_network.modelOne(word2Vec, training_set, validation_set, testing_set, 200, 100, 0.4, "undersample")

def renderJudgment(ahole_val):
    HARDCODED_AHOLE_PERCENTILE_VALS = [0.4858675, 0.4973156, 0.4984976, 0.49924588, 0.49978673, 0.5002333, 0.50062394,
                                       0.500941, 0.5012192, 0.5014622, 0.5016853, 0.50190926, 0.50211155, 0.50229716,
                                       0.5024698, 0.5026424, 0.50280184, 0.5029518, 0.50310445, 0.50324863, 0.5033819,
                                       0.5035142, 0.50364834, 0.50377387, 0.50389296, 0.5040094, 0.50412357, 0.50423443,
                                       0.5043442, 0.5044556, 0.50455964, 0.50465906, 0.50476396, 0.50486606, 0.504971,
                                       0.5050711, 0.5051642, 0.5052613, 0.5053596, 0.50545335, 0.5055459, 0.50563735,
                                       0.5057272, 0.50581586, 0.5059095, 0.50599736, 0.5060883, 0.50617546, 0.5062685,
                                       0.5063645, 0.5064572, 0.5065496, 0.50663805, 0.5067233, 0.50681317, 0.50690156,
                                       0.5069938, 0.5070853, 0.5071774, 0.50726295, 0.50735515, 0.50745213, 0.5075471,
                                       0.5076454, 0.50773996, 0.5078362, 0.5079348, 0.50803185, 0.5081241, 0.50822306,
                                       0.50832254, 0.5084301, 0.50853544, 0.5086416, 0.5087538, 0.508864, 0.5089799,
                                       0.50909597, 0.50922006, 0.50934565, 0.50947, 0.50960267, 0.5097382, 0.50987875,
                                       0.51002675, 0.51018935, 0.51035315, 0.5105321, 0.5107253, 0.510922, 0.5111434,
                                       0.5113786, 0.5116402, 0.5119272, 0.51225203, 0.5126265, 0.51305795, 0.51362216,
                                       0.5144165, 0.51574045, 0.52941465]
    if ahole_val < .5:
        print("Model Predicts: You're in the wrong here. YTA.")
    else:
        print("Model Predicts: You're in the clear! NTA.")

    if ahole_val < HARDCODED_AHOLE_PERCENTILE_VALS[0]:
        print("In fact, you're worse than anyone in our dataset of almost 100,000 posts!")
    elif ahole_val > HARDCODED_AHOLE_PERCENTILE_VALS[-1]:
        print("In fact, you're better than anyone in our dataset of almost 100,000 posts!")
    else:
        for i in range(101):
            if ahole_val < HARDCODED_AHOLE_PERCENTILE_VALS[i]:
                if str(i)[-1] == "1": numeral_ending = "st"
                elif str(i)[-1] == "2": numeral_ending = "nt"
                elif str(i)[-1] == "3": numeral_ending = "rd"
                else: numeral_ending = "th"
                print(f"Overall, you're in the {i}{numeral_ending} percentile for not-the-a-hole-ness in our data.")
                break

neural_network = getNeuralNetwork()
solicit_text
create_post_object
ahole_val = evaluate_post_with_NN
renderJudgment(ahole_val)