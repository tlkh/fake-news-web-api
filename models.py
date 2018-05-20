print(" * [i] Loading Keras modules...")
import keras
import pickle
import re
import time
import numpy as np
from keras.layers import *
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
print(" * [i] System Keras version is", keras.__version__)

class api_model(object):
    """A structure for a model to be used with the Flask web server"""

    def __init__(self, debug=True):
        self.name = "Model's Name"
        # override with procedure to load model
        # leave the debug option open
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name + " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def run_self_test(self):
        # leave in a simple test to see if the model runs
        # also to take a quick benchmark to test performance
        return True


    def predict(self, input_data):
        # wrap the model.predict function here
        # it is a good idea to just do the pre-processing here also
        return NotImplementedError

    def preprocess(self, input_data):
        # preprocessing function
        return NotImplementedError


class clickbait_detector(api_model):
    """
    Model to detect if the article title is clickbait
    Required files:
      Weights: "clickbait_weights.h5"
      Tokenizer: "clickbait_tokenizer.pickle"
    Usage:
      clickbait.predict()
    """

    def __init__(self, debug=True):
        print(" * [i] Loading model...")
        self.name = "Clickbait Detector"
        self.model = load_model("clickbait_weights.h5")
        with open('clickbait_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.classes = ["not_clickbait", "clickbait"]
        self.debug = debug
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name + " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def run_self_test(self):
        print(" * [i] Performing self-test...")
        try:
            # warm-up run
            test_string = self.preprocess("32 ways to test a server. You won't believe no. 3!")
            self.model.predict(test_string)
            # benchmark run
            start = time.time()
            test_string = self.preprocess("99 ways to wreck a paper. You will believe no. 4!")
            self.model.predict(test_string)
            print(" * [i] Server can process ", round(1/(time.time()-start), 1), "predictions per second")
            return True
        except Exception as e:
            print(" * [!] An error has occured:")
            print(e)
            return False

    def predict(self, input_string):
        processed_input = self.preprocess(input_string)
        preds = self.model.predict(processed_input)
        pred = preds.argmax(axis=-1)

        output = self.classes[pred[0]]

        if self.debug:
            print(output)

        return output

    def preprocess(self, input_string):
        input_string = str(input_string).lower()
        input_string = re.sub(r'[^\w\s]', '', input_string)

        input_token = self.tokenizer.texts_to_sequences([input_string])
        output_t = pad_sequences(input_token, padding='pre', maxlen=(15))
        processed_input = pad_sequences(output_t, padding='post', maxlen=(20))

        if self.debug:
            print(" * [d] Cleaned string", input_string)
            print(" * [d] Test sequence", processed_input)

        return processed_input
