import functools
import re
from nltk.corpus import stopwords

from model_base import *

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class article_profile_classifier(api_model):
    """
    Model to match the writing style of an article to one of few categories
    Required files:
      Weights: "profile_weights.h5"
      Tokenizer: "profile_tokenizer.pickle"
    """

    def __init__(self, debug=True):
        self.name = "Article Profile Classifier"
        self.debug = debug
        self.model = load_model("profile_weights.h5")
        self.model._make_predict_function()
        self.classes = ["fake", "satire", "opinion piece",
                        "conspiracy", "state", "junk science", "hate speech", "clickbait",
                        "unreliable", "political", "reliable"]
        with open('profile_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name +
                      " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def run_self_test(self):
        # leave in a simple test to see if the model runs
        # also to take a quick benchmark to test performance
        input_sequence = self.preprocess(
            ["Hello there this is to cache the stops words I think not really sure why else got error when threaded lmao"])
        self.model.predict(input_sequence)
        return True

    @functools.lru_cache(maxsize=512, typed=False)
    def predict(self, input_data):
        input_sequence = self.preprocess(input_data)
        preds = self.model.predict(input_sequence)
        pred = preds.argmax(axis=-1)
        output = self.classes[pred[0]]
        return output

    def clean_text(self, text):
        output = ""
        text = str(text).replace("\n", "")
        text = re.sub(r'[^\w\s]', '', text).lower().split(" ")
        for word in text:
            if word not in stopwords.words("english"):
                output = output + " " + word
        return output.strip().replace("  ", " ")

    def preprocess(self, input_data):
        input_string = self.clean_text(input_data)
        input_token = self.tokenizer.texts_to_sequences([input_string])
        processed_input = pad_sequences(
            input_token, padding='post', maxlen=(200))
        return processed_input


class subjectivity_classifier(api_model):
    """
    Model to detect subjective language in an article
    Required files:
      Tokenizer: "subjective_tokenizer.pickle"
      Weights: "subjective_weights.h5"
    """

    def __init__(self, debug=True):
        self.name = "Subjectivity Classifier"
        self.debug = debug
        self.model = load_model("subjective_weights.h5")
        with open("subjective_tokenizer.pickle", "rb") as handle:
            self.tokenizer = pickle.load(handle)
        self.model._make_predict_function()
        self.MAX_SEQUENCE_LENGTH = 40
        self.classes = ["subjective", "objective"]
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name +
                      " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def run_self_test(self):
        # leave in a simple test to see if the model runs
        # also to take a quick benchmark to test performance
        try:
            test = ["Hello this is a random statement",
                    "Hello this is another statement"]
            test = np.array(test, dtype=object)[:, np.newaxis]
            self.model.predict(test)
            return True
        except Exception as e:
            return str(e)

    @functools.lru_cache(maxsize=512, typed=False)
    def predict(self, input_data):
        batch = self.preprocess(input_data)
        preds_list = self.model.predict_on_batch(batch)
        output = np.asarray([0, 0])
        for pred in preds_list:
            output = output + pred
        return output.tolist()

    def preprocess(self, input_data):
        text = re.sub(r'[^\w\s]', ' ', input_data.lower()).replace(
            "\n", " ").replace("  ", " ").strip()
        [text] = self.tokenizer.texts_to_sequences([text])
        batch = []
        count = 0
        seq = []
        for i, word in enumerate(text):
            if count > self.MAX_SEQUENCE_LENGTH-1:
                count = 0
                batch.append(seq)
                seq = []
            elif int(word) > 0:
                seq.append(word)
                count += 1
            elif i == len(text)-1:
                if len(seq) > 5:
                    batch.append(seq)

        batch = pad_sequences(batch, padding='post',
                              maxlen=(self.MAX_SEQUENCE_LENGTH))
        return batch


class clickbait_detector(api_model):
    """
    Model to detect if the article title is clickbait
    Required files:
      Weights: "clickbait_weights.h5"
      Tokenizer: "clickbait_tokenizer.pickle"
    """

    def __init__(self, debug=False):
        print(" * [i] Loading model...")
        self.name = "Clickbait Detector"
        self.model = load_model("clickbait_weights.h5")
        self.model._make_predict_function()
        with open('clickbait_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.classes = ["not_clickbait", "clickbait"]
        self.debug = debug
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name +
                      " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def run_self_test(self):
        print(" * [i] Performing self-test...")
        try:
            # warm-up run
            test_string = self.preprocess(
                "32 ways to test a server. You won't believe no. 3!")
            self.model.predict(test_string)
            # benchmark run
            start = time.time()
            test_string = self.preprocess(
                "99 ways to wreck a paper. You will believe no. 4!")
            self.model.predict(test_string)
            print(" * [i] Server can process ", round(1 /
                                                      (time.time()-start), 1), "predictions per second")
            return True
        except Exception as e:
            print(" * [!] An error has occured:")
            print(e)
            return False

    @functools.lru_cache(maxsize=512, typed=False)
    def predict(self, input_data):
        processed_input = self.preprocess(input_data)
        preds = self.model.predict(processed_input)
        pred = preds.argmax(axis=-1)

        output = self.classes[pred[0]]

        if self.debug:
            print(output)

        return output

    def preprocess(self, input_data):
        input_string = str(input_data).lower()
        input_string = re.sub(r'[^\w\s]', '', input_string)

        input_token = self.tokenizer.texts_to_sequences([input_string])
        output_t = pad_sequences(input_token, padding='pre', maxlen=(15))
        processed_input = pad_sequences(output_t, padding='post', maxlen=(20))

        if self.debug:
            print(" * [d] Cleaned string", input_string)
            print(" * [d] Test sequence", processed_input)

        return processed_input
