from model_base import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords


class article_profile_classifier(api_model):
    """TODO"""

    def __init__(self, debug=True):
        self.name = "Article Profile Classifier"
        self.debug = debug
        self.model = load_model("profile_weights.h5")
        self.model._make_predict_function()
        self.classes = classes = ["fake", "satire", "bias",
                                  "conspiracy", "state", "junksci", "hate", "clickbait",
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
        return True

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
        return str(output.strip())[1:-3].replace("  ", " ")

    def preprocess(self, input_data):
        input_string = self.clean_text(input_data)
        input_token = self.tokenizer.texts_to_sequences([input_string])
        processed_input = pad_sequences(
            input_token, padding='post', maxlen=(400))
        return processed_input


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
