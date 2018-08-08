from model_base import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords


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
        return output.strip().replace("  ", " ")

    def preprocess(self, input_data):
        input_string = self.clean_text(input_data)
        input_token = self.tokenizer.texts_to_sequences([input_string])
        processed_input = pad_sequences(
            input_token, padding='post', maxlen=(400))
        return processed_input


import tensorflow as tf
import tensorflow_hub as hub

from keras import regularizers, initializers, optimizers, callbacks
from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Model
from keras import backend as K


class subjectivity_classifier(api_model):
    """
    Model to detect subjective language in an article
    Required files:
      Weights: "subjective_weights.h5"
    """

    def __init__(self, debug=True):
        self.name = "Subjectivity Classifier"
        self.debug = debug
        self.model = self.build_model()
        self.model.load_weights("subjective_weights.h5")
        self.model._make_predict_function()
        self.MAX_SEQUENCE_LENGTH = 35
        self.classes = ["subjective", "objective"]
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name +
                      " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def build_model(self):
        """Text Classification with pre-trainedELMo embeddings from TF Hub"""
        sess = tf.Session()
        K.set_session(sess)

        elmo_model = hub.Module(
            "https://tfhub.dev/google/elmo/2", trainable=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        sequence_input = Input(shape=(1,), dtype=tf.string)
        embedded_sequences = Lambda(lambda x: elmo_model(tf.squeeze(tf.cast(
            x, tf.string)), signature="default", as_dict=True)["default"], output_shape=(1024,))(sequence_input)
        embedded_sequences = Reshape((1024, 1,))(embedded_sequences)
        l_drop = Dropout(0.4)(embedded_sequences)
        l_flat = Flatten()(l_drop)
        l_dense = Dense(32, activation='relu')(l_flat)
        preds = Dense(2, activation='softmax')(l_dense)
        model = Model(sequence_input, preds)

        model.compile(loss='binary_crossentropy',
                      optimizer="adam", metrics=['acc'])

        return model

    def run_self_test(self):
        # leave in a simple test to see if the model runs
        # also to take a quick benchmark to test performance
        test = ["Hello this is a random statement",
                "Hello this is another statement"]
        test = np.array(test, dtype=object)[:, np.newaxis]
        preds_list = self.model.predict(test)
        return True

    def predict(self, input_data):
        batch = self.preprocess(input_data)
        preds_list = self.model.predict(batch)
        output = np.asarray([0, 0])
        for pred in preds_list:
            output = output + pred
        return output.tolist()

    def preprocess(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower()).replace(
            "\n", " ").replace("  ", " ")
        text = "".join([c for c in text if (c.isalpha() or c == " ")])
        text = text.split(" ")
        batch = []
        count = 0
        seq = ""
        for i, word in enumerate(text):
            if count > self.MAX_SEQUENCE_LENGTH-1:
                count = 0
                batch.append(seq)
                seq = ""
            elif (word.isalpha() and word != " "):
                seq = seq + word + " "
                count += 1
            elif i == len(text)-1:
                if len(seq) > 5:
                    batch.append(seq)
        return np.array(batch, dtype=object)[:, np.newaxis]


class toxic_classifier(api_model):
    """
    Model to detect hateful language in an article
    Required files:
      Weights: "toxic_weights.h5"
      Tokenizer: "toxic_tokenizer.pickle"
    """

    def __init__(self, debug=True):
        self.name = "Toxicity Classifier"
        self.debug = debug
        self.model = load_model("toxic_weights.h5")
        self.model._make_predict_function()
        self.MAX_SEQUENCE_LENGTH = 128
        self.classes = ["toxic", "severe_toxic",
                        "obscene", "threat", "insult", "identity_hate", "none"]
        with open('toxic_tokenizer.pickle', 'rb') as handle:
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
        preds_list = self.model.predict(input_sequence)
        output = []
        for pred in preds_list:
            for i, class_pred in enumerate(pred):
                if class_pred > 0.6:
                    if self.classes[i] not in output:
                        output.append(self.classes[i])

        if len(output) < 1:
            return ["none"]
        else:
            return output

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower()).replace(
            "\n", " ").replace("  ", " ")
        text = "".join([c for c in text if (c.isalpha() or c == " ")])
        text = text.split(" ")
        batch = []
        count = 0
        seq = ""
        for i, word in enumerate(text):
            if count > 110:
                count = 0
                batch.append(seq)
                seq = ""
            elif (word.isalpha() and word != " " and word not in stopwords.words("english")):
                seq = seq + word + " "
                count += 1
            elif i == len(text)-1:
                if len(seq) > 5:
                    batch.append(seq)
        return batch

    def preprocess(self, input_data):
        input_string = self.clean_text(input_data)
        input_token = self.tokenizer.texts_to_sequences(input_string)
        processed_input = pad_sequences(
            input_token, padding='pre', maxlen=(self.MAX_SEQUENCE_LENGTH-3))
        processed_input = pad_sequences(
            processed_input, padding='post', maxlen=(self.MAX_SEQUENCE_LENGTH))
        return processed_input


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
