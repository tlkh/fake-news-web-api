print(" * [i] Loading Python modules...")
import numpy as np
import pandas as pd
import re, sys, os, csv, keras, pickle, time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model, load_model
print(" * [i] System Keras version is",keras.__version__)

import flask

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
tokenizer = None
classes = None

def load_ML():
    global model, tokenizer, classes
    model = load_model("best_weights.h5")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    classes = ["not_clickbait", "clickbait"]

    try: 
        start = time.time()
        print(" * [i] Performing self-test....")
        test_string = preprocess("32 ways to test a server. You won't believe no. 3!")
        model.predict(test_string)
        print(" * [i] Model is functioning!")
        print(" * [i] Model took", round(time.time()-start, 4), "seconds / prediction")
    except Exception as e:
        print(" * [!] An error has occured")
        print(e)
        print(" * [i] Exiting...")


def preprocess(input_string):
    global tokenizer
    input_string = str(input_string).lower()
    input_string = re.sub(r'[^\w\s]','', input_string)
    print(" * [d] Cleaned string", input_string)
    input_token = tokenizer.texts_to_sequences([input_string])
    output_t = pad_sequences(input_token, padding='pre', maxlen=(15))
    output = pad_sequences(output_t, padding='post', maxlen=(20))
    print(" * [d] Test sequence", output)
    return output

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    global model, classes

    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        test_string = flask.request.args.get("article_title")
        test_string = test_string.replace("%20", " ")
        print("Incoming article title:", test_string)
        test_string = preprocess(test_string)

        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model.predict(test_string)
        pred = preds.argmax(axis=-1)
        print(pred)
        data["predictions"] = [classes[pred[0]]]

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(" * [i] Loading Keras model")
    load_ML()
    print(" * [i] Starting Flask server")
    app.run(host = '0.0.0.0', port=5000)
