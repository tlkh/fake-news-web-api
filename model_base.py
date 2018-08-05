print(" * [i] Loading ML modules...")

import keras
import pickle
import re
import time
import numpy as np
import pandas as pd

from keras.layers import *
from keras.models import Model, load_model

print(" * [i] System Keras version is", keras.__version__)


class api_model(object):
    """A structure for a model to be used with the Flask web server"""

    def __init__(self, debug=True):
        self.name = "Model's Name"
        self.debug = debug
        # override with procedure to load model
        # leave the debug option open
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
        # wrap the model.predict function here
        # it is a good idea to just do the pre-processing here also
        return NotImplementedError

    def preprocess(self, input_data):
        # preprocessing function
        return NotImplementedError
