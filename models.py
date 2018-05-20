#print(" * [i] Loading Keras modules...")
import keras
import pickle
import re
import time
import numpy as np
from keras.layers import *
from keras.models import Model, load_model
from keras.applications import vgg16
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import cv2
import pandas as pd
from matplotlib.pyplot import imshow  # for debugging on jupyter
import matplotlib.pyplot as plt  # for debugging on jupyter
from PIL import Image  # for debugging on jupyter
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

class hoax_image_search(api_model):
    '''
    Given url through POST, output history of the file
    Required files:
        'index_subimage.csv'
    Usage:
        search()
    '''

    def __init__(self, debug=True):
        print("loading model")
        self.model = self.init_model()
        csv_filename = 'index_subimage.csv'
        self.df, self.feature_vectors = load_feature_vectors(csv_filename)
 
    def init_model(self):
        ''' 
        Returns initialised model 
        Probably change to Xception something, but need update index_subimage.csv
        '''
        model = vgg16.VGG16(weights='imagenet', include_top=True)
    
        model.layers.pop()
        model.layers.pop()
    
        new_layer = Dense(10, activation='softmax', name='my_dense')
    
#        inp = model.input
#        out = new_layer(model.layers[-1].output)
        
        return model     
       
    def load_feature_vectors(self, csv_filename):
        '''
        We load 'index_subimage.csv' and also parse the feature vectors
        '''
        df = pd.read_csv(csv_filename='index_subimage.csv')
        # df.head()
        
        feature_vectors = df['feature_vector'].apply(lambda x: 
                                   np.fromstring(
                                       x.replace('\n','')
                                        .replace('[','')
                                        .replace(']','')
                                        .replace('  ',' '), sep=' '))
        # ref: https://stackoverflow.com/questions/45704999/
        # model2 = Model(inp, out)    
        return df,feature_vectors
    
    
    def get_bounding_boxes(self, img):
        '''
        To dissect incoming image to individual "pictures"
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cp_height, cp_breath = np.shape(gray)
        ret,thresh = cv2.threshold(gray,225,255,cv2.THRESH_BINARY_INV)
    
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #     img_ann = img
        output_boxes = []
        
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w > cp_height/2.5 and h > cp_breath/2.5:
        #       cv2.rectangle(img_ann,(x,y),(x+w,y+h),(0,0,255),3)
                output_boxes.append([x,y,w,h])
        if output_boxes == []:
            output_boxes.append([0,0,cp_breath,cp_height])
        return output_boxes
    
    def analyse_image(self, model, path_image_to_analyse, plotting=False):
        '''
        When the web server catches the URL, it downloads it and saves it.
        The image is saved in path_image_to_analyse
        This function is then called.
        It finds the bounding boxes with get_bounding_boxes()
        For each "subimage" then the feature vector is calculated and compared with the database.
        It returns some results, such as the degree of match, and the source of the original photo (to be added)
        '''
        img = cv2.imread(path_image_to_analyse)
    
        if plotting:
            print("============ FRAME ANALYSED =============")
            imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
            print("----------- NEAREST PICTURES ------------")
        
        output_boxes = self.get_bounding_boxes(img)
        result = []
    
        for i, box in enumerate(output_boxes):
            [x,y,w,h] = box
            output_img = np.array(img[y:y+h, x:x+w])
            imshow(output_img)
            print(np.shape(output_img))
            cv2.imwrite("temp.jpg",output_img)
            imgsearch = self.calc_feature_vector(model, "temp.jpg")
    
            match = [imgsearch.dot(fv) for fv in self.feature_vectors]
            top4 = np.argpartition(match,np.arange(-4,0,1))[-4:][::-1]
            print(top4)
            
        for pic in top4:
            print(pic)
            print("percentage match: {}".format(match[pic]))
            result.append([match[pic], pic])
            
            if plotting:
                imshow(np.asarray(Image.open('database/' + self.df['img_file_name'][pic], 'r')))
                plt.axis("off")
                plt.show()
            
        print("\n\n\n\n\n\n\n\n\n\n")
        
        return result  # will include the original url of the image in the future

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
