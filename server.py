print(" * [i] Loading Python modules...")
import time
import numpy as np
import flask
import urllib3
from newspaper import Article
from threading import Thread

from claimreview import ClaimReview
cr = ClaimReview()

print(" * [i] Loading NLP models...")
from model_nlp import *

app = flask.Flask(__name__)

model_clickbait = None
model_profile = None
model_subj = None

urllib3.disable_warnings()
np.warnings.filterwarnings('ignore')

def load_ML():
    print(" * [i] Building Keras models")
    global model_clickbait, model_profile, model_toxic, model_subj

    model_subj = subjectivity_classifier()
    model_clickbait = clickbait_detector()

    # article profile/type classifier
    model_profile = article_profile_classifier()

load_ML()

def pred_clickbait(input_):
    global model_clickbait, data
    data["clickbait"] = model_clickbait.predict(input_)


def pred_profile(input_):
    global model_profile, data
    data["article_profile"] = model_profile.predict(input_)


def pred_subj(input_):
    global model_subj, data
    data["article_subjectivity"] = model_subj.predict(input_)


data = {"success": False}

@functools.lru_cache(maxsize=512, typed=False)
def download_article(article_url):
    article = Article(article_url, fetch_images=False)
    article.download()
    article.parse()

    article_title = article.title
    article_text = article.text
    image_list = article.images

    return article_title, article_text, image_list

@app.route("/predict", methods=["POST"])
def predict():
        # initialize the data dictionary that will be returned from the
        # view
    global model_clickbait, model_profile, model_subj
    global data, cr

    data = {"success": False}

    # get the respective args from the post request
    if flask.request.method == "POST":
        start_time = time.time()
        article_url = flask.request.args.get("article_url")

        article_title, article_text, image_list = download_article(article_url)
        
        article_time = time.time()
        print(" * [i] Article download time:", round(article_time-start_time, 3), "seconds")

        threads = []

        if article_text is not None:
            t = Thread(target=pred_profile, args=([article_text]))
            threads.append(t)
            t.start()

            t = Thread(target=pred_subj, args=([article_text]))
            threads.append(t)
            t.start()

        if article_title is not None:
            article_title = article_title.replace("%20", " ")
            print(" * [i] Incoming article title:", article_title)
            data["article_title"] = article_title

            t = Thread(target=pred_clickbait, args=([article_title]))
            threads.append(t)
            t.start()

            data["claimReview"] = cr.search_fc(article_title)

        if image_list is not None:
            results = []
            # TODO
            data["hoax_image_search"] = results

        [t.join() for t in threads]
        data["success"] = True

        print(" * [i] Inference took", round(time.time()-article_time, 3), "seconds")
        print(" * [i] Request took", round(time.time()-start_time, 3), "seconds")

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if file was executed by itself, start the server process
if __name__ == "__main__":
    print(" * [i] Starting Flask server")
    app.run(host='0.0.0.0', port=5000)
