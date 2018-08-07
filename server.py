print(" * [i] Loading Python modules...")
import time
import numpy as np
import flask
import urllib3
from PIL import Image
from io import BytesIO

from newspaper import Article

app = flask.Flask(__name__)

from model_nlp import *

model_clickbait = None
model_profile = None

urllib3.disable_warnings()


def load_ML():
    print(" * [i] Building Keras models")
    global model_clickbait, model_profile

    # clickbait title detector
    model_clickbait = clickbait_detector()

    # article profile classifier
    model_profile = article_profile_classifier()


@app.route("/predict", methods=["POST"])
def predict():
        # initialize the data dictionary that will be returned from the
        # view
    global model_clickbait, model_profile

    data = {"success": False}

    # get the respective args from the post request
    if flask.request.method == "POST":
        start_time = time.time()
        try:
            # retrieve parameters from arguments
            article_url = flask.request.args.get("article_url")

            article = Article(article_url, fetch_images=False)
            article.download()
            article.parse()

            article_title = article.title
            article_text = article.text
            image_list = article.images

            if article_title is not None:
                article_title = article_title.replace("%20", " ")
                print(" * [i] Incoming article title:", article_title)
                pred_clickbait = model_clickbait.predict(article_title)
                data["clickbait"] = pred_clickbait
                data["article_title"] = article_title

            if article_text is not None:
                pred_profile = model_profile.predict(article_text)
                data["article_profile"] = pred_profile

            if image_list is not None:
                results = []
                # TODO

                data["hoax_image_search"] = results

            data["success"] = True

        except Exception as e:
            data["prediction_error"] = str(e)
            data["success"] = False
            print(" * [!]", e)

        print("Request took", int(time.time()-start_time), "seconds")

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if file was executed by itself, start the server process
if __name__ == "__main__":
    load_ML()
    print(" * [i] Starting Flask server")
    app.run(host='0.0.0.0', port=5000)
