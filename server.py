print(" * [i] Loading Python modules...")
import numpy as np
import flask
import urllib3
from PIL import Image
from io import BytesIO

from newspaper import Article

app = flask.Flask(__name__)

from models import *

model_clickbait = None
model_hoaximage = None
model_articleprofile = None

urllib3.disable_warnings()


def load_ML():
    print(" * [i] Building Keras models")
    global model_clickbait, model_articleprofile
    model_clickbait = clickbait_detector()
    model_articleprofile = article_profile_classifier()


@app.route("/predict", methods=["POST"])
def predict():
        # initialize the data dictionary that will be returned from the
        # view
    global model_clickbait, model_hoaximage, model_articleprofile

    data = {"success": False}

    # get the respective args from the post request
    if flask.request.method == "POST":
        try:
            # retrieve parameters from arguments
            article_url = flask.request.args.get("article_url")

            article = Article(article_url)
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

            if article_text is not None:
                print(article_text)
                data["article_profile"] = article_text

            if image_list is not None:
                results = []
                # TODO

                data["hoax_image_search"] = results

            data["success"] = True

        except Exception as e:
            data["prediction_error"] = str(e)
            data["success"] = False
            print(" * [!]", e)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if file was executed by itself, start the server process
if __name__ == "__main__":
    load_ML()
    print(" * [i] Starting Flask server")
    app.run(host='0.0.0.0', port=5000)
