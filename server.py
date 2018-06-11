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
    global model_clickbait, model_hoaximage, model_articleprofile
    model_clickbait = clickbait_detector()
    model_hoaximage = hoax_image_search()
    model_articleprofile = article_profile_classifier()


def download_image(url):
    try:
        url = str(url)
        print(' * [i] --- Trying to get', url)
        http = urllib3.PoolManager()
        response = http.request('GET', url, timeout=3)
        image_data = response.data
    except Exception as e:
        print(" * [!] --- Warning: Could not download image", url)
        print(" * [!] --- ", e)
        return False

    try:
        pil_image = Image.open(BytesIO(image_data)).convert('RGB')
        output = np.array(pil_image)[:, :, ::-1].copy()
    except Exception as e:
        print(" * [!] --- Warning: Failed to parse image", url)
        print(" * [!] --- ", e)
        return False

    return output


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
                NotImplementedError
                data["article_profile"] = NotImplemented

            if image_list is not None:
                results = []
                print(" * [i] Incoming image list")
                image_list = image_list.split(",")
                for image_url in image_list:
                    print(" * [+] >>", image_url)
                    image = download_image(image_url)
                    pred = model_hoaximage.predict(image)
                    print(" * [+] -->", pred)
                    results.append(pred)

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
