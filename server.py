print(" * [i] Loading Python modules...")
import numpy as np
import flask

app = flask.Flask(__name__)

from models import clickbait_detector

model_clickbait = None

def load_ML():
    print(" * [i] Building Keras models")
    global model_clickbait
    model_clickbait = clickbait_detector()

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    global model_clickbait

    data = {"success": False}

    # get the respective args from the post request
    if flask.request.method == "POST":
        try:
            # arg: article_title (for clickbait detection)
            article_title = flask.request.args.get("article_title")
            article_title = article_title.replace("%20", " ")
            print("Incoming article title:", article_title)

            pred_clickbait = model_clickbait.predict(article_title)

            data["clickbait"] = pred_clickbait

            # indicate that the request was a success
            data["success"] = True

        except Exception as e:
            data["prediction_error"] = str(e)
            data["success"] = False

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if file was executed by itself, start the server process
if __name__ == "__main__":
    load_ML()
    print(" * [i] Starting Flask server")
    app.run(host = '0.0.0.0', port=5000)
