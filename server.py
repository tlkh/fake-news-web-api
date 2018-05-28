print(" * [i] Loading Python modules...")
import numpy as np
import flask

app = flask.Flask(__name__)

from models import clickbait_detector, hoax_image_search

model_clickbait = None
model_hoaximage = None

def load_ML():
    print(" * [i] Building Keras models")
    global model_clickbait, model_hoaximage
    model_clickbait = clickbait_detector()
    model_hoaximage = hoax_image_search()

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    global model_clickbait, model_hoaximage

    data = {"success": False}

    # get the respective args from the post request
    if flask.request.method == "POST":
        try:

            try: # arg: article_title (for clickbait detection)
                article_title = flask.request.args.get("article_title")
                article_title = article_title.replace("%20", " ")

                print(" * [i] Clickbait functionality")
                print(" * [i] Incoming article title:", article_title)

                pred_clickbait = model_clickbait.predict(article_title)
                data["clickbait"] = pred_clickbait

            except:
                print(" * [i] No article title sent")

            try: # arg: image array
                images_list = flask.request.args.get("images_list")

                print(" * [i] Hoax image search functionality")
                print(images_list)
                print(" * [!] Not implemented yet")

                #for image in images_list:
                    #model_hoaximage.analyse_image(image)

                # indicate that the request was a success

            except:
                print(" * [i] No image list sent")

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
