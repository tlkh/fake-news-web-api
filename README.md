# Fake News Web API

This provides the backend endpoint for the [Fake News Chrome Extension](https://github.com/tlkh/fake-news-chrome-extension).

## Instructions

### Running the server

Running the command `python3 server.py` will start the backend Flask server. It will bind to all interfaces (localhost, 127.0.0.1, external IP etc.) at port 5000 by default.

**System Package Dependencies**

* `python3-dev libxml2-dev libxslt-dev libjpeg-dev zlib1g-dev libpng12-dev`
* NOTE: If you find problem installing `libpng12-dev`, try installing `libpng-dev`.

**Python Package Dependencies**

* `pip3 install -r requirements.txt`
* TensorFlow
* Required NLTK corpora: `curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3`

