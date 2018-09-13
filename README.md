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

### Input and Response

Example url: `https://www.theburningplatform.com/2018/09/03/mccain-the-hero-nearly-sunk-an-aircraft-carrier-killed-134-sailors/`

Input: `
http://ip-address:5000/predict?article_url=https://www.theburningplatform.com/2018/09/03/mccain-the-hero-nearly-sunk-an-aircraft-carrier-killed-134-sailors/
`

Response:

```
{
    "article_profile": "opinion piece",
    "article_subjectivity": [5.828659430146217, 1.171340538567165],
    "article_title": "McCAIN THE HERO NEARLY SUNK AN AIRCRAFT CARRIER & KILLED 134 SAILORS",
    "claimReview": {
        "authors": ["PolitiFact", "Snopes.com"],
        "dates": ["2017-08-07", "2011-08-11"],
        "titles": ["U.S. Sen. John McCain \"was singlehandedly responsible for starting a fire on (the) USS Forrestal aircraft carrier.\"", "Photographs show a new Chinese aircraft carrier."],
        "urls": ["http://www.politifact.com/punditfact/statements/2017/aug/07/blog-posting/posts-blame-john-mccain-deadly-1967-fire-aboard-us/", "https://www.snopes.com/fact-check/chinese-aircraft-carrier/"],
        "verdicts": ["Pants on Fire", "Unproven"]
    },
    "clickbait": "clickbait",
    "hoax_image_search": [],
    "success": true
}
```
