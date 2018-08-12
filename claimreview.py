import re
import json
import functools
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class ClaimReview():

    def __init__(self):
        print(" * [i] Checking NLTK data:")
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('stopwords')
        print("")
        self.lemmatizer = WordNetLemmatizer()

        fc_path = "fact_checks_20180502.txt"
        print(" * [i] Loading ClaimReview Database from:", fc_path)

        with open(fc_path) as f:
            fc_raw = f.readlines()

        self.title_list = []
        self.org_title_list = []
        self.url_list = []
        self.date_list = []
        self.reviewby_list = []

        for fc in tqdm(fc_raw):
            fc = fc.strip("\n")
            fc = fc.replace(
                "</script>", "").replace('<script type="application/ld+json">', "")
            fc = json.loads(fc)
            title = self.norm_text(fc["claimReviewed"])
            url = fc["url"]
            try:
                date_published = fc["datePublished"]
            except Exception as e:
                date_published = "None"
            author = fc["author"]["name"]
            self.title_list.append(title)
            self.org_title_list.append(self.strip_html(fc["claimReviewed"]).replace("\\", "").strip())
            self.url_list.append(url)
            self.reviewby_list.append(author)
            self.date_list.append(date_published)

        print(" * [i] Loaded Claims:", len(self.title_list))

    def strip_html(self, data):
        p = re.compile(r'<.*?>')
        return p.sub('', data)

    def clean_text(self, data):
        text = re.sub(r'[^\w\s]', ' ', data.lower()).replace(
            "\n", "").replace("  ", " ")
        text = "".join([c for c in text if (c.isalpha() or c == " ")])
        text = text.split(" ")
        output = ""
        for word in text:
            if word not in stopwords.words("english"):
                output = output + " " + word
        return output.strip().replace("  ", " ")

    def nltk2wn_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(self, sentence):
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        wn_tagged = map(lambda x: (x[0], self.nltk2wn_tag(x[1])), nltk_tagged)
        res_words = []
        for word, tag in wn_tagged:
            if tag is None:
                res_words.append(word)
            else:
                res_words.append(self.lemmatizer.lemmatize(word, tag))
        return " ".join(res_words)

    def norm_text(self, data):
        raw = self.strip_html(data)
        text = self.clean_text(raw)
        norm_text = self.lemmatize_sentence(text)
        return norm_text

    @functools.lru_cache(maxsize=128, typed=False)
    def search_fc(self, article_title):
        match_title = []
        match_url = []
        match_author = []
        match_date = []

        article_title = self.norm_text(article_title)

        for index, fc_title in tqdm(enumerate(self.title_list), total=len(self.title_list)):
            match = fuzz.token_sort_ratio(article_title, fc_title)
            if match > 55:
                match_title.append(self.org_title_list[index])
                match_url.append(self.url_list[index])
                match_date.append(self.date_list[index])
                match_author.append(self.reviewby_list[index])

        return {"titles": match_title, "urls": match_url, "dates": match_date, "authors": match_author}
