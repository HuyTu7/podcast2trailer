import json
from typing import Dict, List
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from nltk.corpus import stopwords


def read_data(mypath: str) -> Dict:
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    examples = {}
    for fname in onlyfiles:
        f = open(f'{mypath}{fname}')
        json_f = json.load(f)
        for t in json_f['results']['transcripts']:
            examples[fname.split(".json")[0]] = t['transcript']
    return examples


def rm_stopwords_from_text(text: str) -> str:
    # Remove stopwords from text
    _stopwords = stopwords.words('english')
    text = text.split()
    word_list = [word for word in text if word not in _stopwords]
    return ' '.join(word_list)


def load_word_embeddings() -> Dict:
    word_embeddings = {}
    f = open('glove/glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings


def preprocessing(sentences: List[str]):
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = clean_sentences.str.lower()
    clean_sentences = clean_sentences.apply(rm_stopwords_from_text)
    return clean_sentences
