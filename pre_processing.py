import html.parser
import gc
import re
from itertools import islice

import nltk
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences

from constants import DataFrameHeaders, MAX_WORD_NUMBER, TARGET_PADDING, INPUT_PADDING
from utils import get_wordnet_pos


class PreProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.n_entities = 0
        self.entity_to_index = None

    @staticmethod
    def clean_word(word):
        # Escape HTML char if present
        html_parser = html.parser.HTMLParser()
        word = html_parser.unescape(word)

        # Remove all unnecessary special character and number
        word = re.sub('[^A-Za-z ]+', '', word)

        # convert to lower case and remove spaces
        word = word.lower().strip()

        return word

    def load_and_clean_data(self):
        all_sentences, named_entities = [], []
        with open(self.file_path) as train_file:
            words, entities, unique_entities = [], [], set()
            for line in islice(train_file, 2, None):
                # read the file from 2nd line
                word = line.split(' ')[0]
                named_entity = line.split(' ')[-1].strip('\n')

                word = self.clean_word(word)

                if line in ('\n', '\r\n'):
                    # end of a sentence
                    all_sentences.append(' '.join(words))
                    named_entities.append(' '.join(entities))
                    unique_entities |= set(entities)
                    words, entities = [], []
                else:
                    if word:
                        # Performing Word Lemmatization on text
                        word_lemmatizer = WordNetLemmatizer()
                        word, typ = nltk.pos_tag(word_tokenize(word))[0]

                        typ = get_wordnet_pos(typ)
                        if typ:
                            lemmatized_word = word_lemmatizer.lemmatize(word, typ)
                        else:
                            lemmatized_word = word_lemmatizer.lemmatize(word)

                        words.append(lemmatized_word)
                        entities.append(named_entity)

        self.n_entities = len(unique_entities)
        self.entity_to_index = {t: i for i, t in enumerate(unique_entities)}
        self.df = pd.DataFrame(
            data={DataFrameHeaders.sentence: all_sentences, DataFrameHeaders.named_entity: named_entities})

    def run(self):
        self.load_and_clean_data()

        # Perform padding
        self.df[DataFrameHeaders.sentence] = self.df[DataFrameHeaders.sentence].apply(lambda x: x.split(' '))
        sentences = pad_sequences(
            sequences=self.df[DataFrameHeaders.sentence].tolist(),
            maxlen=MAX_WORD_NUMBER,
            dtype=object,
            padding='post',
            truncating='post',
            value=INPUT_PADDING
        )
        self.df[DataFrameHeaders.named_entity] = self.df[DataFrameHeaders.named_entity].apply(lambda x: x.split(' '))
        entities = pad_sequences(
            sequences=self.df[DataFrameHeaders.named_entity].tolist(),
            maxlen=MAX_WORD_NUMBER,
            dtype=object,
            padding='post',
            truncating='post',
            value=TARGET_PADDING
        )

        del self.df
        gc.collect()

        return sentences, entities
