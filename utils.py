import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from nltk.corpus import wordnet

from constants import NERModel
from constants import MAX_WORD_NUMBER
from constants import GOOGLE_WORD_TO_VEC_PATH

word_vectors = KeyedVectors.load_word2vec_format(GOOGLE_WORD_TO_VEC_PATH, binary=True, limit=None)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def get_all_ner_models():
    obj = NERModel()
    return [getattr(obj, attr) for attr in dir(obj) if not attr.startswith("__")]


def generate_data_by_batch(x, y, n_classes, entity_to_index, batch_size, shuffle=True):
    len_data = len(x)
    start_index = 0

    while True:
        if start_index + batch_size > len_data:
            start_index = 0
            if shuffle:
                perm = np.arange(len_data)
                np.random.shuffle(perm)
                x = x[perm]
                y = y[perm]

        word2vec_array = np.zeros([batch_size, MAX_WORD_NUMBER, 300], np.float32)
        for batch_idx, sentence in enumerate(x[start_index:start_index + batch_size]):
            for word_idx, word in enumerate(sentence):
                # Get google word2vector
                try:
                    word2vec_array[batch_idx, word_idx] = word_vectors[word]
                except Exception as exc:
                    word2vec_array[batch_idx, word_idx] = np.random.uniform(low=-0.25, high=0.25, size=(1, 300))

        # TODO: find a better way to one_hot encode
        batch_y = y[start_index: start_index + batch_size]
        t = [[entity_to_index[word or 'O'] for word in sentence] for sentence in batch_y]
        target = np.array([to_categorical(i, num_classes=n_classes) for i in t])

        start_index += batch_size
        yield (word2vec_array, target)


def custom_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def custom_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def custom_f1(y_true, y_pred):
    precision = custom_precision(y_true, y_pred)
    recall = custom_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
