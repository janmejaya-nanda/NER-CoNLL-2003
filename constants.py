TRAIN_DATA_PATH = './data/train.txt'
VALIDATION_DATA_PATH = './data/valid.txt'
TEST_DATA_PATH = './data/test.txt'
GOOGLE_WORD_TO_VEC_PATH = './data/embedding/GoogleNews-vectors-negative300.bin'
LSTM_MODEL_PATH = './output/saved_model/LSTM'

MAX_WORD_NUMBER = 52
WORD_VECTOR_SIZE = 300

INPUT_PADDING = '<pad>'
TARGET_PADDING = 'O'

# Hyper-parameters
BATCH_SIZE = 128
N_EPOCH = 12


class DataFrameHeaders:
    sentence = "Sentence"
    named_entity = "Named Entity"


class NERModel:
    bilstm = 'bilstm'
