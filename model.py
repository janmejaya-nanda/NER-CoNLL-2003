from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

from constants import MAX_WORD_NUMBER
from constants import LSTM_MODEL_PATH
from constants import WORD_VECTOR_SIZE
from utils import custom_f1
from utils import custom_precision
from utils import custom_recall


class BaseModel:
    def __init__(self, model_path):
        """
        All other model should be derived from it. Basic functionality of all models(like saving, loading etc)
        should be defined in base class. This will maintain uniformity across all models.
        """
        self.model = None
        self.saved_model_path = model_path

    def save(self):
        if not self.model:
            raise Exception
        # Currently this way of saving model doesn't support custom metrics.
        # self.model.save(self.saved_model_path + "/ner.h5")

        # Save train model
        model_json = self.model.to_json()
        with open(self.saved_model_path + '/ner.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.saved_model_path + "/ner.h5")
        print("Saved model to {0}".format(self.saved_model_path))

    def load(self):
        # Currently this way of loading model doesn't load custom metrics.
        # self.model = keras.models.load_model(self.saved_model_path + "/ner.h5")

        with open(self.saved_model_path + '/ner.json') as json_file:
            json_config = json_file.read()
        self.model = keras.models.model_from_json(json_config)
        self.model.load_weights(self.saved_model_path + "/ner.h5")


class BiLSTM(BaseModel):
    def __init__(self, n_class):
        super(BiLSTM, self).__init__(LSTM_MODEL_PATH)
        self.n_class = n_class
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(
            units=42,
            return_sequences=True,
            recurrent_dropout=0.25,
            dropout=0.50,
        ), input_shape=(MAX_WORD_NUMBER, WORD_VECTOR_SIZE)))
        self.model.add(Bidirectional(LSTM(
            units=42,
            return_sequences=True,
            recurrent_dropout=0.25,
            dropout=0.50,
        ), input_shape=(MAX_WORD_NUMBER, WORD_VECTOR_SIZE)))
        self.model.add(TimeDistributed(Dense(self.n_class, activation="softmax")))

        self.model.compile(
            optimizer="nadam",
            loss="categorical_crossentropy",
            metrics=["accuracy", custom_f1, custom_precision, custom_recall]
        )
