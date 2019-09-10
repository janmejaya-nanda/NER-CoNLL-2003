import argparse

from constants import NERModel, BATCH_SIZE, N_EPOCH
from constants import TEST_DATA_PATH
from model import BiLSTM
from pre_processing import PreProcessor
from utils import get_all_ner_models, generate_data_by_batch, custom_f1, custom_precision, custom_recall


class Test:

    def train_bilstm(self):
        # Load Data
        pre_processor = PreProcessor(file_path=TEST_DATA_PATH)
        sentences, entities = pre_processor.run()

        n_test_data = len(sentences)

        test_generator = generate_data_by_batch(
            x=sentences,
            y=entities,
            n_classes=pre_processor.n_entities + 1,
            entity_to_index=pre_processor.entity_to_index,
            batch_size=BATCH_SIZE
        )

        bilstm = BiLSTM(n_class=pre_processor.n_entities + 1)
        bilstm.load()

        # Saving model with `model.save()` doesn't store custom loss or metrics function. Model has to be stored
        # separately into "config" and "weight" file and loaded from both. This causes an essential step of compiling
        #  before evaluating. I think this issue exist from keras 2.0.
        # https://github.com/keras-team/keras/issues/5916
        bilstm.model.compile(
            optimizer="nadam",
            loss="categorical_crossentropy",
            metrics=["accuracy", custom_f1, custom_precision, custom_recall]
        )
        bilstm.model.evaluate_generator(
            test_generator,
            steps=n_test_data//BATCH_SIZE,
            verbose=1,
        )

    def run(self, model):
        if model == NERModel.bilstm:
            self.train_bilstm()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=get_all_ner_models(), help='Provide a Model name to train')
    args = parser.parse_args()

    Test().run(model=args.model)
