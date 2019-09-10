import argparse
import gc

from constants import NERModel
from constants import TRAIN_DATA_PATH
from constants import TEST_DATA_PATH
from constants import N_EPOCH
from constants import BATCH_SIZE
from constants import VALIDATION_DATA_PATH
from model import BiLSTM
from pre_processing import PreProcessor
from utils import generate_data_by_batch
from utils import get_all_ner_models


class Train:

    def train_bilstm(self):
        # Load train Data
        pre_processor = PreProcessor(file_path=TRAIN_DATA_PATH)
        sentences, entities = pre_processor.run()

        val_pre_processor = PreProcessor(file_path=VALIDATION_DATA_PATH)
        val_sentences, val_entities = val_pre_processor.run()

        n_train_data = len(sentences)
        n_val_data = len(val_sentences)

        train_generator = generate_data_by_batch(
            x=sentences,
            y=entities,
            n_classes=pre_processor.n_entities + 1,
            entity_to_index=pre_processor.entity_to_index,
            batch_size=BATCH_SIZE
        )
        validate_generator = generate_data_by_batch(
            x=val_sentences,
            y=val_entities,
            n_classes=val_pre_processor.n_entities + 1,
            entity_to_index=val_pre_processor.entity_to_index,
            batch_size=BATCH_SIZE
        )

        bilstm = BiLSTM(n_class=pre_processor.n_entities + 1)
        bilstm.model.fit_generator(
            train_generator,
            steps_per_epoch=n_train_data // BATCH_SIZE,
            epochs=N_EPOCH,
            verbose=1,
            validation_data=validate_generator,
            validation_steps=n_val_data // BATCH_SIZE,
        )
        bilstm.save()

        # Free some memory
        del sentences, val_sentences, entities, val_entities
        gc.collect()

        # It's a work around. Ideally testing should be done from `test.py` with loaded model. This temporary
        # work around is adopted due to issue of Loading a model in Keras 2.0.
        # There might be better solution, need to dig more dipper to find it.
        # https://github.com/keras-team/keras/issues/6977
        # TODO: Find a better way to resolve the above issue
        pre_processor = PreProcessor(file_path=TEST_DATA_PATH)
        test_sentences, test_entities = pre_processor.run()

        n_test_data = len(test_sentences)

        test_generator = generate_data_by_batch(
            x=test_sentences,
            y=test_entities,
            n_classes=pre_processor.n_entities + 1,
            entity_to_index=pre_processor.entity_to_index,
            batch_size=BATCH_SIZE
        )

        bilstm.model.evaluate_generator(
            test_generator,
            steps=n_test_data // BATCH_SIZE,
            verbose=1,
        )

    def run(self, model):
        if model == NERModel.bilstm:
            self.train_bilstm()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=get_all_ner_models(), help='Provide a Model name to train')
    args = parser.parse_args()

    Train().run(model=args.model)
