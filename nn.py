from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers.legacy import Adam

from logger import Logger

logger = Logger()


class NeuralNetwork:
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        label_col: str,
        study: str,
        k=5,
        epochs=20,
    ):
        self._data = data
        self._labels = labels
        self.label_col = label_col
        self._results: Dict[str, History] = {}
        self.k = k
        self.epochs = epochs
        self.history = {}
        self.study: str = study

    @logger.log
    def _build_model(self, input_shape: Tuple):
        model = keras.Sequential()
        model.add(keras.layers.Dense(512, input_shape=input_shape, activation="relu"))
        model.add(keras.layers.Dense(512, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=[keras.metrics.Precision(), keras.metrics.BinaryAccuracy()],
        )
        return model

    k = 4
    num_epochs = 100

    @staticmethod
    def show_results(history: History, code: Optional[str] = None):
        if code:
            print(f" - Results for {code} - ".center(40))
        for k, v in history.history.items():
            print(f"{k} : {v[-1]}")

    def results(self):
        history = list(self.history.values())[0]
        text = ""
        # for k, v in history.history.items():
        #     text += f"{k} : {v[-1]} \n"
        text += f"{self.study},NN,{history.history['binary_accuracy'][-1]},{history.history['loss'][-1]},"
        return text

    @logger.log
    def _k_fold_validation(self, folds: int, epochs: int, study: str):
        train_data = self._data
        train_labels = self._labels
        (
            partial_train_data,
            val_data,
            partial_train_labels,
            val_labels,
        ) = train_test_split(train_data, train_labels)
        model = self._build_model((train_data.shape[1],))
        num_val_samples = len(train_data) // 4
        all_scores = list()
        for i in range(folds):
            # Prepare the validation data: data from partition i
            a, b = i * num_val_samples, (i + 1) * num_val_samples
            val_data = train_data[a:b]
            val_targets = train_labels[a:b]

            partial_train_data = np.concatenate(
                [
                    train_data[: i * num_val_samples],
                    train_data[(i + 1) * num_val_samples :],
                ],
                axis=0,
            )
            partial_train_targets = np.concatenate(
                [
                    train_labels[: i * num_val_samples],
                    train_labels[(i + 1) * num_val_samples :],
                ],
                axis=0,
            )
            history: History = model.fit(
                partial_train_data.astype(np.float32),
                partial_train_targets,
                batch_size=1,
                verbose=0,
                epochs=epochs,
                validation_data=(val_data.astype(np.float32), val_targets),
            )
            ## NeuralNetwork.show_results(history)
            all_scores.append(history.history)
        return history

    @logger.log
    def run(self, study: str) -> History:
        history = self._k_fold_validation(self.k, self.num_epochs, study)
        self.history[study] = history
        return history
