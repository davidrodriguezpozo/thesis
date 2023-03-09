# %%

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from tensorflow.keras.callbacks import History

from logger import Logger
from nn import NeuralNetwork
from preprocess import Preprocessor
from tremor import preprocessor as tremor_preprocessor
from voice import preprocessor as voice_preprocessor

logger = Logger()
preprocessors = [tremor_preprocessor, voice_preprocessor]


classifiers = {"RF": RandomForestClassifier, "SVM": SVC}


class Results:
    def __init__(
        self,
        model: Union[RandomForestClassifier, SVC],
        method: str,
        results: np.ndarray,
        actual: np.ndarray,
        x_test: np.ndarray,
        cross_val_results: np.ndarray,
        study: str,
        params: Dict[str, Any],
    ):
        self.model = model
        self.method = method
        self._results = results
        self.actual = actual
        self.X_test = x_test
        self.cross_val_results = cross_val_results
        self.study = study
        self.params = params

    @property
    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self._results, self.actual)

    @property
    def accuracy(self) -> float:
        return accuracy_score(self.actual, self._results)

    def plot(self) -> None:
        print("")
        print(f"Results for {self.method}".center(40))
        print("-------------".center(40))
        print(f"Accuracy: {self.accuracy}".center(40))
        print(f"Confusion matrix:".center(40))
        print(f"Cross validation: {np.mean(self.cross_val_results)}")
        print(f"{self.confusion_matrix}".center(40))

    def results(self) -> str:
        text = ""
        text += f"{self.study},{self.method},{self.accuracy},,{json.dumps(self.params)}"
        # text += f"Results for {self.method} \n"
        # text += "------------- \n"
        # text += f"Accuracy: {self.accuracy} \n"
        # text += f"Cross validation: {np.mean(self.cross_val_results)} \n"
        # text += f"Confusion matrix: \n"
        # text += f"{self.confusion_matrix} \n"
        return text

    def plot_confusion_matrix(self) -> None:
        cm = self.confusion_matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.model.classes_
        )
        disp.plot()
        plt.show()


@logger.log
def train_predict(
    method: str,
    model: Union[RandomForestClassifier, SVC],
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    study: str,
    params: Dict[str, Any],
) -> Results:
    model.fit(x, y)
    results = cross_val_score(model, x, y, cv=10)
    return Results(
        model,
        method,
        model.predict(x_test),
        actual=y_test,
        x_test=x_test,
        cross_val_results=results,
        study=study,
        params=params,
    )


# preprocessor.preprocess(directory)
# results: Dict[str, Dict[str, List[Results]]] = defaultdict(lambda: defaultdict(list))
# nn_results: Dict[str, Any] = {}
# for study in studies:
#     X_train, X_test, y_train, y_test = preprocessor.train_test_split(study)
#     for key, model in classifiers.items():
#         for params in preprocessor.get_parameters()[key]:
#             results[study][key].append(
#                 train_predict(key, model(**params), X_train, y_train, X_test, y_test)
#             )

#     results[study] = {
#         k: sorted(v, key=lambda x: x.accuracy, reverse=True)
#         for k, v in results[study].items()
#     }

#     nn = NeuralNetwork(data=X_train, labels=y_train, label_col="illness")
#     nn_results[study] = nn.run()


# for k, study_results in results.items():
#     logger.print_results(study_results, k)

# for k, study_results in nn_results.items():
#     NeuralNetwork.show_results(study_results, k)


# LOAD VOICE DATASETS


def run_analysis(
    preprocessor: Preprocessor, reduce_dimensions: bool = False
) -> Tuple[Dict[str, Dict[str, List[Results]]], Dict[str, List[NeuralNetwork]]]:
    results: Dict[str, Dict[str, List[Results]]] = defaultdict(
        lambda: defaultdict(list)
    )
    nn_results: Dict[str, Any] = {}
    for study in preprocessor.dataframes:
        print("Study: ", study)
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(
            study, reduce_dimensions=reduce_dimensions
        )
        for key, model in classifiers.items():
            for params in preprocessor.get_parameters()[key]:
                results[study][key].append(
                    train_predict(
                        method=key,
                        model=model(**params),
                        x=X_train,
                        y=y_train,
                        x_test=X_test,
                        y_test=y_test,
                        study=study,
                        params=params,
                    )
                )

        results[study] = {
            k: sorted(v, key=lambda x: x.accuracy, reverse=True)
            for k, v in results[study].items()
        }

        nn = NeuralNetwork(
            data=X_train, labels=y_train, label_col="illness", study=study
        )
        nn.run(study)
        nn_results[study] = [nn]

    # for k, study_results in results.items():
    #     logger.print_results(study_results, k)

    # for k, study_results in nn_results.items():
    #     NeuralNetwork.show_results(study_results, k)

    return results, nn_results


def main(
    filename: str, runs: Optional[int] = 10, reduce_dimensions: bool = False
) -> List[Tuple[Results, History]]:
    results = []
    folder = Path("results")
    runs = runs or 0
    for i in range(runs):
        for prep in preprocessors:
            filepath = Path(
                folder, filename, f"run_{i}", prep.__class__.__name__ + ".csv"
            )
            filepath.parent.mkdir(exist_ok=True, parents=True)
            prep.preprocess()
            res = run_analysis(prep, reduce_dimensions=reduce_dimensions)
            ml_results = res[0]
            nn_results = res[1]
            results.append(res)
            with open(filepath, "w") as f:
                f.write("study, method, accuracy, loss, params \n")
                f.write(
                    "\n".join(
                        [
                            re.results()
                            for methods in ml_results.values()
                            for method_results in methods.values()
                            for re in method_results
                        ]
                    )
                    + " \n"
                )
                f.write(
                    "\n".join(
                        [re.results() for study in nn_results.values() for re in study]
                    )
                    + " \n"
                )
    return results


if __name__ == "__main__":
    date = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    results = main(date)
    folder = Path("results")
    filename = folder / f"{date} / total.csv"
    with open(filename, "w") as f:
        f.write("\n".join([r.results() for res in results for r in res]))
