# %%

import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             plot_confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

from _types.results import BaseResults
from logger import Logger
from nn import NeuralNetwork

logger = Logger()


@dataclass
class Preprocessor:
    dataframe: Optional[pd.DataFrame] = None
    labels: List[str] = None
    SVC_params = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [_ for _ in range(3, 6)],
        "tol": [1e-3, 1e-4],
    }
    RF_params = {
        "n_estimators": [100, 150, 200, 250, 300],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
    }

    @staticmethod
    def normalize_columns_split_label(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
        """
        Normalize all columns of the dataframe, except for the label column (which doesn't need
        normalization)
        Then return data and labels separately.
        """
        df = df.clone()
        types = df.dtypes
        for i, col in enumerate(df.columns):
            if types[i] != pl.Float32:
                continue
            if col != label_col:
                df = df.with_column(
                    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
                )

        return df

    def get_parameters(self):
        def _permutate_dict(parameters: Dict[str, List[Any]]) -> List[Dict[str, str]]:
            keys, values = zip(*parameters.items())
            permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
            return permutations_dicts

        return {
            "SVM": _permutate_dict(self.SVC_params),
            "RF": _permutate_dict(self.RF_params),
        }

    @logger.log
    def train_test_split(self, code: Optional[str] = None):
        df = self.dataframe

        if df is None:
            raise Exception("First you have to preprocess some data")
        if code:
            df = df.filter(pl.col("study") == code)

        X_train, X_test, y_train, y_test = train_test_split(
            df.select(
                [col for col in df.columns if col not in ["illness", "study"]]
            ).to_numpy(),
            df.get_column("illness").to_numpy(),
        )

        return X_train, X_test, y_train, y_test


class TremorGaitPreprocessor(Preprocessor):
    @staticmethod
    def _split_name(file_path: Path) -> Tuple[str, str]:
        key = "Si"
        if "Ga" in file_path.name:
            key = "Ga"
        if "Ju" in file_path.name:
            key = "Ju"

        file_name = file_path.name.split(".")[0]
        if "Pt" in file_name:
            subject = file_name.split("Pt")[1]
        else:
            subject = file_name.split("Co")[1]

        return key, subject

    def load_file(
        self,
        file_path: Path,
        dfs: Dict[str, List[pl.DataFrame]],
    ) -> None:
        df = pl.from_pandas(pd.read_csv(file_path, header=None, sep="\t"))
        key, subject = TremorGaitPreprocessor._split_name(file_path)
        if self.labels:
            df = df.with_columns(
                [pl.col(f"{i}").alias(l) for i, l in enumerate(self.labels)]
            ).drop(columns=[str(i) for i in range(len(self.labels))])
        df = df.with_column(pl.lit(subject).alias("subject"))
        if "Pt" in file_path.name:
            dfs[key].append(df.with_column(pl.lit(1).alias("illness")))
        else:
            dfs[key].append(df.with_column(pl.lit(0).alias("illness")))

    def explode_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        aggregates = ["mean", "std", "max", "min"]
        return (
            df.groupby("subject")
            .agg(
                [
                    *[
                        getattr(pl.col(col), agg)().alias(f"{col}_{agg}")
                        for col in self.labels[1:]
                        for agg in aggregates
                    ],
                    pl.col("illness").max().keep_name(),
                ]
            )
            .drop(columns=["subject"])
        )

    def process_dataframes(self, dfs: List[pl.DataFrame]) -> pl.DataFrame:
        total_df = pl.DataFrame()
        for df in dfs:
            df = self.explode_columns(df)
            total_df = pl.concat([total_df, df], how="diagonal")
        return self.__class__.normalize_columns_split_label(
            total_df, label_col="illness"
        )

    @logger.log
    def preprocess(self, directory: str) -> None:
        dfs: DefaultDict[Any, list] = defaultdict(list)
        for file in os.listdir(directory):
            if any(file.startswith(c) for c in studies):
                path = Path(directory) / file
                self.load_file(path, dfs)

        compacted_dfs = None
        for key, _dfs in dfs.items():
            df_to_append = self.process_dataframes(_dfs).with_column(
                pl.lit(key).alias("study")
            )
            compacted_dfs = (
                pl.concat([compacted_dfs, df_to_append], how="diagonal")
                if compacted_dfs is not None
                else df_to_append
            )

        # CAMBIAR ESTO !!!!!!!!
        self.dataframe = compacted_dfs


preprocessor = TremorGaitPreprocessor(
    labels=[
        "Time",
        *[f"L{i}" for i in range(1, 9)],
        *[f"R{i}" for i in range(1, 9)],
        "LT",
        "RT",
    ]
)

studies = ["Ga", "Ju", "Si"]
directory = "tremor and gait/gait-in-parkinsons-disease-1.0.0/"

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
    ):
        self.model = model
        self.method = method
        self.results = results
        self.actual = actual
        self.X_test = x_test
        self.cross_val_results = cross_val_results

    @property
    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.results, self.actual)

    @property
    def accuracy(self) -> float:
        return accuracy_score(self.actual, self.results)

    def plot(self) -> None:
        print("")
        print(f"Results for {self.method}".center(40))
        print("-------------".center(40))
        print(f"Accuracy: {self.accuracy}".center(40))
        print(f"Confusion matrix:".center(40))
        print(f"Cross validation: {np.mean(self.cross_val_results)}")
        print(f"{self.confusion_matrix}".center(40))

    def plot_confusion_matrix(self) -> None:
        plot_confusion_matrix(self.model, self.X_test, self.actual)
        plt.show()


@logger.log
def train_predict(
    method: str,
    model: Union[RandomForestClassifier, SVC],
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
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


# TODO

# LOAD VOICE DATASETS
voice2_directory = "./voice/VOICE 2/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"
voice3_directory = "./voice/VOICE 3/parkinsons_data.csv"

# Italian Parkinsons Dataset

italian_parkinsons = Path("./voice/Italian Parkinson's Voice and speech")

young_healthy = pd.read_excel(
    italian_parkinsons / "15 Young Healthy Control" / "15 YHC.xlsx",
    sheet_name=0,
    header=1,
).dropna(axis=0)
older_healthy = (
    pd.read_excel(
        italian_parkinsons / "22 Elderly Healthy Control" / "Tab 3.xlsx",
        sheet_name=0,
        header=1,
    )
    .dropna(axis=0)
    .replace("-", np.nan)
)
disease_set = (
    pd.read_excel(
        italian_parkinsons / "28 People with Parkinson's disease" / "TAB 5.xlsx",
        sheet_name=0,
        header=1,
        usecols=list(range(1, 11)),
    )
    .dropna(axis=0)
    .replace("//", np.nan)
)

healthy_young: pl.DataFrame = (
    pl.from_pandas(young_healthy)
    .with_columns([pl.lit(0).alias("illness")])
    .drop(columns="from")
)
healthy_old: pl.DataFrame = (
    pl.from_pandas(older_healthy)
    .with_columns([pl.lit(0).alias("illness")])
    .drop(columns="from")
)
disease_old: pl.DataFrame = (
    pl.from_pandas(disease_set)
    .select([pl.col(c).alias(c.replace(" ", "")) for c in disease_set.columns])
    .with_columns([pl.lit(1).alias("illness")])
)
italian_dataset = pl.concat(
    [healthy_young, healthy_old, disease_old], how="diagonal"
).with_columns([pl.lit("it").alias("study")])
italian_dataset = italian_dataset.with_columns(
    [(pl.col("name") + " " + pl.col("surname")).alias("name")]
).drop(columns=["surname"])
pl.Config.set_tbl_cols(1000)

# VOICE 2

voice2_df = pl.from_pandas(pd.read_csv(voice2_directory))
print(voice2_df)

# VOICE 3

# COMPARE
