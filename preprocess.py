import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from logger import logger


@dataclass
class Preprocessor:
    dataframes: Optional[Dict[str, pl.DataFrame]] = None
    labels: Optional[List[str]] = None
    SVC_params = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [_ for _ in range(2, 6)],
        "tol": [1e-3, 1e-4],
    }
    RF_params = {
        "n_estimators": [100, 150, 200, 250, 300],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
    }

    def preprocess(self):
        raise NotImplementedError()

    @staticmethod
    def normalize_columns_split_label(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
        """
        Normalize all columns of the dataframe, except for the label column (which doesn't need
        normalization)
        Then return data and labels separately.
        """

        def sigmoid(val: float, max: float):
            if val == 0 and max == 0:
                return 0
            exponent = val / max
            return 1 / (1 + math.exp(-exponent))

        df = df.clone()
        types = df.dtypes
        for i, col in enumerate(df.columns):
            if types[i] != pl.Float64:
                continue
            if col != label_col:
                max_val = df.select(pl.col(col).max()).item()
                df = df.with_column(
                    pl.col(col)
                    .apply(lambda x: sigmoid(x, max_val), pl.Float64)
                    .alias(col)
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
    def train_test_split(
        self, code: Optional[str] = None, reduce_dimensions: bool = False
    ):
        dfs = self.dataframes

        if dfs is None:
            raise Exception("First you have to preprocess some data")
        if code:
            df = dfs[code]
        else:
            df = pl.concat([d for d in dfs.values()])

        X_data = df.select(
            [col for col in df.columns if col not in ["illness", "study"]]
        ).to_numpy()

        if reduce_dimensions:
            pca = PCA(
                n_components="mle"
                if X_data.shape[0] >= X_data.shape[1]
                else X_data.shape[0],
                copy=True,
            )
            print("Reducing dimension. Original shape: ", X_data.shape)
            X_data = pca.fit_transform(X_data)
            print("Final shape: ", X_data.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            df.get_column("illness").to_numpy(),
        )

        return X_train, X_test, y_train, y_test
