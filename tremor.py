import os
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import pandas as pd
import polars as pl

from logger import logger
from preprocess import Preprocessor

studies = ["Ga", "Ju", "Si"]
directory = "./data/tremor and gait/gait-in-parkinsons-disease-1.0.0/"


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
    def preprocess(self) -> None:
        self.dataframes = {}
        dfs: DefaultDict[Any, list] = defaultdict(list)
        for file in os.listdir(directory):
            if any(file.startswith(c) for c in studies):
                path = Path(directory) / file
                self.load_file(path, dfs)

        for key, _dfs in dfs.items():
            self.dataframes[key] = self.process_dataframes(_dfs)



preprocessor = TremorGaitPreprocessor(
    labels=[
        "Time",
        *[f"L{i}" for i in range(1, 9)],
        *[f"R{i}" for i in range(1, 9)],
        "LT",
        "RT",
    ]
)
