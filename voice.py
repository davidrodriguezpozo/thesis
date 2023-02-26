from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from logger import logger
from preprocess import Preprocessor


class VoicePreprocessor(Preprocessor):
    ...

    @logger.log
    def preprocess(self) -> None:
        voice2_directory = (
            "./data/voice/VOICE 2/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"
        )
        voice3_directory = "./data/voice/VOICE 3/parkinsons_data.csv"

        italian_parkinsons = Path("./data/voice/Italian Parkinson's Voice and speech")

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
                italian_parkinsons
                / "28 People with Parkinson's disease"
                / "TAB 5.xlsx",
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
        )
        italian_dataset = (
            italian_dataset.drop(columns=["surname", "name"])
            .with_columns(
                [pl.when(pl.col("sex") == "M").then(1).otherwise(1).alias("sex")]
            )
            .fill_null(0)
        )
        pl.Config.set_tbl_cols(1000)

        # VOICE 2

        voice2_df = (
            pl.from_pandas(pd.read_csv(voice2_directory))
            .drop(columns=["ID"])
            .rename({"Status": "illness"})
            .fill_null(0)
        )

        # VOICE 3

        voice3_df = (
            pl.from_pandas(pd.read_csv(voice3_directory))
            .drop(columns=["name"])
            .rename({"status": "illness"})
            .fill_null(0)
        )

        self.dataframes = {
            "study1": italian_dataset,
            "study2": voice2_df,
            "study3": voice3_df,
        }


preprocessor = VoicePreprocessor()
