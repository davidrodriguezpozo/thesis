import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import json_normalize

sns.set(font_scale=2)
sns.set_theme(style="ticks", palette="pastel")


def boxplot(df: pd.DataFrame, x="method", hue="study", title=""):
    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(x=x, y="accuracy", hue=hue, palette=["m", "g"], data=df).set(
        title=title
    )
    sns.despine(trim=True)


def load_dataset(path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    voice_dfs = None
    tremor_dfs = None

    def parse_json(x: str):
        if pd.isna(x) or not x.strip():
            return {}
        return json.loads(x)

    def load_params(df: pd.DataFrame) -> pd.DataFrame:
        params = df["params"].apply(parse_json)
        parsed_params = json_normalize(params)
        return df.merge(parsed_params, left_index=True, right_index=True).drop(
            columns=["params"]
        )

    for folder in os.listdir(path):
        for idx, file in enumerate(os.listdir(path / folder)):
            df = pd.read_csv(path / folder / file)
            df["run"] = folder.split("_")[1]
            df.columns = [c.strip() for c in df.columns]
            if idx == 0:
                tremor_dfs = (
                    pd.concat([tremor_dfs, df]) if tremor_dfs is not None else df
                )
            else:
                voice_dfs = pd.concat([voice_dfs, df]) if voice_dfs is not None else df
    return load_params(voice_dfs), load_params(tremor_dfs)


def plot_nine_grid(df):
    sns.displot(
        df,
        kind="hist",
        x="accuracy",
        col="study",
        row="method",
        kde=True,
        common_norm=False,
        stat="density",
    )


def plot_accuracy_by_method_study(df):
    sns.displot(
        df,
        kind="hist",
        x="accuracy",
        hue="method",
        kde=True,
        common_norm=False,
        stat="density",
    ).set(title="Accuracy distribution by method and study")


def main():
    path = Path("results/03-08-2023_00-33-32")
    voice_dfs, tremor_dfs = load_dataset(path)
    # XXX: Embed IPython
    _esc = __import__("functools").partial(__import__("os")._exit, 0)  # FIXME
    __import__("IPython").embed()  # FIXME
    # /XXX: Embed IPython
    # PLOT BOXES
    # sns.histplot(data=voice_dfs, x="accuracy", hue="method", stat='density', common_norm=False, multiple='stack').set(title="Accuracy distribution by method")
    # sns.relplot(data=df, x="method", y="accuracy", hue="study")
    # sns.histplot(data=voice_dfs, x="accuracy", hue="study", stat='density', common_norm=False, multiple='stack').set(title="Accuracy distribution by study")
    # sns.jointplot(data=voice_dfs, x="accuracy", y="method", hue="study").set(title="Accuracy by study and method")
    # sns.catplot(voice_dfs, kind='box', x="accuracy", y="study", hue="method").set(title="Accuracy distribution by method and study")
    # sns.displot(voice_dfs, kind='hist', x="accuracy", hue="method", kde=True, common_norm=False, stat='density').set(title="Accuracy distribution by method and study")
    # sns.displot(voice_dfs, kind='hist', x="accuracy", col="study", row="method", kde=True, common_norm=False, stat='density')


if __name__ == "__main__":
    main()
