import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import json_normalize

sns.set(font_scale=2)
sns.set_theme(style="ticks", palette="pastel", font_scale=1.5)


def boxplot(df: pd.DataFrame, x="method", hue="study", title=""):
    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(x=x, y="accuracy", hue=hue, palette=["m", "g"], data=df).set(
        title=title
    )
    sns.despine(trim=True)


def load_dataset(
    path, reduced: bool = False, normalized: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    voice_dfs = None
    tremor_dfs = None

    if normalized and reduced:
        path = Path(path + "_norm_red")
    elif normalized:
        path = Path(path + "_normalized")
    elif reduced:
        path = Path(path + "_reduced")
    else:
        path = Path(path)

    def add_spec_columns(df: pd.DataFrame) -> None:
        df["normalized"] = True if normalized else False
        df["reduced"] = True if reduced else False

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

    add_spec_columns(voice_dfs)
    add_spec_columns(tremor_dfs)

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
    ).set(title="Accuracy distribution by method")


def load_all_datasets(path: str):
    reg_voice, reg_tremor = load_dataset(path)
    norm_voice, norm_tremor = load_dataset(path, normalized=True)
    red_voice, red_tremor = load_dataset(path, reduced=True)
    norm_red_voice, norm_red_tremor = load_dataset(path, normalized=True, reduced=True)
    return pd.concat([reg_voice, norm_voice, red_voice, norm_red_voice]), pd.concat(
        [reg_tremor, norm_tremor, red_tremor, norm_red_tremor]
    )


def main():
    path = Path("results/03-08-2023_00-33-32")
    voice_dfs, tremor_dfs = load_dataset(path)

    # PLOT BOXES
    # sns.histplot(data=voice_dfs, x="accuracy", hue="method", stat='density', common_norm=False, multiple='stack').set(title="Accuracy distribution by method")
    # sns.relplot(data=df, x="method", y="accuracy", hue="study")
    # sns.histplot(data=voice_dfs, x="accuracy", hue="study", stat='density', common_norm=False, multiple='stack').set(title="Accuracy distribution by study")
    # sns.jointplot(data=voice_dfs, x="accuracy", y="method", hue="study").set(title="Accuracy by study and method")

    # This one is very similar to boxplot, but flipped
    # sns.catplot(voice_dfs, kind='box', x="accuracy", y="study", hue="method").set(title="Accuracy distribution by method and study")

    # sns.displot(voice_dfs, kind='hist', x="accuracy", hue="method", kde=True, common_norm=False, stat='density').set(title="Accuracy distribution by method and study")
    # sns.displot(voice_dfs, kind='hist', x="accuracy", col="study", row="method", kde=True, common_norm=False, stat='density')


if __name__ == "__main__":
    main()
