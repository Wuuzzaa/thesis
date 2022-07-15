from pathlib import Path
import pandas as pd
import warnings
from sklearn.decomposition import PCA


def create_pca_features(X_train: pd.DataFrame, X_test: pd.DataFrame, X_train_pca_file: Path, X_test_pca_file: Path, pca_params: dict, prefix: str):
    # check if file already exists -> load from files the features
    if X_train_pca_file.is_file() and X_test_pca_file.is_file():
        warnings.warn(f"pca files found load from {X_train_pca_file} and {X_test_pca_file}")
        df_pca_train = pd.read_feather(X_train_pca_file)
        df_pca_test = pd.read_feather(X_test_pca_file)

        return df_pca_train, df_pca_test

    # make a pca instance
    pca = PCA(**pca_params)

    # X_train pca features dataframe
    df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

    # X_test pca features dataframe
    df_pca_test = pd.DataFrame(pca.transform(X_test)).add_prefix(prefix)

    # check for NaN
    if df_pca_train.isnull().values.any():
        raise ValueError("train dataframe pca features contain Nan Values")
    if df_pca_test.isnull().values.any():
        raise ValueError("test dataframe pca features contain Nan Values")

    # store pca features
    df_pca_train.to_feather(X_train_pca_file)
    df_pca_test.to_feather(X_test_pca_file)

    return df_pca_train, df_pca_test


if __name__ == "__main__":
    X_train = pd.DataFrame(
        data={
            "a": [i for i in range(1000)],
            "b": [i+3 for i in range(1000)],
            "c": [i**1 for i in range(1000)],
            "d": [i / 8 for i in range(1000)],
        }
    )

    X_test = pd.DataFrame(
        data={
            "a": [i for i in range(100)],
            "b": [i+3 for i in range(100)],
            "c": [i**1 for i in range(100)],
            "d": [i / 8 for i in range(100)],
        }
    )

    pca_params = {
        "n_components": 3,
        "random_state": 42
    }

    create_pca_features(
        X_train=X_train,
        X_test=X_test,
        X_train_pca_file=Path("..//data//datasets//3//pca_train_clean.feather"),
        X_test_pca_file=Path("..//data//datasets//3//pca_test_clean.feather"),
        pca_params=pca_params,
        prefix="pca_"
    )
