from pathlib import Path
import pandas as pd
import warnings
from sklearn.decomposition import PCA, KernelPCA


def create_pca_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_train_pca_file: Path,
        X_test_pca_file: Path,
        pca_params: dict,
        prefix: str,
        mode: str,
        random_state: int,
):
    # check if file already exists -> load from files the features
    if X_train_pca_file.is_file() and X_test_pca_file.is_file():
        warnings.warn(f"pca files found load from {X_train_pca_file} and {X_test_pca_file}")
        df_pca_train = pd.read_feather(X_train_pca_file)
        df_pca_test = pd.read_feather(X_test_pca_file)

        return df_pca_train, df_pca_test

    # make a pca instance
    if mode == "pca":
        pca = PCA(**pca_params)

    elif mode == "kpca":
        pca = KernelPCA(**pca_params)

    else:
        raise ValueError(f"mode {mode} is not implemented")

    # for performance sake check the size of the train matrix and reduce n_components when it is too huge
    # most likely when "mle" mode is used for n_components
    train_size = X_train.shape[0] * X_train.shape[1]

    if train_size > 10_000_000 and mode == "pca":
        # fall back to n_components = 10
        warnings.warn("Train size is too huge. Fall back to n_components = 10")
        pca.set_params(**{"n_components": 10})

    # X_train pca features dataframe
    if mode == "pca":
        try:
            print("pca fit transform train")
            df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

        except ValueError:
            warnings.warn("n_components='mle' is only supported if n_samples >= n_features. Fall back to n_components = 3")

            # fall back to n_components = 3
            pca.set_params(**{"n_components": 3})
            df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

    elif mode == "kpca":
        # kernel pca uses far too much ram even on mid sized datasets or higher.
        # So we need to use a sample of the train data
        if len(X_train) > 1000:
            X_train_sample = X_train.sample(n=1000, random_state=random_state)

            print("pca fit")
            pca.fit(X_train_sample)

            print("pca transform train")
            df_pca_train = pd.DataFrame(pca.transform(X_train)).add_prefix(prefix)

        else:
            print("pca fit transform train")
            df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

    # X_test pca features dataframe
    print("pca transform test")
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
