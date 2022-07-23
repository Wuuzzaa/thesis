from pathlib import Path
import pandas as pd
import warnings
from sklearn.decomposition import PCA, KernelPCA
from tqdm import tqdm

from src.calc_scores import get_X_train_X_test_y_train_y_test


def _create_pca_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        pca_params: dict,
        prefix: str,
        mode: str,
        random_state: int,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Function to create pca features.

    :param X_train:
    :param X_test:
    :param pca_params: dict with the parameters for sklearns PCA or Kernel PCA
    :param prefix: prefix for the columnname
    :param mode: on of "pca" or "kpca" which one to use
    :param random_state:
    :return:
    """
    # make a pca instance
    if mode == "pca":
        pca = PCA(**pca_params)

    elif mode == "kpca":
        pca = KernelPCA(**pca_params)

    else:
        raise ValueError(f"mode {mode} is not implemented")

    # X_train pca features dataframe
    if mode == "pca":
        print("pca fit transform train")
        df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

    elif mode == "kpca":
        # kernel pca uses far too much ram even on mid-sized datasets or higher.
        # So we need to use a sample of the train data

        sample_size = 10_000

        if len(X_train) > sample_size:
            warnings.warn("Dataset is huge. Kernel PCA needs huge memory with too much rows. -> Sample of 10000 rows is used")
            X_train_sample = X_train.sample(n=sample_size, random_state=random_state)

            print("pca fit")
            pca.fit(X_train_sample)

            print("pca transform train")
            df_pca_train = pd.DataFrame(pca.transform(X_train)).add_prefix(prefix)

        else:
            print("pca fit transform train")
            df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

    else:
        raise ValueError(f"unkown mode {mode}")

    # X_test pca features dataframe
    print("pca transform test")
    df_pca_test = pd.DataFrame(pca.transform(X_test)).add_prefix(prefix)

    # check for NaN
    if df_pca_train.isnull().values.any():
        raise ValueError("train dataframe pca features contain Nan Values")
    if df_pca_test.isnull().values.any():
        raise ValueError("test dataframe pca features contain Nan Values")

    return df_pca_train, df_pca_test


def create_pca_features(
        pca_train_filename: str,
        pca_test_filename: str,
        datasets_folder: Path,
        pca_params: dict,
        prefix: str,
        mode: str,
        random_state: int,
        X_file_name: str,
        y_file_name: str
):
    print()
    print("#"*80)
    print(f"create pca features with mode: {mode}".upper())
    print("#" * 80)
    print()

    # store the dataset folders
    dataset_folders = []

    # get the dataset folders in the data folder
    for path in datasets_folder.iterdir():
        if path.is_dir():
            dataset_folders.append(path)

    # make pca features for each dataset
    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}")

        X_train_pca_file_path = dataset_folder.joinpath(pca_train_filename)
        X_test_pca_file_path = dataset_folder.joinpath(pca_test_filename)

        # check if file already exists -> load from files the features
        if X_train_pca_file_path.is_file() and X_test_pca_file_path.is_file():
            warnings.warn(f"pca files found  {X_train_pca_file_path} and {X_test_pca_file_path}. Done")
            continue

        # get X and y train and test splitted
        X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
            dataset_folder=dataset_folder,
            random_state=random_state,
            X_file_name=X_file_name,
            y_file_name=y_file_name,
        )

        # make the features dataframes for train and test
        df_pca_train, df_pca_test = _create_pca_features(
                X_train=X_train,
                X_test=X_test,
                pca_params=pca_params,
                prefix=prefix,
                mode=mode,
                random_state=random_state,
        )

        # store pca features
        df_pca_train.to_feather(X_train_pca_file_path)
        df_pca_test.to_feather(X_test_pca_file_path)

