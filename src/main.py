import warnings
from pathlib import Path
import openml
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from load_and_clean_suite_datasets import load_and_clean_suite_datasets
from extract_datasets_info import extract_datasets_info, extract_amount_ohe_features
from calc_scores import calc_scores


if __name__ == "__main__":
    random_state = 42

    # load suite
    # https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html#sphx-glr-examples-20-basic-simple-suites-tutorial-py
    suite = openml.study.get_suite(99)

    # first load the datasets from the suit and use ohe etc
    load_and_clean_suite_datasets(suite, random_state)

    # extract infos like amount features, classes etc.
    extract_datasets_info(suite)

    # amount features after one hot encoding
    extract_amount_ohe_features(
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
    )

    # calc the train and test scores for the baseline.
    # Baseline means random forest on cleaned data. No filter no additional features
    calc_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
        mode="baseline",
    )

    # calc the train and test scores for the "pca_clean".
    # "pca_clean": Runs a random forest on the cleaned data with pca additional features no feature selection

    pca_params = {
        "n_components": 3,
        "random_state": random_state
    }

    calc_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
        mode="pca_clean",
        X_train_pca_file_name="pca_train_clean.feather",
        X_test_pca_file_name="pca_test_clean.feather",
        pca_params=pca_params,
        prefix="pca_",
    )




