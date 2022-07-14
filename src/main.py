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


    #load_and_clean_suite_datasets(suite, random_state)
    #extract_datasets_info(suite)
    extract_amount_ohe_features(
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
    )

    calc_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
        mode="baseline",
    )




