import warnings
from pathlib import Path
import openml
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from load_and_clean_suite_datasets import load_and_clean_suite_datasets
from extract_datasets_info import extract_datasets_info


def calc_baseline_scores(random_state, path_datasets_folder, path_results_file):
    train_cv_scores = []
    test_scores = []
    dataset_folders = []

    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    # get the dataset folders in the data folder
    for path in path_datasets_folder.iterdir():
        if path.is_dir():
            dataset_folders.append(path)

    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}...")

        # get X, y
        path_X = dataset_folder.joinpath("X_clean.feather")
        path_y = dataset_folder.joinpath("y.feather")

        X = pd.read_feather(path_X)
        y = pd.read_feather(path_y)["y"]

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=0.75)

        # fit model
        rf.fit(X_train, y_train)

        # score model on test data and cv score for train data
        test_scores.append(rf.score(X_test, y_test))
        train_cv_scores.append(cross_val_score(rf, X_train, y_train, cv=10, n_jobs=-1).mean())

    # make a dataframe from the scorelists
    scores_df = pd.DataFrame(data={
        "baseline_random_forest_train_cv_score": train_cv_scores,
        "baseline_random_forest_test_score": test_scores
    })

    # load the results dataframe
    results_df = pd.read_feather(path_results_file)

    # concat the results dataframe with the scores dataframe
    results_df = pd.concat([results_df, scores_df], axis="columns")

    # store results with scores
    results_df.to_feather(path_results_file)
    pass



if __name__ == "__main__":
    random_state = 42

    # load suite
    # https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html#sphx-glr-examples-20-basic-simple-suites-tutorial-py
    suite = openml.study.get_suite(99)


    #load_and_clean_suite_datasets(suite, random_state)
    #extract_datasets_info(suite)
    calc_baseline_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
    )




