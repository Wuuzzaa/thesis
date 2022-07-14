import warnings

import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


def calc_scores(random_state, path_datasets_folder, path_results_file, mode):
    """
    Function to calc cross validation score for the train data and score for the test data. The scores are appended to
    the results file.

    :param random_state: int
    :param path_datasets_folder: Path to the folder with the datasets
    :param path_results_file: Path to the results file
    :param mode: str. Current modes:
        "baseline": Runs a random forest on the cleaned data
    :return: None
    """
    # check if already done, by checking if the new to generate columns are already in the results dataframe
    if mode == "baseline":
        train_cv_score_column_name = "baseline_random_forest_train_cv_score"
        test_score_column_name = "baseline_random_forest_test_score"

    else:
        raise NotImplemented(f"mode: {mode} is not implemented")

    new_columns = [train_cv_score_column_name, test_score_column_name]

    # load the results dataframes columns
    results_df_columns = pd.read_feather(path_results_file).columns.tolist()

    # when all column we would calculate by this mode are already in the results dataframe we can skip it
    if set(new_columns).issubset(results_df_columns):
        warnings.warn(f"All columns for mode: {mode} are already in the results file. Done")
        return

    # make some lists
    train_cv_scores = []
    test_scores = []
    dataset_folders = []

    # use a random forest as classifier
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    # get the dataset folders in the data folder
    for path in path_datasets_folder.iterdir():
        if path.is_dir():
            dataset_folders.append(path)

    # score each dataset
    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}")

        if mode == "baseline":
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

    # make a dataframe from the score lists
    scores_df = pd.DataFrame(data={
        train_cv_score_column_name: train_cv_scores,
        test_score_column_name: test_scores
    })

    # load the results dataframe
    results_df = pd.read_feather(path_results_file)

    # concat the results dataframe with the scores dataframe
    results_df = pd.concat([results_df, scores_df], axis="columns")

    # store results with scores
    results_df.to_feather(path_results_file)
