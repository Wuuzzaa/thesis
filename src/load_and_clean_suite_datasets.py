from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from pathlib import Path
import openml
import pandas as pd
import numpy as np


def load_and_clean_suite_datasets(suite, random_state):
    """
    Loads all datasets from an openml suite. Cleans the data and stores dataframes for X and y.
    Cleaning steps:
        - impute NaN values
        - one hot encode
        - standard scale
        - label encode
        - shuffle data
        - store X and y as dataframes in the feather format.

    :param random_state: int
    :param suite: openml suite
    :return: None
    """
    print(suite)

    # get all tasks of the study
    tasks = suite.tasks

    for task_id in tasks:
        print()
        task = openml.tasks.get_task(task_id)
        print(task)

        # show dataset id is not equal the taskid
        print()
        print(f"dataset_id: {str(task.dataset_id)}")

        # set paths for files
        path = Path(f"..//data//{str(task.dataset_id)}//")
        path_X = path.joinpath(f"X_clean.feather")
        path_y = path.joinpath(f"y.feather")

        # check if already done -> skip the task/dataset
        if path_X.exists() and path_y.exists():
            print("already done skip")
            continue

        print(f"download dataset...")
        dataset = task.get_dataset()

        # get X as Dataframe and y as Series
        target = dataset.default_target_attribute
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=target)

        # impute NaN values
        if X.isnull().values.any():
            n_features_before = len(X.columns)
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

            if len(categorical_features) > 0:
                X[categorical_features] = CategoricalImputer(ignore_format=False).fit_transform(X[categorical_features], y)

            if len(numeric_features) > 0:
                X[numeric_features] = MeanMedianImputer(imputation_method="mean").fit_transform(X[numeric_features], y)

            n_features_after = len(X.columns)

            if n_features_before != n_features_after:
                raise ValueError("Features before and after NaN imputation are not the same!")

        # X one hot encode
        try:
            encoder = OneHotEncoder()
            X = encoder.fit_transform(X, y)
        except ValueError as e:
            # not nice but there are different value error thrown, so we need to handle by error message
            if "No categorical variables found in this dataframe" in str(e):
                warnings.warn(
                    "ValueError: No categorical variables found in this dataframe. -> No One hot encoding used.")
            else:
                raise

        # store feature names after encoding
        feature_names_encoded = X.columns

        # standard scale X
        X = StandardScaler().fit_transform(X)

        # get a dataframe again after scaling from ndarray
        X = pd.DataFrame(X)
        X.columns = feature_names_encoded

        # y label encode
        y = LabelEncoder().fit_transform(y)

        # shuffle data
        X["y"] = y
        X = X.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # split y and X again
        y = X["y"]
        X = X.drop(labels="y", axis="columns")

        # create folder for dataset
        path.mkdir(parents=True, exist_ok=True)

        # store X
        X.to_feather(path=path_X)

        # store y as Dataframe cause feather can not store Series
        y = pd.DataFrame(y, columns=["y"])
        y.to_feather(path=path_y)
