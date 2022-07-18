from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from pathlib import Path
import openml
import pandas as pd
import numpy as np
from itertools import compress
from constans import *


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
    print("")
    print("#"*80)
    print("load and clean suite datasets".upper())
    print("#" * 80)
    print("")

    print(suite)

    # get all tasks of the study
    tasks = suite.tasks

    for task_id in tasks:
        print()
        task = openml.tasks.get_task(task_id)

        # show dataset id is not equal the taskid
        print()
        print(f"dataset_id: {str(task.dataset_id)}")

        # set paths for files
        path = DATASETS_FOLDER_PATH.joinpath(str(task.dataset_id))
        path_X = path.joinpath(X_CLEAN_FILE_NAME)
        path_y = path.joinpath(y_FILE_NAME)

        # check if already done -> skip the task/dataset
        if path_X.exists() and path_y.exists():
            print("already done skip")
            continue

        print(f"download dataset...")
        dataset = task.get_dataset()

        # get X as Dataframe and y as Series
        target = dataset.default_target_attribute
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=target)

        # make lists of numerical and categorical features
        categorical_features = list(compress(attribute_names, categorical_indicator))
        numeric_features = list(set(attribute_names) - set(categorical_features))

        # check if object type numerical features if so kick it and switch it to categorical features list.
        # its not possible to impute mean on not numeric data
        not_numeric_columns = list(X[numeric_features].select_dtypes(exclude=np.number).columns)
        if not_numeric_columns:
            categorical_features.extend(not_numeric_columns)
            numeric_features = list(set(numeric_features) - set(not_numeric_columns))

        # impute NaN values
        if X.isnull().values.any():
            if len(categorical_features) > 0:
                X = CategoricalImputer(ignore_format=True, variables=categorical_features).fit_transform(X, y)

            if len(numeric_features) > 0:
                X = MeanMedianImputer(imputation_method="mean", variables=numeric_features).fit_transform(X, y)

            if X.isnull().values.any():
                raise ValueError("There are still NaN values in the dataframe")

        # X one hot encode
        n_features_before = len(X.columns)

        try:
            if len(categorical_features) > 0:
                encoder = OneHotEncoder(variables=categorical_features, ignore_format=True)
                X = encoder.fit_transform(X, y)
        except ValueError as e:
            # not nice but there are different value error thrown, so we need to handle by error message
            if "No categorical variables found in this dataframe" in str(e):
                warnings.warn(
                    "ValueError: No categorical variables found in this dataframe. -> No One hot encoding used.")
            else:
                raise

        n_features_after = len(X.columns)
        if n_features_after < n_features_before:
            raise Exception("Amount of features after OHE is lower than before encoding")

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
