# https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html#sphx-glr-examples-20-basic-simple-suites-tutorial-py
import warnings
from pathlib import Path

import openml
import pandas as pd

from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import RandomSampleImputer, CategoricalImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

random_state = 42

# load study
suite = openml.study.get_suite(99)
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

    # check if already done -> skip the task/dataset
    path = Path(f".//data//{str(task.dataset_id)}//")
    path_X = path.joinpath(f"X.feather")
    path_y = path.joinpath(f"y.feather")

    if path_X.exists() and path_y.exists():
        print("already done skip")
        continue

    print(f"download dataset...")
    dataset = task.get_dataset()

    # get X as Dataframe and y as Series
    target = dataset.default_target_attribute
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=target)

    # impute NaN values
    X = CategoricalImputer(ignore_format=True).fit_transform(X, y)

    # X one hot encode
    try:
        encoder = OneHotEncoder()
        X = encoder.fit_transform(X, y)
    except ValueError as e:
        # not nice but there are different value error thrown so we need to handle by error message
        if "No categorical variables found in this dataframe" in str(e):
            warnings.warn("ValueError: No categorical variables found in this dataframe. -> No One hot encoding used.")
        else:
            raise

    # store feature names
    feature_names_encoded = X.columns

    # Standardscale X
    X = StandardScaler().fit_transform(X)

    # get a dataframe again after scaling from ndarray
    X = pd.DataFrame(X)
    X.columns = feature_names_encoded

    # y label encode
    y = LabelEncoder().fit_transform(y)

    # shuffel data
    X["y"] = y
    X = X.sample(frac=1).reset_index(drop=True)

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

