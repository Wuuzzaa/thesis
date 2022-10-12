import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomTreesEmbedding, StackingClassifier
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from constants import *
import numpy as np
from shutil import make_archive, unpack_archive
import time
from time import perf_counter
import warnings

from src.calc_scores import get_X_train_X_test_y_train_y_test
from pycaret.classification import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

from pycaret_util import run_pycaret

# dataset_id = "1489"
# #dataset_id = "40923" # 1489 # 3
#
# # load data
# X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
#     dataset_folder=DATASETS_FOLDER_PATH.joinpath(dataset_id), random_state=RANDOM_STATE,
#     X_file_name=X_FILTERED_FILE_NAME, y_file_name=Y_FILE_NAME)
#
# # short feedback of the data and classes
# print(f"X_train shape: {X_train.shape}")
# print(f"target classes: \n{y_train.value_counts()}")
# print(f"total {len(y_train.value_counts())} classes\n")
#
# #sample if needed
# sample_size = 1_000
#
# if len(X_train) > sample_size:
#     print(f"Sample is used of {sample_size}")
#     X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=sample_size, random_state=RANDOM_STATE)
#
# else:
#     X_train_sample = X_train
#     y_train_sample = y_train

# get all dataset folders
from src.util import get_sub_folders

dataset_folders = get_sub_folders(DATASETS_FOLDER_PATH)

model_types_to_use = [
    #"rf",
    "nb",
    "lr",
    #"knn",  # too much ram usage?
    # "mlp",  # too slow
    "dt",
    # "xgboost",
    # "lightgbm",
    # "catboost",
]

# score each dataset
for dataset_folder in tqdm(dataset_folders):
    print()
    print("---")
    print(f"current folder: {dataset_folder}")
    print("---")

    # make paths for savefiles
    path_caret_baseline = dataset_folder.joinpath("caret_baseline_results.feather")
    path_caret_baseline_and_improved_features = dataset_folder.joinpath("caret_baseline_and_improved_features_results.feather")

    # check if already done
    if path_caret_baseline.exists() and path_caret_baseline_and_improved_features.exists():
        print("Both results files are created. Skip this dataset")
        continue

    # get X and y train and test splitted
    X_train_baseline, X_test_baseline, y_train, y_test = get_X_train_X_test_y_train_y_test(
        dataset_folder=dataset_folder,
        random_state=RANDOM_STATE,
        X_file_name=X_FILTERED_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    caret_results_df_baseline = run_pycaret(
        X_train=X_train_baseline.copy(),
        X_test=X_test_baseline.copy(),
        y_train=y_train.copy(),
        y_test=y_test.copy(),
        random_state=RANDOM_STATE,
        model_types_to_use=model_types_to_use,
        cv=5,
        n_iter_tune=50,
    )

    # store file
    caret_results_df_baseline.to_feather(path_caret_baseline)

    print("#"*80)
    print("pycaret results with baseline features".upper())
    print("#" * 80)
    print(caret_results_df_baseline)
    print()

    # store dataframes to concat later on for the train and test dataframe
    X_train_dfs = []
    X_test_dfs = []

    print("add baseline dataframes")
    X_train_dfs.append(X_train_baseline)
    X_test_dfs.append(X_test_baseline)

    # load results to determine which features should be added
    df_results = pd.read_feather(RESULTS_FILE_PATH)

    dataset_row = df_results[df_results["dataset_id"] == int(dataset_folder.name)]
    print()

    for new_feature_type in NEW_FEATURE_TYPE_LIST:
        print("\n---")
        print(f"check new feature type: {new_feature_type}")

        baseline_filtered_train_score = float(dataset_row['baseline_filtered_train_cv_score'])
        baseline_filtered_new_feature_type_train_score = float(dataset_row[f'baseline_filtered_{new_feature_type}_train_cv_score'])

        print(f"baseline filtered train score: {baseline_filtered_train_score}")
        print(f"baseline filtered with {new_feature_type} train score: {baseline_filtered_new_feature_type_train_score}\n")

        if baseline_filtered_train_score < baseline_filtered_new_feature_type_train_score:
            print("performance was increased on train cross validation use the feature")
            print(f"add {new_feature_type} dataframes")
            X_train_dfs.append(pd.read_feather(dataset_folder.joinpath(f"{new_feature_type}_train_clean.feather")))
            X_test_dfs.append(pd.read_feather(dataset_folder.joinpath(f"{new_feature_type}_test_clean.feather")))

        else:
            print("performance was not increased. Do not use this feature for stacking")

    # concat all needed dataframes for train and test data
    X_train = pd.concat(X_train_dfs, axis="columns")
    X_test = pd.concat(X_test_dfs, axis="columns")

    caret_results_df_baseline_and_improved_features = run_pycaret(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=RANDOM_STATE,
        model_types_to_use=model_types_to_use,
        cv=5,
        n_iter_tune=50
    )

    # store file
    caret_results_df_baseline_and_improved_features.to_feather(path_caret_baseline_and_improved_features)

    print("#"*80)
    print("pycaret results with baseline features and features which improved the train score".upper())
    print("#" * 80)
    print(caret_results_df_baseline_and_improved_features)
    print()
