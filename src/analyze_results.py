import warnings
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from constants import *
import joblib
import numpy as np

from calc_scores import get_X_train_X_test_y_train_y_test
from util import get_sub_folders, print_function_header


def add_compare_scores_columns(results_file_path: Path):
    # print header
    print_function_header(f"add compare scores columns to results dataframe")

    # read file into dataframe
    df = pd.read_feather(results_file_path)

    mode_test_score_better_baseline_columns_clean = []
    mode_test_score_better_baseline_columns_clean_filtered = []

    for mode in CALC_SCORES_MODES:
        # skip baseline and stacking modes
        if mode == "baseline_filtered" or "stacking" in mode:
            continue

        # train score vs baseline
        df[f"{mode}_train_score > baseline_filtered_train_score"] = df[f"{mode}_train_cv_score"] > df["baseline_filtered_train_cv_score"]

        # test score vs baseline
        df[f"{mode}_test_score > baseline_filtered_test_score"] = df[f"{mode}_test_score"] > df["baseline_filtered_test_score"]

        # at the moment we have features on clean data and features on clean and filtered data
        if "_filtered" in mode and not "baseline_filtered" in mode:
            mode_test_score_better_baseline_columns_clean_filtered.append(f"{mode}_test_score > baseline_filtered_test_score")
        else:
            mode_test_score_better_baseline_columns_clean.append(f"{mode}_test_score > baseline_filtered_test_score")

        # test score change in percent vs baseline
        df[f"{mode}_test_score_change_to_baseline_filtered"] = (df[f"{mode}_test_score"] / df["baseline_filtered_test_score"] - 1) * 100

    # check if any new feature type improved the score compared to the baseline
    df["any_feature_type_clean_test_score > baseline_filtered_test_score"] = df[mode_test_score_better_baseline_columns_clean].any(axis='columns')
    df["any_feature_type_clean_filtered_test_score > baseline_filtered_test_score"] = df[mode_test_score_better_baseline_columns_clean_filtered].any(axis='columns')

    # store again
    df.to_feather(results_file_path)


def print_info_performance_overview(results_file_path: Path):
    # load results dataframe
    df = pd.read_feather(results_file_path)

    # do some statistics
    n_datasets = len(df)

    ####################################################################################################################
    # test data
    ####################################################################################################################
    print("#" * 80)
    print("Statistics on test data".upper())
    print("#" * 80)
    print("")

    for mode in CALC_SCORES_MODES:
        if mode == "baseline_filtered" or "stacking" in mode:
            continue

        n_improved_datasets = sum(df[f'{mode}_test_score > baseline_filtered_test_score'])
        improved_dataset_percent = round(n_improved_datasets / n_datasets * 100, 2)

        print(f"{mode}:".upper())
        print(f"{mode} data improved the performance on {n_improved_datasets} datasets = {improved_dataset_percent}%")
        print("")

    # All feature modes together on clean data (not filtered)
    n_any_new_feature_type_improved_test_score_compared_to_baseline = sum(df["any_feature_type_clean_test_score > baseline_filtered_test_score"])
    any_new_feature_type_improved_test_score_compared_to_baseline_percent = round(n_any_new_feature_type_improved_test_score_compared_to_baseline / n_datasets * 100, 2)

    print("---")
    print(f"When all modes were tried the performance improved on {n_any_new_feature_type_improved_test_score_compared_to_baseline} datasets = {any_new_feature_type_improved_test_score_compared_to_baseline_percent}% on at least one new featuretype mode. Features were generated on cleaned data NOT filtered")
    print("---")

    # All feature modes together on clean data and filtered
    n_any_new_feature_type_improved_test_score_compared_to_baseline = sum(df["any_feature_type_clean_filtered_test_score > baseline_filtered_test_score"])
    any_new_feature_type_improved_test_score_compared_to_baseline_percent = round(n_any_new_feature_type_improved_test_score_compared_to_baseline / n_datasets * 100, 2)

    print("---")
    print(f"When all modes were tried the performance improved on {n_any_new_feature_type_improved_test_score_compared_to_baseline} datasets = {any_new_feature_type_improved_test_score_compared_to_baseline_percent}% on at least one new featuretype mode. Features were generated on cleaned and filtered data")
    print("---")

    ####################################################################################################################
    # train data
    ####################################################################################################################
    print("")
    print("#"*80)
    print("Statistics on train data".upper())
    print("#"*80)
    print("")

    for mode in CALC_SCORES_MODES:
        if mode == "baseline_filtered":
            continue

        n_improved_datasets = sum(df[f'{mode}_train_score > baseline_filtered_train_score'])
        improved_dataset_percent = round(n_improved_datasets / n_datasets * 100, 2)

        print(f"{mode}:".upper())
        print(f"{mode} data improved the performance on {n_improved_datasets} datasets = {improved_dataset_percent}%")
        print("")


def extract_tuned_hyperparameter_from_models(
        path_datasets_folder: Path,
        path_results_file: Path,
):
    print("")
    print("#"*80)
    print("extract tuned hyperparameter from models".upper())
    print("#" * 80)
    print("")

    # get all dataset folders
    dataset_folders = get_sub_folders(path_datasets_folder)

    # load the result dataframe
    df_results = pd.read_feather(path_results_file)

    # key mode, value dict with key dataset_id and value str of a dict with tuned hyperparameter
    hyperparameter_dict = {}

    # set empty dicts for each mode
    for mode in CALC_SCORES_MODES:
        hyperparameter_dict[mode] = {}

    dataset_folder: Path
    for dataset_folder in tqdm(dataset_folders):
        for mode in CALC_SCORES_MODES:
            print()
            print("---")
            print(f"folder: {dataset_folder}, mode: {mode}")

            # set the model file path suffix
            if "stacking" in mode:
                model_file_path_suffix = CALC_SCORES_STACKING_FILE_PATH_SUFFIX

            else:
                model_file_path_suffix = CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX

            # set the column name to add for the model according to mode
            columnname = f"model_hyperparameter_{mode}{model_file_path_suffix}".replace(".joblib", "")

            # check if already done
            if columnname in df_results.columns:
                warnings.warn(f"{columnname} already in results dataframe. skip")
                continue

            # load the model file
            model_file_path = dataset_folder.joinpath(f"{mode}{model_file_path_suffix}")
            model = joblib.load(model_file_path)

            # extract the params dict as str to store it in the results dataframe
            params_str = str(model.get_params())

            # add params to dict
            hyperparameter_dict[mode][int(dataset_folder.name)] = params_str

            # some info about the depth of the trees when it is a tree based model
            try:
                print("Tree depth information")
                print(f"mean depth: {np.mean([estimator.get_depth() for estimator in model.estimators_])}")
                print(f"min depth: {np.min([estimator.get_depth() for estimator in model.estimators_])}")
                print(f"max depth: {np.max([estimator.get_depth() for estimator in model.estimators_])}")

            except:
                pass

    # add new columns to the results dataframe
    for mode in CALC_SCORES_MODES:
        # set the model file path suffix
        if "stacking" in mode:
            model_file_path_suffix = CALC_SCORES_STACKING_FILE_PATH_SUFFIX

        else:
            model_file_path_suffix = CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX

        # set the column name to add for the model according to mode
        columnname = f"model_hyperparameter_{mode}{model_file_path_suffix}".replace(".joblib", "")

        # add a new column to the results dataframe with the whole params if not already done
        if columnname not in df_results.columns:
            # select the dict with the current mode from the hyperparameter dict and sort it
            temp_dict = hyperparameter_dict[mode]
            temp_dict = dict(sorted((temp_dict.items())))
            df_results[columnname] = temp_dict.values()

    df_results.to_feather(path_results_file)







