import warnings
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from constants import *
import joblib
import numpy as np

from src.util import get_sub_folders


def add_compare_scores_columns(results_file_path: Path):
    # read file into dataframe
    df = pd.read_feather(results_file_path)

    # compare pca with baseline
    df["pca_clean_train_score > baseline_train_score"]  = df["pca_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["pca_clean_test_score > baseline_test_score"]    = df["pca_clean_test_score"] > df["baseline_test_score"]

    # compare kpca with baseline
    df["kpca_clean_train_score > baseline_train_score"]  = df["kpca_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["kpca_clean_test_score > baseline_test_score"]    = df["kpca_clean_test_score"] > df["baseline_test_score"]

    # compare umap with baseline
    df["umap_clean_train_score > baseline_train_score"] = df["umap_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["umap_clean_test_score > baseline_test_score"] = df["umap_clean_test_score"] > df["baseline_test_score"]

    # compare kmeans with baseline
    df["kmeans_clean_train_score > baseline_train_score"]   = df["kmeans_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["kmeans_clean_test_score > baseline_test_score"]     = df["kmeans_clean_test_score"] > df["baseline_test_score"]

    # compare pca_kpca_umap_kmeans_clean with baseline
    df["pca_kpca_umap_kmeans_clean_train_score > baseline_train_score"]   = df["pca_kpca_umap_kmeans_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["pca_kpca_umap_kmeans_clean_test_score > baseline_test_score"]     = df["pca_kpca_umap_kmeans_clean_test_score"] > df["baseline_test_score"]

    # kpca and pca > baseline
    df["pca_clean_test_score & kpca_clean_test_score > baseline_test_score"] = df["pca_clean_test_score > baseline_test_score"] & df["kpca_clean_test_score > baseline_test_score"]
    df["pca_clean_train_score & kpca_clean_train_score > baseline_train_score"] = df["pca_clean_train_score > baseline_train_score"] & df["kpca_clean_train_score > baseline_train_score"]
    
    # kpca and pca > baseline on train and test at the same time
    df["pca_kpca_clean_train_and_test_score > baseline_train_and_test_score"] = df["pca_clean_test_score & kpca_clean_test_score > baseline_test_score"] & df["pca_clean_train_score & kpca_clean_train_score > baseline_train_score"]

    # change in percent
    df["pca_clean_test_score_change_to_baseline"]           = (df["pca_clean_test_score"] / df["baseline_test_score"] - 1) * 100
    df["kpca_clean_test_score_change_to_baseline"]          = (df["kpca_clean_test_score"] / df["baseline_test_score"] - 1) * 100
    df["umap_clean_test_score_change_to_baseline"]          = (df["umap_clean_test_score"] / df["baseline_test_score"] - 1) *100
    df["kmeans_clean_test_score_change_to_baseline"]          = (df["kmeans_clean_test_score"] / df["baseline_test_score"] - 1) *100
    df["pca_kpca_umap_kmeans_clean_test_score_change_to_baseline"]          = (df["pca_kpca_umap_kmeans_clean_test_score"] / df["baseline_test_score"] - 1) *100

    # check if any new feature type improved the score compared to the baseline
    df["any_feature_type_test_score > baseline_test_score"] = df[
        [
            "pca_clean_test_score > baseline_test_score",
            "kpca_clean_test_score > baseline_test_score",
            "umap_clean_test_score > baseline_test_score",
            "kmeans_clean_test_score_change_to_baseline",
            "pca_kpca_umap_kmeans_clean_test_score_change_to_baseline",
        ]
    ].any(axis='columns')

    # store again
    df.to_feather(results_file_path)


def print_info_performance_overview(results_file_path: Path):
    # load results dataframe
    df = pd.read_feather(results_file_path)

    # do some statistics
    n_datasets = len(df)

    ####################################################################################################################
    # TEST DATA
    ####################################################################################################################

    # pca test data
    n_pca_improved_datasets_test = sum(df['pca_clean_test_score > baseline_test_score'])
    pca_improved_dataset_percent_test = round(n_pca_improved_datasets_test / n_datasets * 100, 2)

    # kpca test data
    n_kpca_improved_datasets_test = sum(df['kpca_clean_test_score > baseline_test_score'])
    kpca_improved_dataset_percent_test = round(n_kpca_improved_datasets_test / n_datasets * 100, 2)

    # umap test data
    n_umap_improved_datasets_test = sum(df['umap_clean_test_score > baseline_test_score'])
    umap_improved_dataset_percent_test = round(n_umap_improved_datasets_test / n_datasets * 100, 2)

    # kmeans test data
    n_kmeans_improved_datasets_test = sum(df['kmeans_clean_test_score > baseline_test_score'])
    kmeans_improved_dataset_percent_test = round(n_kmeans_improved_datasets_test / n_datasets * 100, 2)

    # pca_kpca_umap_kmeans_clean test data
    n_pca_kpca_umap_kmeans_clean_improved_datasets_test = sum(df['pca_kpca_umap_kmeans_clean_test_score > baseline_test_score'])
    pca_kpca_umap_kmeans_clean_improved_dataset_percent_test = round(n_pca_kpca_umap_kmeans_clean_improved_datasets_test / n_datasets * 100, 2)

    # any new feature type improved the score compared to the baseline
    n_any_new_feature_type_improved_test_score_compared_to_baseline = sum(df["any_feature_type_test_score > baseline_test_score"])
    any_new_feature_type_improved_test_score_compared_to_baseline_percent = round(n_any_new_feature_type_improved_test_score_compared_to_baseline / n_datasets * 100, 2)

    ####################################################################################################################
    # TRAIN DATA
    ####################################################################################################################
    # pca train data
    n_pca_improved_datasets_train = sum(df['pca_clean_train_score > baseline_train_score'])
    pca_improved_dataset_percent_train = round(n_pca_improved_datasets_train / n_datasets * 100, 2)

    # kpca train data
    n_kpca_improved_datasets_train = sum(df['kpca_clean_train_score > baseline_train_score'])
    kpca_improved_dataset_percent_train = round(n_kpca_improved_datasets_train / n_datasets * 100, 2)

    # umap train data
    n_umap_improved_datasets_train = sum(df['umap_clean_train_score > baseline_train_score'])
    umap_improved_dataset_percent_train = round(n_umap_improved_datasets_train / n_datasets * 100, 2)

    # kmeans train data
    n_kmeans_improved_datasets_train = sum(df['kmeans_clean_train_score > baseline_train_score'])
    kmeans_improved_dataset_percent_train = round(n_kmeans_improved_datasets_train / n_datasets * 100, 2)

    # pca_kpca_umap_kmeans_clean train data
    n_pca_kpca_umap_kmeans_clean_improved_datasets_train = sum(df['pca_kpca_umap_kmeans_clean_train_score > baseline_train_score'])
    pca_kpca_umap_kmeans_clean_improved_dataset_percent_train = round(n_pca_kpca_umap_kmeans_clean_improved_datasets_train / n_datasets * 100, 2)

    # print it out
    print()
    print("#"*80)
    print("performance overview".upper())
    print("#" * 80)
    print()
    print(f"Amount of datasets tested: {n_datasets}")
    print("")
    print("#"*80)

    # TEST DATA

    print("Statistics on test data".upper())
    print("#" * 80)
    print("")
    print("PCA:")
    print(f"pca on clean data improved the performance on {n_pca_improved_datasets_test} datasets = {pca_improved_dataset_percent_test}%")
    print("")
    print("KPCA:")
    print(f"kpca on clean data improved the performance on {n_kpca_improved_datasets_test} datasets = {kpca_improved_dataset_percent_test}%")
    print("")
    print("UMAP:")
    print(f"umap on clean data improved the performance on {n_umap_improved_datasets_test} datasets = {umap_improved_dataset_percent_test}%")
    print("")
    print("KMEANS:")
    print(f"kmeans on clean data improved the performance on {n_kmeans_improved_datasets_test} datasets = {kmeans_improved_dataset_percent_test}%")
    print("")
    print("PCA KPCA UMAP KMEANS TOGETHER:")
    print(f"all together on clean data improved the performance on {n_pca_kpca_umap_kmeans_clean_improved_datasets_test} datasets = {pca_kpca_umap_kmeans_clean_improved_dataset_percent_test}%")
    print("")
    print(f"When all modes were tried the performance improved on {n_any_new_feature_type_improved_test_score_compared_to_baseline} datasets = {any_new_feature_type_improved_test_score_compared_to_baseline_percent}% on at least on new featuretype")

    # TRAIN DATA

    print("")
    print("#"*80)
    print("Statistics on train data".upper())
    print("#"*80)
    print("")
    print("PCA:")
    print(f"pca on clean data improved the cross validation performance on {n_pca_improved_datasets_train} datasets = {pca_improved_dataset_percent_train}%")
    print("")
    print("KPCA:")
    print(f"kpca on clean data improved the cross validation performance on {n_kpca_improved_datasets_train} datasets = {kpca_improved_dataset_percent_train}%")
    print("")
    print("UMAP:")
    print(f"umap on clean data improved the performance on {n_umap_improved_datasets_train} datasets = {umap_improved_dataset_percent_train}%")
    print("")
    print("KMEANS:")
    print(f"kmeans on clean data improved the performance on {n_kmeans_improved_datasets_train} datasets = {kmeans_improved_dataset_percent_train}%")
    print("")
    print("PCA KPCA UMAP KMEANS TOGETHER:")
    print(f"all together on clean data improved the performance on {n_pca_kpca_umap_kmeans_clean_improved_datasets_train} datasets = {pca_kpca_umap_kmeans_clean_improved_dataset_percent_train}%")
    print("")


def analyze_feature_importance(path_results_file: Path, path_datasets_folder: Path, path_feature_importance_folder: Path):
    # todo add umap analyze_feature_importance?
    # todo add kmeans analyze_feature_importance?
    # todo add pca_kpca_umap_kmeans_clean analyze_feature_importance?
    # first get the feature importance from each model and store them in files
    _extract_feature_importance_from_models(path_datasets_folder, path_feature_importance_folder)

    # load results dataframe
    df_results = pd.read_feather(path_results_file)

    # we just care for modes which have pca features
    pca_modes = [mode for mode in CALC_SCORES_MODES if "pca" in mode]

    # check if results dataframe has columns if so stop it.
    all_modes_done = True
    for mode in pca_modes:
        if f"{mode}_pca_features_importance_mean_factor" not in df_results.columns:
            all_modes_done = False

    if all_modes_done:
        warnings.warn("Feature importance columns already in results dataframe. Done")
        return

    # containers
    feature_importance_files = []

    # key mode, value dict with key dataset_id and value pca relative feature importance to mean
    feature_importance_dict = {}

    # set empty dicts for each mode
    for mode in pca_modes:
        if ("pca" in mode) and (mode != "pca_kpca_umap_kmeans_clean"):
            feature_importance_dict[mode] = {}

    # get each feature importance file path
    for path in path_feature_importance_folder.iterdir():
        for mode in pca_modes:
            feature_importance_files.append(path.joinpath(mode, FEATURE_IMPORTANCE_FILE_NAME))

    # calculate the factor how much more important pca features are compared to the others
    for file in feature_importance_files:
        df = pd.read_feather(file)
        mode = file.parts[-2]
        dataset_id = int(file.parts[-3]) # need int of the dataset id for sorting on

        mean_importance = df["importance"].mean()
        mean_pca_importance = df[df['feature'].str.contains('pca')]["importance"].mean()
        feature_importance_dict[mode][dataset_id] = mean_pca_importance / mean_importance

    # add columns to the results dataframe
    for mode in pca_modes:
        temp_dict = feature_importance_dict[mode]
        temp_dict = dict(sorted((temp_dict.items())))

        df_results[f"{mode}_pca_features_importance_mean_factor"] = temp_dict.values()

    # store the results dataframe
    df_results.to_feather(path_results_file)


def _extract_feature_importance_from_models(path_datasets_folder: Path, path_feature_importance_folder: Path):
    if path_feature_importance_folder.exists():
        warnings.warn(f"Feature importance folder {path_feature_importance_folder} already exists. Done")
        return

    dataset_folders = get_sub_folders(path_datasets_folder)

    # extract feature importance from each dataset
    dataset_folder: Path
    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}")

        # each mode has its own feature importance
        for mode in CALC_SCORES_MODES:
            # set the path to the random forest model
            model_file_path = dataset_folder.joinpath(f"{mode}{CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX}")

            # load the model
            rf: RandomForestClassifier = joblib.load(model_file_path)

            # make a dataframe with the features and the importance sorted
            data = {
                "feature": np.array(rf.feature_names_in_),
                "importance": np.array(rf.feature_importances_)
            }

            df_feature_importance = pd.DataFrame(data). \
                sort_values(by=["importance"], ascending=False). \
                reset_index(drop=True)

            # create feature importance folder
            folder_path = path_feature_importance_folder.joinpath(dataset_folder.name, mode)
            folder_path.mkdir(parents=True, exist_ok=True)

            # make the feather file
            file_path = folder_path.joinpath(FEATURE_IMPORTANCE_FILE_NAME)
            print(f"store feature importance to: {file_path}")
            df_feature_importance.to_feather(file_path)


def extract_tuned_hyperparameter_from_models(path_datasets_folder: Path, path_results_file: Path, model_file_path_suffix: str):
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
        # select the dict with the current mode from the hyperparameter dict and sort it
        temp_dict = hyperparameter_dict[mode]
        temp_dict = dict(sorted((temp_dict.items())))

        # set the column name to add for the model according to mode
        columnname = f"model_hyperparameter_{mode}{model_file_path_suffix}".replace(".joblib", "")

        # add a new column to the results dataframe with the whole params if not already done
        if columnname not in df_results.columns:
            df_results[columnname] = temp_dict.values()

    df_results.to_feather(path_results_file)

