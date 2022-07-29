from sklearn.ensemble import RandomForestClassifier

from constants import *
from pathlib import Path
import openml

from load_and_clean_suite_datasets import load_and_clean_suite_datasets
from extract_datasets_info import extract_datasets_info, extract_amount_ohe_features
from calc_scores import calc_scores
from analyze_results import add_compare_scores_columns, print_info_performance_overview, analyze_feature_importance, \
    extract_tuned_hyperparameter_from_models
from src.constants import RANDOM_STATE
from src.create_features import create_features
from src.feature_selection import feature_selection

if __name__ == "__main__":
    ####################################################################################################################
    # LOAD DATA PREPROCESSING
    ####################################################################################################################

    # load suite
    # https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html#sphx-glr-examples-20-basic-simple-suites-tutorial-py
    suite = openml.study.get_suite(99)

    # first load the datasets from the suit and use ohe etc
    load_and_clean_suite_datasets(suite, RANDOM_STATE)

    # extract infos like amount features, classes etc.
    extract_datasets_info(suite)

    # amount features after one hot encoding
    extract_amount_ohe_features(
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
    )

    ####################################################################################################################
    # GENERATE PCA FEATURES
    ####################################################################################################################

    # pca
    pca_params = {
        "n_components": N_COMPONENTS_PCA_UMAP,
        "random_state": RANDOM_STATE
    }

    create_features(
        feature_type="pca",
        train_filename=X_TRAIN_CLEAN_PCA_FILE_NAME,
        test_filename=X_TEST_CLEAN_PCA_FILE_NAME,
        datasets_folder=DATASETS_FOLDER_PATH,
        transformer_params=pca_params,
        prefix="pca_",
        pca_mode="pca",
        random_state=RANDOM_STATE,
        X_file_name=X_CLEAN_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    # kernel pca
    kpca_params = {
        "n_components": N_COMPONENTS_PCA_UMAP,
        "random_state": RANDOM_STATE,
        "kernel": "rbf",
        "n_jobs": -1,
        "copy_X": False,
        "eigen_solver": "randomized"  # "auto" did run in a first test. "randomized" is faster and should be used when n_components is low according to sklearn docu/guide.
    }

    create_features(
        feature_type="pca",
        train_filename=X_TRAIN_CLEAN_KPCA_FILE_NAME,
        test_filename=X_TEST_CLEAN_KPCA_FILE_NAME,
        datasets_folder=DATASETS_FOLDER_PATH,
        transformer_params=kpca_params,
        prefix="kpca_",
        pca_mode="kpca",
        random_state=RANDOM_STATE,
        X_file_name=X_CLEAN_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )
    
    ####################################################################################################################
    # GENERATE UMAP FEATURES
    ####################################################################################################################
    umap_params = { 
        "n_neighbors": 100,  # default 15
        "n_components": N_COMPONENTS_PCA_UMAP,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": True,
    }
    
    create_features(
        feature_type="umap",
        train_filename=X_TRAIN_CLEAN_UMAP_FILE_NAME,
        test_filename=X_TEST_CLEAN_UMAP_FILE_NAME,
        datasets_folder=DATASETS_FOLDER_PATH,
        transformer_params=umap_params,
        prefix="umap_",
        random_state=RANDOM_STATE,
        X_file_name=X_CLEAN_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    ####################################################################################################################
    # GENERATE KMEANS FEATURES
    ####################################################################################################################

    kmeans_params = {
        # "n_clusters": 8, # DO NOT SET THIS BECAUSE WE USE BRUTE FORCE TO DETERMIN THIS VALUE
        "batch_size": 256 * 16,  # 256 * cpu threads is suggested in sklearn docu
        "verbose": 0,
        "random_state": RANDOM_STATE,
    }

    create_features(
        feature_type="kmeans",
        train_filename=X_TRAIN_CLEAN_KMEANS_FILE_NAME,
        test_filename=X_TEST_CLEAN_KMEANS_FILE_NAME,
        datasets_folder=DATASETS_FOLDER_PATH,
        transformer_params=kmeans_params,
        prefix="kmeans_",
        random_state=RANDOM_STATE,
        X_file_name=X_CLEAN_FILE_NAME,
        y_file_name=Y_FILE_NAME,
        kmeans_n_cluster_range=range(2, 11)
    )

    ####################################################################################################################
    # FEATURE SELECTION
    ####################################################################################################################
    feature_selection(
        random_state=RANDOM_STATE,
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        X_filtered_file_name=X_FILTERED_FILE_NAME,
        X_clean_file_name=X_CLEAN_FILE_NAME,
        y_file_name=Y_FILE_NAME,
        max_features=MAX_FEATURES_FEATURE_SELECTION,
        sample_size=10_000
    )

    ####################################################################################################################
    # CALC SCORES - BASELINE
    ####################################################################################################################

    # calc the train and test scores for the "baseline".
    # "baseline": see calc_scores docu
    calc_scores(
        random_state=RANDOM_STATE,
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        mode="baseline",
        estimator=RandomForestClassifier(),
        estimator_param_grid=PARAM_GRID_RANDOM_FOREST,
        cv=5,
        estimator_file_path_suffix=CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX,
        X_file_name=X_FILTERED_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    ####################################################################################################################
    # CALC SCORES - PCA
    ####################################################################################################################

    # calc the train and test scores for the "pca_clean".
    # "pca_clean": see calc_scores docu

    calc_scores(
        random_state=RANDOM_STATE,
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        mode="pca_clean",
        X_train_pca_file_name=X_TRAIN_CLEAN_PCA_FILE_NAME,
        X_test_pca_file_name=X_TEST_CLEAN_PCA_FILE_NAME,
        estimator=RandomForestClassifier(),
        estimator_param_grid=PARAM_GRID_RANDOM_FOREST,
        cv=5,
        estimator_file_path_suffix=CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX,
        X_file_name=X_FILTERED_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    ####################################################################################################################
    # CALC SCORES - KERNEL PCA
    ####################################################################################################################

    # calc the train and test scores for the "kpca_clean".
    # "kpca_clean": see calc_scores docu

    calc_scores(
        random_state=RANDOM_STATE,
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        mode="kpca_clean",
        X_train_pca_file_name=X_TRAIN_CLEAN_KPCA_FILE_NAME,
        X_test_pca_file_name=X_TEST_CLEAN_KPCA_FILE_NAME,
        estimator=RandomForestClassifier(),
        estimator_param_grid=PARAM_GRID_RANDOM_FOREST,
        cv=5,
        estimator_file_path_suffix=CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX,
        X_file_name=X_FILTERED_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    ####################################################################################################################
    # CALC SCORES - PCA AND KERNEL PCA TOGETHER
    ####################################################################################################################

    # calc the train and test scores for the "pca_and_kpca_clean".
    # "pca_and_kpca_clean": see calc_scores docu

    calc_scores(
        random_state=RANDOM_STATE,
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        mode="pca_and_kpca_clean",
        estimator=RandomForestClassifier(),
        estimator_param_grid=PARAM_GRID_RANDOM_FOREST,
        cv=5,
        estimator_file_path_suffix=CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX,
        X_file_name=X_FILTERED_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    ####################################################################################################################
    # CALC SCORES - UMAP
    ####################################################################################################################

    calc_scores(
        random_state=RANDOM_STATE,
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        mode="umap_clean",
        estimator=RandomForestClassifier(),
        estimator_param_grid=PARAM_GRID_RANDOM_FOREST,
        cv=5,
        estimator_file_path_suffix=CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX,
        X_file_name=X_FILTERED_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )


    ####################################################################################################################
    # RESULTS STATISTICS
    ####################################################################################################################
    analyze_feature_importance(
        path_results_file=RESULTS_FILE_PATH,
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_feature_importance_folder=FEATURE_IMPORTANCE_FOLDER_PATH
    )

    add_compare_scores_columns(results_file_path=RESULTS_FILE_PATH)
    extract_tuned_hyperparameter_from_models(
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        model_file_path_suffix=CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX,
    )

    print_info_performance_overview(results_file_path=RESULTS_FILE_PATH)




