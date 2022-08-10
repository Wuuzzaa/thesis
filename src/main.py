from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from constants import *
from pathlib import Path
import openml

from load_and_clean_suite_datasets import load_and_clean_suite_datasets
from extract_datasets_info import extract_datasets_info, extract_amount_ohe_features
from calc_scores import calc_scores
from analyze_results import add_compare_scores_columns, print_info_performance_overview, analyze_feature_importance, \
    extract_tuned_hyperparameter_from_models, compare_stacking_prediction_with_stacking_features
from constants import RANDOM_STATE
from create_features import create_features
from feature_selection import feature_selection


def _preprocessing():
    # load suite
    # https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html#sphx-glr-examples-20-basic-simple-suites-tutorial-py

    print("load suite")
    print("can take some time openml servers are not great and sometimes down ;-)")
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


def _generate_features():
    for X_file_name in [X_CLEAN_FILE_NAME, X_FILTERED_FILE_NAME]:
        ################################################################################################################
        # GENERATE PCA FEATURES
        ################################################################################################################
        # pca
        # set train, test filenames according to the Type of X (clean, filtered, etc.)
        if X_file_name == X_CLEAN_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_PCA_FILE_NAME
            test_filename = X_TEST_CLEAN_PCA_FILE_NAME

        elif X_file_name == X_FILTERED_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_FILTERED_PCA_FILE_NAME
            test_filename = X_TEST_CLEAN_FILTERED_PCA_FILE_NAME

        else:
            raise (NotImplemented(f"{X_file_name} not implemented"))

        create_features(
            feature_type="pca",
            train_filename=train_filename,
            test_filename=test_filename,
            datasets_folder=DATASETS_FOLDER_PATH,
            transformer_params=PCA_PARAMS,
            prefix="pca_",
            pca_mode="pca",
            random_state=RANDOM_STATE,
            X_file_name=X_file_name,
            y_file_name=Y_FILE_NAME,
            path_results_file=RESULTS_FILE_PATH,
        )

        # kernel pca
        # set train, test filenames according to the Type of X (clean, filtered, etc.)
        if X_file_name == X_CLEAN_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_KPCA_FILE_NAME
            test_filename = X_TEST_CLEAN_KPCA_FILE_NAME

        elif X_file_name == X_FILTERED_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_FILTERED_KPCA_FILE_NAME
            test_filename = X_TEST_CLEAN_FILTERED_KPCA_FILE_NAME

        else:
            raise (NotImplemented(f"{X_file_name} not implemented"))

        create_features(
            feature_type="pca",
            train_filename=train_filename,
            test_filename=test_filename,
            datasets_folder=DATASETS_FOLDER_PATH,
            transformer_params=KPCA_PARAMS,
            prefix="kpca_",
            pca_mode="kpca",
            random_state=RANDOM_STATE,
            X_file_name=X_file_name,
            y_file_name=Y_FILE_NAME,
            path_results_file=RESULTS_FILE_PATH,
        )

        # ################################################################################################################
        # # GENERATE UMAP FEATURES
        # ################################################################################################################
        # umap_params = {
        #     "n_neighbors": 100,  # default 15
        #     "n_components": N_COMPONENTS_PCA_UMAP_LDA,
        #     "n_jobs": -1,
        #     "random_state": RANDOM_STATE,
        #     "verbose": True,
        # }
        #
        # # set train, test filenames according to the Type of X (clean, filtered, etc.)
        # if X_file_name == X_CLEAN_FILE_NAME:
        #     train_filename = X_TRAIN_CLEAN_UMAP_FILE_NAME
        #     test_filename = X_TEST_CLEAN_UMAP_FILE_NAME
        #
        # elif X_file_name == X_FILTERED_FILE_NAME:
        #     train_filename = X_TRAIN_CLEAN_FILTERED_UMAP_FILE_NAME
        #     test_filename = X_TEST_CLEAN_FILTERED_UMAP_FILE_NAME
        #
        # else:
        #     raise (NotImplemented(f"{X_file_name} not implemented"))
        #
        # create_features(
        #     feature_type="umap",
        #     train_filename=train_filename,
        #     test_filename=test_filename,
        #     datasets_folder=DATASETS_FOLDER_PATH,
        #     transformer_params=umap_params,
        #     prefix="umap_",
        #     random_state=RANDOM_STATE,
        #     X_file_name=X_file_name,
        #     y_file_name=Y_FILE_NAME,
        # )

        ################################################################################################################
        # GENERATE KMEANS FEATURES
        ################################################################################################################
        # set train, test filenames according to the Type of X (clean, filtered, etc.)
        if X_file_name == X_CLEAN_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_KMEANS_FILE_NAME
            test_filename = X_TEST_CLEAN_KMEANS_FILE_NAME

        elif X_file_name == X_FILTERED_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_FILTERED_KMEANS_FILE_NAME
            test_filename = X_TEST_CLEAN_FILTERED_KMEANS_FILE_NAME

        else:
            raise (NotImplemented(f"{X_file_name} not implemented"))

        create_features(
            feature_type="kmeans",
            train_filename=train_filename,
            test_filename=test_filename,
            datasets_folder=DATASETS_FOLDER_PATH,
            transformer_params=KMEANS_PARAMS,
            prefix="kmeans_",
            random_state=RANDOM_STATE,
            X_file_name=X_file_name,
            y_file_name=Y_FILE_NAME,
            kmeans_n_cluster_range=range(2, 101),
            path_results_file=RESULTS_FILE_PATH,
        )

        ################################################################################################################
        # GENERATE LDA FEATURES
        ################################################################################################################
        # set train, test filenames according to the Type of X (clean, filtered, etc.)
        if X_file_name == X_CLEAN_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_LDA_FILE_NAME
            test_filename = X_TEST_CLEAN_LDA_FILE_NAME

        elif X_file_name == X_FILTERED_FILE_NAME:
            train_filename = X_TRAIN_CLEAN_FILTERED_LDA_FILE_NAME
            test_filename = X_TEST_CLEAN_FILTERED_LDA_FILE_NAME

        else:
            raise (NotImplemented(f"{X_file_name} not implemented"))

        create_features(
            feature_type="lda",
            train_filename=train_filename,
            test_filename=test_filename,
            datasets_folder=DATASETS_FOLDER_PATH,
            transformer_params=LDA_PARAMS,
            prefix="lda_",
            random_state=RANDOM_STATE,
            X_file_name=X_file_name,
            y_file_name=Y_FILE_NAME,
            path_results_file=RESULTS_FILE_PATH,
        )

        # ################################################################################################################
        # # GENERATE STACKING FEATURES
        # ################################################################################################################
        #
        # # todo not ideal to skip so stupid maybe rewrite all of this function
        # # skip clean features. just takes to long to generate
        # if X_file_name == X_CLEAN_FILE_NAME:
        #     continue
        #
        # # make params dict for stacked classifier
        # params_dict = {
        #     "estimators": [
        #         ("random_forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        #         ("logistic_regression", LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)),
        #         ("knn", KNeighborsClassifier(n_jobs=-1)),
        #         ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        #         ("hist_gradient_boosting_classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        #         ("mlp", MLPClassifier(random_state=RANDOM_STATE))
        #     ],
        #     "final_estimator": LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        #     "cv": 5,
        #     "n_jobs": -1,
        #     "passthrough": False,
        #     "verbose": 2,
        # }
        #
        # # set train, test filenames according to the Type of X (clean, filtered, etc.)
        # if X_file_name == X_CLEAN_FILE_NAME:
        #     train_filename = X_TRAIN_CLEAN_STACKING_FILE_NAME
        #     test_filename = X_TEST_CLEAN_STACKING_FILE_NAME
        #
        # elif X_file_name == X_FILTERED_FILE_NAME:
        #     train_filename = X_TRAIN_CLEAN_FILTERED_STACKING_FILE_NAME
        #     test_filename = X_TEST_CLEAN_FILTERED_STACKING_FILE_NAME
        #
        # else:
        #     raise (NotImplemented(f"{X_file_name} not implemented"))
        #
        # create_features(
        #     feature_type="stacking",
        #     train_filename=train_filename,
        #     test_filename=test_filename,
        #     datasets_folder=DATASETS_FOLDER_PATH,
        #     transformer_params=params_dict,
        #     prefix="stacking_",
        #     random_state=RANDOM_STATE,
        #     X_file_name=X_file_name,
        #     y_file_name=Y_FILE_NAME,
        # )


if __name__ == "__main__":
    _preprocessing()

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

    _generate_features()

    for mode in CALC_SCORES_MODES:
        calc_scores(
            random_state=RANDOM_STATE,
            path_datasets_folder=DATASETS_FOLDER_PATH,
            path_results_file=RESULTS_FILE_PATH,
            mode=mode,
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
    # todo refactor if needed at all?
    # analyze_feature_importance(
    #     path_results_file=RESULTS_FILE_PATH,
    #     path_datasets_folder=DATASETS_FOLDER_PATH,
    #     path_feature_importance_folder=FEATURE_IMPORTANCE_FOLDER_PATH
    # )

    add_compare_scores_columns(results_file_path=RESULTS_FILE_PATH)

    extract_tuned_hyperparameter_from_models(
        path_datasets_folder=DATASETS_FOLDER_PATH,
        path_results_file=RESULTS_FILE_PATH,
        model_file_path_suffix=CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX,
    )

    # compare_stacking_prediction_with_stacking_features(
    #     path_datasets_folder=DATASETS_FOLDER_PATH,
    #     feature_file_name=X_TEST_CLEAN_FILTERED_STACKING_FILE_NAME,
    #     path_results_file=RESULTS_FILE_PATH,
    # )

    print_info_performance_overview(results_file_path=RESULTS_FILE_PATH)




