from pathlib import Path

import psutil
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

########################################################################################################################
# TEST MODE
########################################################################################################################
"""
Test mode is used for running the whole script against a few subset of all datasets.
Use the flag to set the Testmode to active.
Specify the dataset ids to use for the test in a list.
Make sure to run the test mode on a clear folder without any data downloaded so far. 
Otherwise all downloaded datasets will be used as usual
"""
USE_TESTMODE = True
TESTMODE_DATASET_ID_LIST = [
    40923,  # huge and many classes 92k samples 1k ohe features, 46 classes
    3,      # small 2 classes 3k samples 73 features ohe
    1489,   # random selected just want 3 datasets
]


########################################################################################################################
# INTEGERS
########################################################################################################################
RANDOM_STATE = 42
MAX_FEATURES_FEATURE_SELECTION = 100

########################################################################################################################
# FLOATS
########################################################################################################################
TRAIN_TEST_SPLIT_TRAIN_SIZE = 0.66

########################################################################################################################
# FILENAMES
# Filenames NO Paths see above
# suffix: _FILE_NAME
########################################################################################################################
# X and y
X_CLEAN_FILE_NAME = "X_clean.feather"
X_FILTERED_FILE_NAME = "X_filtered.feather"
Y_FILE_NAME = "y.feather"

# pca
X_TRAIN_CLEAN_PCA_FILE_NAME = "pca_train_clean.feather"
X_TEST_CLEAN_PCA_FILE_NAME = "pca_test_clean.feather"
X_TRAIN_CLEAN_FILTERED_PCA_FILE_NAME = "pca_train_clean_filtered.feather"
X_TEST_CLEAN_FILTERED_PCA_FILE_NAME = "pca_test_clean_filtered.feather"

# kpca
X_TRAIN_CLEAN_KPCA_FILE_NAME = "kpca_train_clean.feather"
X_TEST_CLEAN_KPCA_FILE_NAME = "kpca_test_clean.feather"
X_TRAIN_CLEAN_FILTERED_KPCA_FILE_NAME = "kpca_train_clean_filtered.feather"
X_TEST_CLEAN_FILTERED_KPCA_FILE_NAME = "kpca_test_clean_filtered.feather"

# umap
X_TRAIN_CLEAN_UMAP_FILE_NAME    = "umap_train_clean.feather"
X_TEST_CLEAN_UMAP_FILE_NAME     = "umap_test_clean.feather"
X_TRAIN_CLEAN_FILTERED_UMAP_FILE_NAME    = "umap_train_clean_filtered.feather"
X_TEST_CLEAN_FILTERED_UMAP_FILE_NAME     = "umap_test_clean_filtered.feather"

# kmeans
X_TRAIN_CLEAN_KMEANS_FILE_NAME    = "kmeans_train_clean.feather"
X_TEST_CLEAN_KMEANS_FILE_NAME     = "kmeans_test_clean.feather"
X_TRAIN_CLEAN_FILTERED_KMEANS_FILE_NAME    = "kmeans_train_clean_filtered.feather"
X_TEST_CLEAN_FILTERED_KMEANS_FILE_NAME     = "kmeans_test_clean_filtered.feather"

# lda
X_TRAIN_CLEAN_LDA_FILE_NAME    = "lda_train_clean.feather"
X_TEST_CLEAN_LDA_FILE_NAME     = "lda_test_clean.feather"
X_TRAIN_CLEAN_FILTERED_LDA_FILE_NAME    = "lda_train_clean_filtered.feather"
X_TEST_CLEAN_FILTERED_LDA_FILE_NAME     = "lda_test_clean_filtered.feather"

# results dataframe file
RESULTS_DATAFRAME_FILE_NAME = "results.feather"

# feature importance file
FEATURE_IMPORTANCE_FILE_NAME = "feature_importance.feather"

########################################################################################################################
# PATHS
# Paths from pathlib to folders or files
# suffix: _FOLDER_PATH or _FILE_PATH
########################################################################################################################

# FOLDERS
DATA_FOLDER_PATH = Path("..//data")
DATASETS_FOLDER_PATH = DATA_FOLDER_PATH.joinpath("datasets")
RESULTS_FOLDER_PATH = DATA_FOLDER_PATH.joinpath("results")
FEATURE_IMPORTANCE_FOLDER_PATH = RESULTS_FOLDER_PATH.joinpath("feature_importance")

# FILES
RESULTS_FILE_PATH = RESULTS_FOLDER_PATH.joinpath(RESULTS_DATAFRAME_FILE_NAME)

########################################################################################################################
# CALC_SCORES CONSTANTS
# prefix: CALC_SCORES_
########################################################################################################################

CALC_SCORES_MODES = [
    # """
    # Baseline means the cleaned features after one hot encoding.
    # Filtered means after feature selection.
    #
    # featuremode_filtered means that the filter was generated on filtered basefeatures otherwise all basefeatures were used
    # """
    
    # baseline features
    "baseline_filtered",

    # only new features without baseline features
    "only_pca",
    "only_kpca",
    "only_kmeans",
    "only_lda",
    "only_umap",

    "only_pca_filtered",
    "only_kpca_filtered",
    "only_kmeans_filtered",
    "only_lda_filtered",
    "only_umap_filtered",

    # baseline features and new features
    "baseline_filtered_pca",
    "baseline_filtered_kpca",
    "baseline_filtered_kmeans",
    "baseline_filtered_lda",
    "baseline_filtered_umap",

    "baseline_filtered_pca_filtered",
    "baseline_filtered_kpca_filtered",
    "baseline_filtered_kmeans_filtered",
    "baseline_filtered_lda_filtered",
    "baseline_filtered_umap_filtered",

    # best features selected from basefeatures and new features
    "selected_features",
    "selected_features_filtered",

    # all features used without feature selection
    "all_features",
    "all_features_filtered",

    # stacking
    "stacking_baseline_filtered",
    "stacking_all_features",  # basefeatures filtered, pca, kpca etc. not filtered
]

CALC_SCORES_TRAIN_CV_SCORE_COLUMN_NAME_SUFFIX = "_train_cv_score"
CALC_SCORES_TEST_SCORE_COLUMN_NAME_SUFFIX = "_test_score"
CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX = "_random_forest.joblib"
CALC_SCORES_STACKING_FILE_PATH_SUFFIX = "_stacking.joblib"
CALC_SCORES_TRAIN_TIME_COLUMN_NAME_SUFFIX = "_train_time_in_seconds"

# On my pc there are only 16 gb of ram so i run out of ram when using all cores.
# just test it like this here. psutil... gives the ram size of the pc in gb.
CALC_SCORES_USE_ALWAYS_ALL_CORES_GRIDSEARCH = 8 < (psutil.virtual_memory().total / float(1.074e+9))

########################################################################################################################
# PARAM_GRID_RANDOM_FOREST
# used for hyperparameter tuning
########################################################################################################################
PARAM_GRID_RANDOM_FOREST = {
    "max_depth": [6, 20, None],
    "n_estimators": [100],
    "max_features": ["sqrt"],
    "n_jobs": [-1],
    "class_weight": [None],
    "random_state": [42]
}

PARAM_GRID_STACKING_PARAMS = {
    "estimators": [
        [
            # ("logistic_regression", LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, max_iter=100, solver="saga")), # seems to be slow on huge n_features and or classes
            # ("knn", KNeighborsClassifier(n_jobs=-1)),
            # ("mlp", MLPClassifier(random_state=RANDOM_STATE, early_stopping=True)), # too slow.

            # ("random_forest_deep", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=None)),
            # ("random_forest_shallow ", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=6)),
            # ("dt_deep", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            #("dt_shallow", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6)),
            # ("sgd", SGDClassifier(random_state=RANDOM_STATE, n_jobs=-1, early_stopping=True)),
            # ("hist_gradient_boosting_classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True)),
        ]
    ],
    "final_estimator": [LogisticRegression(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_iter=100,
        solver="saga"
    )],
    "cv": [3],
    "n_jobs": [-1],
    "passthrough": [False],
    "verbose": [1],
}

########################################################################################################################
#  FEATURE GENERATION PARAMETER DICTS
########################################################################################################################
# pca
PCA_PARAMS = {
    #"n_components": 0.8,  # "mle",
    "random_state": RANDOM_STATE,
    "svd_solver": "full",  #"arpack",
}

# kernel pca
KPCA_PARAMS = {
    #"n_components": None,
    "random_state": RANDOM_STATE,
    "kernel": "rbf",
    "n_jobs": -1,
    "copy_X": False,

    # "auto" did run in a first test. "randomized" is faster and should be used when n_components is low according to
    # sklearn docu/guide.
    "eigen_solver": "randomized"
}

# kmeans
KMEANS_PARAMS = {
    # "n_clusters": 8, # DO NOT SET THIS BECAUSE WE USE BRUTE FORCE TO DETERMINE THIS VALUE
    "batch_size": 256 * 16,  # 256 * cpu threads is suggested in sklearn docu
    "verbose": 0,
    "random_state": RANDOM_STATE,
}

# lda
LDA_PARAMS = {}

# umap
UMAP_PARAMS = {
    # for clustering https://umap-learn.readthedocs.io/en/latest/clustering.html
    "n_neighbors": 30,  # default 15. Should be increased to 30
    "n_jobs": -1,
    #"random_state": RANDOM_STATE, # do not use a random state if you want to run umap on all cores according to to faq https://umap-learn.readthedocs.io/en/latest/faq.html
    "verbose": False,
    "min_dist": 0,
}

########################################################################################################################
#  FEATURE GENERATION RANGES
########################################################################################################################
KMEANS_N_CLUSTER_RANGE = range(2, 51)

########################################################################################################################
#  STACKING PARAMETER DICT
########################################################################################################################
# STACKING_PARAMS = {
#     "estimators": [
#         ("random_forest", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
#         ("logistic_regression", LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, max_iter=1000)),
#         ("knn", KNeighborsClassifier(n_jobs=-1)),
#         ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
#         ("hist_gradient_boosting_classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
#         ("mlp", MLPClassifier(random_state=RANDOM_STATE))
#     ],
#     "final_estimator": LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, max_iter=1000),
#     "cv": 5,
#     "n_jobs": -1,
#     "passthrough": False,
#     "verbose": 2,
# }