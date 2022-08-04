from pathlib import Path

########################################################################################################################
# INTEGER
########################################################################################################################
RANDOM_STATE = 42
MAX_FEATURES_FEATURE_SELECTION = 100
N_COMPONENTS_PCA_UMAP_LDA = 2

########################################################################################################################
# FLOAT
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

# stacking
X_TRAIN_CLEAN_STACKING_FILE_NAME    = "stacking_train_clean.feather"
X_TEST_CLEAN_STACKING_FILE_NAME     = "stacking_test_clean.feather"
X_TRAIN_CLEAN_FILTERED_STACKING_FILE_NAME    = "stacking_train_clean_filtered.feather"
X_TEST_CLEAN_FILTERED_STACKING_FILE_NAME     = "stacking_test_clean_filtered.feather"


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
    #todo replace pca_kpca_umap_kmeans and add lda to both modes
    "baseline",

    # features generated on clean data (NOT filtered)
    "pca_clean",
    "kpca_clean",
    "umap_clean",
    "kmeans_clean",
    "lda_clean",
    #"stacking_clean",  # needs to long to create features do not use it.
    "pca_kpca_umap_kmeans_clean",

    # features generated on clean and filtered data
    "pca_clean_filtered",
    "kpca_clean_filtered",
    "umap_clean_filtered",
    "kmeans_clean_filtered",
    "lda_clean_filtered",
    "stacking_clean_filtered",
    "pca_kpca_umap_kmeans_clean_filtered",

]
CALC_SCORES_TRAIN_CV_SCORE_COLUMN_NAME_SUFFIX = "_train_cv_score"
CALC_SCORES_TEST_SCORE_COLUMN_NAME_SUFFIX = "_test_score"
CALC_SCORES_RANDOM_FOREST_FILE_PATH_SUFFIX = "_random_forest.joblib"

########################################################################################################################
# PARAM_GRID_RANDOM_FOREST
# used for hyperparameter tuning
########################################################################################################################
PARAM_GRID_RANDOM_FOREST = {
    "max_depth": [6, None],
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "n_jobs": [-1],
    "class_weight": ["balanced", None],
    "random_state": [1, 42, 1337]
}

