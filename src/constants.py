from pathlib import Path

# RANDOM STATE
RANDOM_STATE = 42

########################################################################################################################
# FILENAMES
# Filenames NO Paths see above
# suffix: _FILE_NAME
########################################################################################################################
# X and y
X_CLEAN_FILE_NAME = "X_clean.feather"
Y_FILE_NAME = "y.feather"

# pca
X_TRAIN_CLEAN_PCA_FILE_NAME = "pca_train_clean.feather"
X_TEST_CLEAN_PCA_FILE_NAME = "pca_test_clean.feather"

# kpca
X_TRAIN_CLEAN_KPCA_FILE_NAME = "kpca_train_clean.feather"
X_TEST_CLEAN_KPCA_FILE_NAME = "kpca_test_clean.feather"

# results dataframe file
RESULTS_DATAFRAME = "results.feather"

########################################################################################################################
# PATHS
# Paths from pathlib to folders or files
# suffix: _FOLDER_PATH or _FILE_PATH
########################################################################################################################

# FOLDERS
DATA_FOLDER_PATH = Path("..//data")
DATASETS_FOLDER_PATH = DATA_FOLDER_PATH.joinpath("datasets")
RESULTS_FOLDER_PATH = DATA_FOLDER_PATH.joinpath("results")

# FILES
RESULTS_FILE_PATH = RESULTS_FOLDER_PATH.joinpath(RESULTS_DATAFRAME)


