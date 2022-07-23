import warnings
import pandas as pd
from sklearn.base import ClassifierMixin

from constants import *
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from pathlib import Path
import joblib


def calc_scores(
        random_state: int,
        path_datasets_folder: Path,
        path_results_file: Path,
        mode: str,
        estimator: ClassifierMixin,
        estimator_param_grid: dict,
        cv: int,
        estimator_file_path_suffix: str,
        X_file_name: str,
        y_file_name: str,
        X_train_pca_file_name: str = None,
        X_test_pca_file_name: str = None,

    ):
    """
    Function to calc cross validation score for the train data and score for the test data. The scores are appended to
    the results file.

    There is a check if already done, by checking if the new to generate columns are already in the results dataframe.

    :param y_file_name:
    :param X_file_name:
    :param estimator_file_path_suffix: A str. for the filename like "_random_forest.joblib" do not forget the filetype
    joblib
    :param cv: int. How many cross validation iterations.
    :param estimator: Estimator to use. Must be an estimator supported by sklearn.
    :param estimator_param_grid: A dict with the grid of hyper-parameters for the estimator.
    See sklearn docu:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

    Also look at the parameters for each estimator on sklearn documentation.

    :param random_state:
    :param path_datasets_folder:
    :param path_results_file:
    :param mode:
        "baseline": Runs a random forest on the cleaned data no additional features no feature selection

        "pca_clean": Runs a random forest on the cleaned data with pca additional features no feature selection.
            pca_params = {
            "n_components": 3,
            "random_state": random_state
            }

        "kpca_clean": Runs a rondom forest on the cleaned data with kernel pca additional features no feature selection.

        "pca_and_kpca_clean": Merges the pca and kpca features on clean data as additional features.

    :param X_train_pca_file_name: Needed for any mode with "pca" exept "pca_and_kpca_clean". Just the filename not the path.
    :param X_test_pca_file_name: Needed for any mode with "pca" exept "pca_and_kpca_clean". Just the filename not the path.
    :return: None
    """

    print("")
    print("#"*80)
    print("calc scores".upper())
    print(f"mode: {mode}".upper())
    print("#" * 80)
    print("")

    modes_pca_parameter_needed = ["pca_clean", "kpca_clean"]

    # check needed parameters are not None
    if mode in modes_pca_parameter_needed:
        needed_pca_parameters = [
            X_train_pca_file_name,
            X_test_pca_file_name,
        ]

        if None in needed_pca_parameters:
            raise ValueError(f"One or more parameter for pca is None. Give it a value.")

    # check if mode is valid
    if mode not in CALC_SCORES_MODES:
        raise NotImplemented(f"mode: {mode} is not implemented. Use on of these modes {CALC_SCORES_MODES}")

    # check if already done, by checking if the new to generate columns are already in the results dataframe
    train_cv_score_column_name = f"{mode}{CALC_SCORES_TRAIN_CV_SCORE_COLUMN_NAME_SUFFIX}"
    test_score_column_name = f"{mode}{CALC_SCORES_TEST_SCORE_COLUMN_NAME_SUFFIX}"

    new_columns = [train_cv_score_column_name, test_score_column_name]

    # load the results dataframes columns
    results_df_columns = pd.read_feather(path_results_file).columns.tolist()

    # when all column we would calculate by this mode are already in the results dataframe we can skip it
    if set(new_columns).issubset(results_df_columns):
        warnings.warn(f"All columns for mode: {mode} are already in the results file. Done")
        return

    # make some dicts
    train_cv_scores_dict = {}
    test_scores_dict = {}

    # store the dataset folders
    dataset_folders = []

    # get the dataset folders in the data folder
    for path in path_datasets_folder.iterdir():
        if path.is_dir():
            dataset_folders.append(path)

    # score each dataset
    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}")

        # get X and y train and test splitted
        X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
            dataset_folder=dataset_folder,
            random_state=random_state,
            X_file_name=X_file_name,
            y_file_name=y_file_name,
        )

        if mode == "baseline":
            pass

        elif mode == "pca_clean":
            # load pca features
            df_pca_train    = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_PCA_FILE_NAME))
            df_pca_test     = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_PCA_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_pca_train], axis="columns")
            X_test = pd.concat([X_test, df_pca_test], axis="columns")

        elif mode == "kpca_clean":
            # load kpca features
            df_kpca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_KPCA_FILE_NAME))
            df_kpca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_KPCA_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_kpca_train], axis="columns")
            X_test = pd.concat([X_test, df_kpca_test], axis="columns")

        elif mode == "pca_and_kpca_clean":
            # load pca features
            df_pca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_PCA_FILE_NAME))
            df_pca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_PCA_FILE_NAME))

            # load kpca features
            df_kpca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_KPCA_FILE_NAME))
            df_kpca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_KPCA_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_pca_train, df_kpca_train], axis="columns")
            X_test = pd.concat([X_test, df_pca_test, df_kpca_test], axis="columns")

        # search the best hyperparameter
        # let n_jobs = 1 because estimator will run on all cores -> less memory usage
        estimator_tuned_grid = GridSearchCV(estimator=estimator, param_grid=estimator_param_grid, n_jobs=-1, cv=cv, verbose=1)
        estimator_tuned_grid.fit(X_train, y_train)

        # store the estimator (tuned) on disk
        estimator_file_path = dataset_folder.joinpath(f"{mode}{estimator_file_path_suffix}")
        print(f"store estimator model to: {estimator_file_path}")
        joblib.dump(estimator_tuned_grid.best_estimator_, filename=estimator_file_path)

        # score model on test data and cv score for train data
        test_scores_dict[int(dataset_folder.name)] = estimator_tuned_grid.score(X_test, y_test)

        # get the train score
        # let n_jobs at 1 here to not run out of RAM. The random forest will use all cores so no problem.
        # train_cv_scores_dict[int(dataset_folder.name)] = cross_val_score(estimator_tuned_grid, X_train, y_train, cv=cv, n_jobs=1).mean()
        train_cv_scores_dict[int(dataset_folder.name)] = estimator_tuned_grid.best_score_

    # sort the dicts by keys to get the same order as the dataframe we want to concat with
    test_scores_dict = dict(sorted((test_scores_dict.items())))
    train_cv_scores_dict = dict(sorted((train_cv_scores_dict.items())))

    # make a dataframe from the score lists
    scores_df = pd.DataFrame(data={
        train_cv_score_column_name: train_cv_scores_dict.values(),
        test_score_column_name: test_scores_dict.values()
    })

    # load the results dataframe
    results_df = pd.read_feather(path_results_file)

    # concat the results dataframe with the scores dataframe
    results_df = pd.concat([results_df, scores_df], axis="columns")

    # store results with scores
    results_df.to_feather(path_results_file)


def get_X_train_X_test_y_train_y_test(dataset_folder: Path, random_state: int, X_file_name: str, y_file_name: str):
    """

    :param dataset_folder: Path to a dataset with X and y files in it.
    :param random_state: int
    :return: tuple (DataFrame, DataFrame, Series, Series) X_train, X_test, y_train, y_test
    """
    # get X, y
    path_X = dataset_folder.joinpath(X_file_name)
    path_y = dataset_folder.joinpath(y_file_name)

    X = pd.read_feather(path_X)
    y = pd.read_feather(path_y)["y"]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=0.66)

    # drop index to be able to concat on axis columns
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test
