import warnings
import pandas as pd
from sklearn.base import ClassifierMixin

from constants import *
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from pathlib import Path
import joblib

from src.util import get_sub_folders, print_function_header


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
        "baseline": Runs a random forest on the cleaned data no additional features with feature selection

        ---
        MODES WITH NEW FEATURES ON CLEANED DATA (NOT FILTERED)
        ---

        "pca_clean": Runs a random forest on the cleaned data with pca additional features with feature selection.
        The new features were generated on clean data not on filtered data.

        "kpca_clean": Runs a rondom forest on the cleaned data with kernel pca additional features with feature selection.
        The new features were generated on clean data not on filtered data.

        "umap_clean": Runs a random forest on the cleaned data with umap additional features with feature selection.
        The new features were generated on clean data not on filtered data.

        "kmeans_clean": Runs a random forest on the cleaned data with kmeans additional features with feature selection.
        The new features were generated on clean data not on filtered data.

        "lda_clean" Runs a random forest on the cleaned data with lda additional features with feature selection.
        The new features were generated on clean data not on filtered data.

        "pca_kpca_umap_kmeans_clean". Runs a random forest on the cleaned data with pca, kpca, umap and kmeans
        additional features with feature selection.
        The new features were generated on clean data not on filtered data.

        ---
        MODES WITH NEW FEATURES ON CLEANED AND FILTERED DATA
        ---

        "pca_clean_filtered": Runs a random forest on the cleaned data with pca additional features with feature selection.
        The new features were generated on cleaned and filtered data.

        "kpca_clean_filtered": Runs a rondom forest on the cleaned data with kernel pca additional features with feature selection.
        The new features were generated on cleaned and filtered data.

        "umap_clean_filtered": Runs a random forest on the cleaned data with umap additional features with feature selection.
        The new features were generated on cleaned and filtered data.

        "kmeans_clean_filtered_filtered": Runs a random forest on the cleaned data with kmeans additional features with feature selection.
        The new features were generated on cleaned and filtered data.

        "lda_clean_filtered": Runs a random forest on the cleaned data with lda additional features with feature selection.
        The new features were generated on cleaned and filtered data.

        "pca_kpca_umap_kmeans_clean". Runs a random forest on the cleaned data with pca, kpca, umap and kmeans
        additional features with feature selection.
        The new features were generated on cleaned and filtered data.

    :return: None
    """
    # print header
    print_function_header(f"calc scores\nmode: {mode}")

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

    # get all dataset folders
    dataset_folders = get_sub_folders(path_datasets_folder)

    # score each dataset
    for dataset_folder in tqdm(dataset_folders):
        print()
        print("---")
        print(f"current folder: {dataset_folder}")
        print("---")

        # get X and y train and test splitted
        X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
            dataset_folder=dataset_folder,
            random_state=random_state,
            X_file_name=X_file_name,
            y_file_name=y_file_name,
        )

        # short feedback of the data and classes
        print("Whiteout new features:")
        print(f"X_train shape: {X_train.shape}")
        print(f"target classes: \n{y_train.value_counts()}")

        if mode == "baseline":
            # no features to add
            pass

        # ---
        # MODES WITH NEW FEATURES ON CLEANED DATA (NOT FILTERED)
        # ---
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

        elif mode == "umap_clean":
            # load umap features
            df_umap_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_UMAP_FILE_NAME))
            df_umap_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_UMAP_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_umap_train], axis="columns")
            X_test = pd.concat([X_test, df_umap_test], axis="columns")

        elif mode == "kmeans_clean":
            # load kmeans features
            df_kmeans_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_KMEANS_FILE_NAME))
            df_kmeans_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_KMEANS_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_kmeans_train], axis="columns")
            X_test = pd.concat([X_test, df_kmeans_test], axis="columns")

        elif mode == "lda_clean":
            # load lda features
            df_lda_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_LDA_FILE_NAME))
            df_lda_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_LDA_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_lda_train], axis="columns")
            X_test = pd.concat([X_test, df_lda_test], axis="columns")

        elif mode == "pca_kpca_umap_kmeans_clean":
            # load pca features
            df_pca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_PCA_FILE_NAME))
            df_pca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_PCA_FILE_NAME))

            # load kpca features
            df_kpca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_KPCA_FILE_NAME))
            df_kpca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_KPCA_FILE_NAME))

            # load umap features
            df_umap_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_UMAP_FILE_NAME))
            df_umap_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_UMAP_FILE_NAME))

            # load kmeans features
            df_kmeans_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_KMEANS_FILE_NAME))
            df_kmeans_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_KMEANS_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_pca_train, df_kpca_train, df_umap_train, df_kmeans_train], axis="columns")
            X_test = pd.concat([X_test, df_pca_test, df_kpca_test, df_umap_test, df_kmeans_test], axis="columns")

        # ---
        # MODES WITH NEW FEATURES ON CLEANED AND FILTERED DATA
        # ---

        elif mode == "pca_clean_filtered":
            # load pca features
            df_pca_train    = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_PCA_FILE_NAME))
            df_pca_test     = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_PCA_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_pca_train], axis="columns")
            X_test = pd.concat([X_test, df_pca_test], axis="columns")

        elif mode == "kpca_clean_filtered":
            # load kpca features
            df_kpca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_KPCA_FILE_NAME))
            df_kpca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_KPCA_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_kpca_train], axis="columns")
            X_test = pd.concat([X_test, df_kpca_test], axis="columns")

        elif mode == "umap_clean_filtered":
            # load umap features
            df_umap_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_UMAP_FILE_NAME))
            df_umap_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_UMAP_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_umap_train], axis="columns")
            X_test = pd.concat([X_test, df_umap_test], axis="columns")

        elif mode == "kmeans_clean_filtered":
            # load kmeans features
            df_kmeans_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_KMEANS_FILE_NAME))
            df_kmeans_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_KMEANS_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_kmeans_train], axis="columns")
            X_test = pd.concat([X_test, df_kmeans_test], axis="columns")

        elif mode == "lda_clean_filtered":
            # load lda features
            df_lda_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_LDA_FILE_NAME))
            df_lda_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_LDA_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_lda_train], axis="columns")
            X_test = pd.concat([X_test, df_lda_test], axis="columns")

        elif mode == "pca_kpca_umap_kmeans_clean_filtered":
            # load pca features
            df_pca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_PCA_FILE_NAME))
            df_pca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_PCA_FILE_NAME))

            # load kpca features
            df_kpca_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_KPCA_FILE_NAME))
            df_kpca_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_KPCA_FILE_NAME))

            # load umap features
            df_umap_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_UMAP_FILE_NAME))
            df_umap_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_UMAP_FILE_NAME))

            # load kmeans features
            df_kmeans_train = pd.read_feather(dataset_folder.joinpath(X_TRAIN_CLEAN_FILTERED_KMEANS_FILE_NAME))
            df_kmeans_test = pd.read_feather(dataset_folder.joinpath(X_TEST_CLEAN_FILTERED_KMEANS_FILE_NAME))

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_pca_train, df_kpca_train, df_umap_train, df_kmeans_train], axis="columns")
            X_test = pd.concat([X_test, df_pca_test, df_kpca_test, df_umap_test, df_kmeans_test], axis="columns")

        else:
            raise NotImplemented(f"{mode} not implemented")

        # path to the tuned model
        estimator_file_path: Path = dataset_folder.joinpath(f"{mode}{estimator_file_path_suffix}")

        # set the n_jobs for the cross validation according to the dataset size. On a huge dataset i ran out of RAM when
        # i run cross validation in parallel
        n_jobs_cv = -1
        if len(X_train) > 10_000:
            n_jobs_cv = 1

        print(f"n_jobs for cross validation: {n_jobs_cv}")

        # hyperparameter tuned model already exists
        if estimator_file_path.exists():
            print(f"found model @{estimator_file_path} load it and score on it.")

            # load model from disk
            model = joblib.load(estimator_file_path)

            # score model on test data and cv score for train data
            test_scores_dict[int(dataset_folder.name)] = model.score(X_test, y_test)

            # get the train score
            # let n_jobs at 1 here to not run out of RAM. The random forest will use all cores so no problem.
            train_cv_scores_dict[int(dataset_folder.name)] = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=n_jobs_cv).mean()

        # search the best hyperparameter and score with them
        else:
            print("no model found. search for best hyperparameter and score on them")

            # let n_jobs = 1 because estimator will run on all cores -> less memory usage
            estimator_tuned_grid = GridSearchCV(estimator=estimator, param_grid=estimator_param_grid, n_jobs=n_jobs_cv, cv=cv, verbose=1)
            estimator_tuned_grid.fit(X_train, y_train)

            # store the estimator (tuned) on disk
            print(f"store estimator model to: {estimator_file_path}")
            joblib.dump(estimator_tuned_grid.best_estimator_, filename=estimator_file_path)

            # score model on test data and cv score for train data
            test_scores_dict[int(dataset_folder.name)] = estimator_tuned_grid.score(X_test, y_test)

            # get the train score
            train_cv_scores_dict[int(dataset_folder.name)] = estimator_tuned_grid.best_score_

        # give feedback of the train and test score
        print("-")
        print(f"train score: {train_cv_scores_dict[int(dataset_folder.name)]}")
        print(f"test score: {test_scores_dict[int(dataset_folder.name)]}")
        print("-\n")

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=TRAIN_TEST_SPLIT_TRAIN_SIZE)

    # drop index to be able to concat on axis columns
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test
