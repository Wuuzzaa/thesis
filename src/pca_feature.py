import pandas as pd
import warnings
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from kneed import KneeLocator


def _create_pca_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        params: dict,
        prefix: str,
        mode: str,
        random_state: int,
        early_stopping: int = 10,
        max_n_components_to_create: int = 100,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Function to create pca features.

    :param X_train:
    :param X_test:
    :param params: dict with the parameters for sklearns PCA or Kernel PCA
    :param prefix: prefix for the columnname
    :param mode: on of "pca" or "kpca" which one to use
    :param random_state:
    :return:
    """
    # kernel pca uses far too much ram even on mid-sized datasets or higher.
    # So we need to use a sample of the train data
    # pca gets slow too. just sample.

    # make a pca instance
    if mode == "pca":
        pca = PCA(**params)
        sample_size = 10_000
        print("pca mode is pca")

    elif mode == "kpca":
        pca = KernelPCA(**params)
        sample_size = 10_000
        print("pca mode is kpca")

    else:
        raise ValueError(f"mode {mode} is not implemented")

    if len(X_train) > sample_size:
        warnings.warn(
            f"Dataset is huge. Kernel PCA needs huge memory with too much rows. PCA gets slow. Sample is used of {sample_size}")
        X_train_sample = X_train.sample(n=sample_size, random_state=random_state)

    else:
        X_train_sample = X_train.copy()

    optimal_n_components = _search_optimal_n_components(
        pca=pca,
        random_state=random_state,
        X_train=X_train,
        max_n_components_to_create=max_n_components_to_create,
        X_train_sample=X_train_sample,
        y_train=y_train,
        early_stopping=early_stopping,
    )

    pca.set_params(**{"n_components": optimal_n_components})

    print("pca fit")
    pca.fit(X_train_sample)

    # X_train pca features dataframe
    print("pca transform train")
    df_pca_train = pd.DataFrame(pca.transform(X_train)).add_prefix(prefix)

    # X_test pca features dataframe
    print("pca transform test")
    df_pca_test = pd.DataFrame(pca.transform(X_test)).add_prefix(prefix)

    return df_pca_train, df_pca_test


def _search_optimal_n_components(
        pca,
        random_state,
        X_train,
        max_n_components_to_create,
        X_train_sample,
        y_train,
        early_stopping,
) -> int:
    """

    :param pca:
    :param random_state:
    :param X_train:
    :param max_n_components_to_create:
    :param X_train_sample:
    :param y_train:
    :param early_stopping:
    :return: int
    """
    # make a random forest instance
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    # cv_scores_dict
    # key n_components: int
    # value cv_score: float
    cv_scores_dict = {}

    #
    last_knee = None
    last_knee_found_n_components = None

    range_upper_bound = min(max_n_components_to_create + 1, len(X_train.columns))

    # test different parameters for n_components
    for n_components in range(1, range_upper_bound):
        print("\n---")
        print(f"test n_components = {n_components}")
        pca.set_params(**{"n_components": n_components})

        print("fit...")
        pca.fit(X_train_sample)

        print("transform train data...")
        X_train_trans = pd.DataFrame(pca.transform(X_train)).add_prefix("pca_")

        cv_score = cross_val_score(rf, X_train_trans, y_train, cv=5).mean()

        cv_scores_dict[n_components] = cv_score

        print(f"Cross validation score pca: {cv_score}")

        # condition of a knee is at least two points
        if len(cv_scores_dict) > 1:
            # detect the knee
            kneedle = KneeLocator(
                x=list(cv_scores_dict.keys()),
                y=list(cv_scores_dict.values()),
                curve="concave",
                direction="increasing"
            )

            # knee
            if kneedle.knee:
                if last_knee is None:
                    last_knee = kneedle.knee
                    last_knee_found_n_components = n_components

                elif kneedle.knee > last_knee:
                    last_knee = kneedle.knee
                    last_knee_found_n_components = n_components

                print(f"knee found at x={kneedle.knee}")

                # check for early stopping
                rounds_since_last_knee = n_components - last_knee_found_n_components
                print(f"rounds since last knee-point found: {rounds_since_last_knee}")

                if rounds_since_last_knee >= early_stopping:
                    print(f"Early stopping of {early_stopping} reached.")
                    break

            else:
                print(f"no knee found so far")

    # optimal n_components should be the last found knee point
    optimal_n_components = last_knee
    print("\n---")
    print(f"optimal n_components = {optimal_n_components}")
    print("---\n")

    # optimal n_components can be None when there are just a few features to test and no knee can be detected.
    if optimal_n_components is None:
        warnings.warn(f"optimal n_components is None. Fall back to the best cv_score n_components")
        optimal_n_components_by_cv_score = max(cv_scores_dict, key=cv_scores_dict.get)

        print(f"optimal n_components determined by cv_score: {optimal_n_components_by_cv_score}")

        return optimal_n_components_by_cv_score

    return optimal_n_components

