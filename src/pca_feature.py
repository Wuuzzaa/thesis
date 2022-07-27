import pandas as pd
import warnings
from sklearn.decomposition import PCA, KernelPCA


def _create_pca_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        params: dict,
        prefix: str,
        mode: str,
        random_state: int,
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
    # make a pca instance
    if mode == "pca":
        pca = PCA(**params)

    elif mode == "kpca":
        pca = KernelPCA(**params)

    else:
        raise ValueError(f"mode {mode} is not implemented")

    # X_train pca features dataframe
    if mode == "pca":
        print("pca fit transform train")
        df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

    elif mode == "kpca":
        # kernel pca uses far too much ram even on mid-sized datasets or higher.
        # So we need to use a sample of the train data

        sample_size = 10_000

        if len(X_train) > sample_size:
            warnings.warn("Dataset is huge. Kernel PCA needs huge memory with too much rows. -> Sample of 10000 rows is used")
            X_train_sample = X_train.sample(n=sample_size, random_state=random_state)

            print("pca fit")
            pca.fit(X_train_sample)

            print("pca transform train")
            df_pca_train = pd.DataFrame(pca.transform(X_train)).add_prefix(prefix)

        else:
            print("pca fit transform train")
            df_pca_train = pd.DataFrame(pca.fit_transform(X_train)).add_prefix(prefix)

    else:
        raise ValueError(f"unkown mode {mode}")

    # X_test pca features dataframe
    print("pca transform test")
    df_pca_test = pd.DataFrame(pca.transform(X_test)).add_prefix(prefix)

    return df_pca_train, df_pca_test


