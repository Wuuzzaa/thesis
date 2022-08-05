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
        sample_size = 1_000
        print("pca mode is kpca")

    else:
        raise ValueError(f"mode {mode} is not implemented")

    if len(X_train) > sample_size:
        warnings.warn(
            f"Dataset is huge. Kernel PCA needs huge memory with too much rows. PCA gets slow. Sample is used of {sample_size}")
        X_train_sample = X_train.sample(n=sample_size, random_state=random_state)

    else:
        X_train_sample = X_train

    # X_train pca features dataframe
    if mode == "pca":
        if params["n_components"] == "mle" and X_train.shape[1] > X_train.shape[0]:
            warnings.warn("n_components='mle' is only supported if n_samples >= n_features. Fall back to n_components=0.8")
            pca.set_params(**{"n_components": 0.8})

    print("pca fit")
    pca.fit(X_train_sample)

    # X_train pca features dataframe
    print("pca transform train")
    df_pca_train = pd.DataFrame(pca.transform(X_train)).add_prefix(prefix)

    # X_test pca features dataframe
    print("pca transform test")
    df_pca_test = pd.DataFrame(pca.transform(X_test)).add_prefix(prefix)

    return df_pca_train, df_pca_test


