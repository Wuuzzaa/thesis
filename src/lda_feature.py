import pandas as pd
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _create_lda_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        params: dict,
        prefix: str,
) -> (pd.DataFrame, pd.DataFrame):
    # set n_componets according to the parameter dict and the restrictions of lda.
    # restrictions from lda:
    # Number of components (<= min(n_classes - 1, n_features)) for dimensionality reduction.
    n_classes = len(y_train.value_counts())
    n_components = min(n_classes - 1, params["n_components"])

    # make a transformer with the correct n_components
    if n_components < params["n_components"]:
        warnings.warn(f"Linear Discriminant Analysis n_components can only be {n_components} and not like the parameter given {params['n_components']}")
        transformer = LinearDiscriminantAnalysis(**params)
        transformer.set_params(**{"n_components": n_components})

    else:
        transformer = LinearDiscriminantAnalysis(**params)

    # fit transform
    transformer.fit(X_train, y_train)

    df_train = pd.DataFrame(transformer.transform(X_train)).add_prefix(prefix)
    df_test = pd.DataFrame(transformer.transform(X_test)).add_prefix(prefix)

    return df_train, df_test
