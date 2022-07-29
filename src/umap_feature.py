import pandas as pd
from umap import UMAP


def _create_umap_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict,
    prefix: str,
):
    transformer = UMAP(**params)

    # UMAP can use the target info (y_train) too. Seems to overfit too hard so do not use it.
    # When the whole train data is used it overfits just use less data?
    transformer.fit(X_train)

    df_train = pd.DataFrame(transformer.transform(X_train)).add_prefix(prefix)
    df_test = pd.DataFrame(transformer.transform(X_test)).add_prefix(prefix)

    #todo POSSIBLE BUG BUT NOT ON THOSE DATASETS SO LET IT RUN ;-)

    # Traceback (most recent call last):
    #   File "C:\Program Files\JetBrains\PyCharm 2022.1.3\plugins\python\helpers\pydev\pydevd.py", line 1491, in _exec
    #     pydev_imports.execfile(file, globals, locals)  # execute the script
    #   File "C:\Program Files\JetBrains\PyCharm 2022.1.3\plugins\python\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
    #     exec(compile(contents+"\n", file, 'exec'), glob, loc)
    #   File "C:/Users/jonas/PycharmProjects/thesis/src/testbert.py", line 37, in <module>
    #     X_test_trans = transformer.transform(X_test)
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\umap\umap_.py", line 2896, in transform
    #     indices, dists = self._knn_search_index.query(
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\pynndescent_.py", line 1627, in query
    #     self._init_search_graph()
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\pynndescent_.py", line 981, in _init_search_graph
    #     self._search_forest = [
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\pynndescent_.py", line 982, in <listcomp>
    #     convert_tree_format(tree, self._raw_data.shape[0])
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\rp_trees.py", line 1158, in convert_tree_format
    #     hyperplane_dim = dense_hyperplane_dim(tree.hyperplanes)
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\rp_trees.py", line 1140, in dense_hyperplane_dim
    #     raise ValueError("No hyperplanes of adequate size were found!")
    # ValueError: No hyperplanes of adequate size were found!

    return df_train, df_test
