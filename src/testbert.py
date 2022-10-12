import math

from src.autoencoder_feature import _create_autoencoder_features
import pandas as pd

from src.calc_scores import get_X_train_X_test_y_train_y_test

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import numpy as np
from constants import *
import matplotlib.pyplot as plt
import seaborn as sns
from constants import *
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from umap import UMAP
import keras
from keras import layers

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

colors = ["navy", "turquoise", "darkorange"]
lw = 2

figsize=(10, 10)

dataset_id = "46"
#dataset_id = "40923" # 1489 # 3

# load data
X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
    dataset_folder=DATASETS_FOLDER_PATH.joinpath(dataset_id), random_state=RANDOM_STATE,
    X_file_name=X_FILTERED_FILE_NAME, y_file_name=Y_FILE_NAME)

X_train_trans, X_test_trans = _create_autoencoder_features(X_train, X_test, AUTOENCODER_PARAMS, "autoencoder_", RANDOM_STATE)

colors_dict = {
    0: "navy",
    1: "turquoise",
    2: "darkorange",
}

# plot df based of X_train
df = X_train_trans.copy()
df["y"] = y_train
df["color"] = df["y"].map(colors_dict)
print("")