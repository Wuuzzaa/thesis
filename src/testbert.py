import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomTreesEmbedding, StackingClassifier
from sklearn.model_selection import cross_val_predict

from constants import *
import numpy as np
from shutil import make_archive, unpack_archive
import time
from time import perf_counter
import warnings

from src.calc_scores import get_X_train_X_test_y_train_y_test
from pycaret.classification import *



dataset_id = "3"
#dataset_id = "40923" # 1489 # 3

# load data
X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(dataset_folder=DATASETS_FOLDER_PATH.joinpath(dataset_id), random_state=RANDOM_STATE, X_file_name=X_FILTERED_FILE_NAME, y_file_name=Y_FILE_NAME)

# short feedback of the data and classes
print(f"X_train shape: {X_train.shape}")
print(f"target classes: \n{y_train.value_counts()}")
print(f"total {len(y_train.value_counts())} classes\n")

#sample if needed
sample_size = 10_000

if len(X_train) > sample_size:
    warnings.warn(
        f"Sample is used of {sample_size}")
    X_train_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)

else:
    X_train_sample = X_train

# pycaret wants the target in a dataframe column
X_train["y"] = y_train
X_test["y"] = y_test

experiment = setup(
    data=X_train,
    target="y",
    test_data=X_test,
    preprocess=False,
    #data_split_shuffle=False,
    #n_jobs=-1,
    # session_id=RANDOM_STATE # maybe a bug so do not set the random state. ERROR: ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
    #fold=5
    fold_shuffle=True,
)

compare_models()

############################################
# # pycaret example
# from pycaret.datasets import get_data
# dataset = get_data('credit')
#
#
# data = dataset.sample(frac=0.95, random_state=786)
# data_unseen = dataset.drop(data.index)
# data.reset_index(inplace=True, drop=True)
# data_unseen.reset_index(inplace=True, drop=True)
# print('Data for Modeling: ' + str(data.shape))
# print('Unseen Data For Predictions: ' + str(data_unseen.shape))
#
#
#
# exp_clf101 = setup(data = data, target = 'default', session_id=123, fold_shuffle=True)




pass