import pandas as pd
import matplotlib.pyplot as plt
from constants import *
import numpy as np
from shutil import make_archive, unpack_archive

# dataset_id = "40923"
# #dataset_id = "40923" # 1489 # 3
#
# # load data
# X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(dataset_folder=DATASETS_FOLDER_PATH.joinpath(dataset_id), random_state=RANDOM_STATE, X_file_name=X_CLEAN_FILE_NAME, y_file_name=Y_FILE_NAME)
#
# # short feedback of the data and classes
# print(f"X_train shape: {X_train.shape}")
# print(f"target classes: \n{y_train.value_counts()}")
# print(f"total {len(y_train.value_counts())} classes\n")

# sample if needed
# sample_size = 10_000
#
# if len(X_train) > sample_size:
#     warnings.warn(
#         f"Dataset is huge. Kernel PCA needs huge memory with too much rows. PCA gets slow. Sample is used of {sample_size}")
#     X_train_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
#
# else:
#     X_train_sample = X_train

# load results dataframe
df = pd.read_feather(RESULTS_FILE_PATH)
