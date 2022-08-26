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

# select columns
# select dataset id and all test score related columns
df = df[df.columns[df.columns.str.contains("dataset_id|test_score")]]

# remove the compare with baseline columns
df = df[df.columns[df.columns.str.contains(">|change") == False]]

# make plots

# set dataset id as index
df_plot = df.set_index("dataset_id")

# rename the column for better readability
columns = list(df_plot.columns)
columns = [column.replace("_", " ").replace("test score", "").rstrip() for column in columns]

df_plot.columns = columns

df_plot = df_plot.head(1).T.sort_values(by=3, ascending=False).T

print()

# adjust color of baseline score
colors = ["blue"] * len(columns)
index_baseline = list(df_plot.columns).index("baseline filtered")
colors[index_baseline] = "red"

plt.figure(figsize=(25, 20))

# make a vertical line for the baseline score
plt.axvline(x=df_plot.values.squeeze()[index_baseline], color='g')

# horizontal barplot
plt.barh(
    y=df_plot.columns,
    width=df_plot.values.squeeze(),
    color=colors,
)

# add accuracy to the bars
for index, value in enumerate(df_plot.values.squeeze()):
    plt.text(value, index, str(value.round(4)))


#plt.show()
plt.savefig(fname="temp_plot")
pass
