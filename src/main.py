import warnings
from pathlib import Path
import openml
import pandas as pd
from load_and_clean_suite_datasets import load_and_clean_suite_datasets

if __name__ == "__main__":
    random_state = 42

    # load suite
    # https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html#sphx-glr-examples-20-basic-simple-suites-tutorial-py
    suite = openml.study.get_suite(99)
    load_and_clean_suite_datasets(suite, random_state)


