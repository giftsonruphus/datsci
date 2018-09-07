import pandas as pd

raw_df_train = pd.read_csv("train.csv")
raw_df_test = pd.read_csv("test.csv")

raw_df_train.head(10)