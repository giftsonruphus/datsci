import pandas as pd
pd.set_option('display.max_rows', 10)

raw_df_train = pd.read_csv("D:\\workspaces\\ds\\de\\train.csv")
raw_df_test = pd.read_csv("D:\\workspaces\\ds\\de\\test.csv")

raw_df_train.head(10)

raw_df_train.dtypes

raw_df_train.describe()

categorical_variables  = raw_df_train.dtypes.loc[raw_df_train.dtypes=='object'].index
categorical_variables

raw_df_train[categorical_variables]

raw_df_train[categorical_variables].apply(lambda x: len(x.unique()), axis='index')

raw_df_train['Race'].value_counts()

raw_df_train['Race'].value_counts()/raw_df_train.shape[0] * 100

raw_df_train['Native.Country'].value_counts()

raw_df_train['Native.Country'].value_counts()/raw_df_train.shape[0]*100
