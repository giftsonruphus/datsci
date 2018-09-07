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

#Q1) : Calculate the Interquartile range(IQR) of the variable "Age" in given data.
raw_df_train['Age'].quantile(0.75) - raw_df_train['Age'].quantile(0.25)


#Q2) : What % of people in the train dataset have >50k income.
raw_df_train['Income.Group'].value_counts().loc['>50K']/raw_df_train.shape[0]*100

#Q3) : What % of people are divorced in the dataset.
get_ipython().run_line_magic('timeit', "raw_df_train['Marital.Status'].value_counts()['Divorced']")

get_ipython().run_line_magic('timeit', "raw_df_train['Marital.Status'].value_counts().loc['Divorced']")

get_ipython().run_line_magic('timeit', "raw_df_train[raw_df_train['Marital.Status'] == 'Divorced']['Marital.Status']")

get_ipython().run_line_magic('timeit', "len(raw_df_train[raw_df_train['Marital.Status'] == 'Divorced'])")

get_ipython().run_line_magic('timeit', "len(raw_df_train[raw_df_train['Marital.Status'] == 'Divorced']['Marital.Status'])")

get_ipython().run_line_magic('timeit', "raw_df_train[raw_df_train['Marital.Status'] == 'Divorced']['Marital.Status'].count()")

get_ipython().run_line_magic('timeit', 'len(raw_df_train)')

get_ipython().run_line_magic('timeit', 'raw_df_train.shape[0]')

len(raw_df_train[raw_df_train['Marital.Status'] == 'Divorced'])/len(raw_df_train)*100