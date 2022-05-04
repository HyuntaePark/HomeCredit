import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import missingno as msno
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import module

import gc
import re

warnings.filterwarnings('ignore')

# print("pandas version: ", pd.__version__)
# pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 150)


# read dataset(removed outlier)
#application_train = pd.read_csv('application_train.csv.csv', encoding='utf-8')
application_train = pd.read_csv('train_remove_outlier.csv', encoding='utf-8')
application_test = pd.read_csv('application_test.csv', encoding='utf-8')
#bureau = pd.read_csv('bureau.csv', encoding='utf-8')
bureau = pd.read_csv('bureau_remove_outlier.csv', encoding='utf-8')
bureau_balance = pd.read_csv('bureau_balance.csv', encoding='utf-8')
previous_application = pd.read_csv('previous_application.csv', encoding='utf-8')
credit_card_balance = pd.read_csv('credit_card_balance.csv', encoding='utf-8')
POS_CASH_balance = pd.read_csv('POS_CASH_balance.csv', encoding='utf-8')
installments_payments = pd.read_csv('installments_payments.csv', encoding='utf-8')


# Glimpse the data
print("application_train -  rows:",application_train.shape[0]," columns:", application_train.shape[1])
print("application_test -  rows:",application_test.shape[0]," columns:", application_test.shape[1])
print("bureau -  rows:",bureau.shape[0]," columns:", bureau.shape[1])
#print("bureau_balance -  rows:",bureau_balance.shape[0]," columns:", bureau_balance.shape[1])
#print("credit_card_balance -  rows:",credit_card_balance.shape[0]," columns:", credit_card_balance.shape[1])
#print("installments_payments -  rows:",installments_payments.shape[0]," columns:", installments_payments.shape[1])
print("previous_application -  rows:",previous_application.shape[0]," columns:", previous_application.shape[1])
#print("POS_CASH_balance -  rows:",POS_CASH_balance.shape[0]," columns:", POS_CASH_balance.shape[1])

# Details about application_train.csv set
print(application_train.head())
print(application_train.columns.values)

# Details about application_test.csv set
print(application_test.head())
print(application_test.columns.values)

# Details about bureau.csv set
print(bureau.head())
print(bureau.columns.values)

# Details about bureau_balance.csv set
# print(bureau_balance.head())

# Details about previous_application.csv set
print(previous_application.head())
print(previous_application.columns.values)


# Details about credit_card_balance.csv set
# print(credit_card_balance.head())
# print(credit_card_balance.columns.values)


# Details about installments_payments.csv set
# print(installments_payments.head())
# print(installments_payments.columns.values)


# Details about POS_CASH_balance.csv set
# print(POS_CASH_balance.head())
# print(POS_CASH_balance.columns.values)

# Set Id, label
label = 'TARGET'
id = 'SK_ID_CURR'

# Merging into one dataframe using merge_df function
# merge_df( merge_df1, merge_df2, target_merge_df, Id, Id in target_merge_df, name for new derived variables)
train_merged, test_merged = module.merge_df(application_train, application_test, bureau, id, 'SK_ID_BUREAU', 'bureau')
train_merged, test_merged = module.merge_df(train_merged, test_merged, previous_application, id, 'SK_ID_PREV', 'prev_app')

# Visualization comparing before & after merging
print('==========BEFORE MERGE==========')
print('Training Data Shape: ', application_train.shape)
print('Testing Data Shape: ', application_test.shape)
print('===========AFTER MERGE==========')
print('Training Data Shape: ', train_merged.shape)
print('Testing Data Shape: ', test_merged.shape)

# Handling missing values on train, test dataset
# handle_missing_values(df1, df2, threshold : "over threshold percentage columns will be dropped")
train_ms, test_ms = module.handle_missing_values(train_merged, test_merged, 50)

# Visualization comparing before & after handling missing values
print('==========BEFORE HANDLING MISSING VALUE==========')
print('Training Data Shape: ', train_merged.shape)
print('Testing Data Shape: ', test_merged.shape)
print('===========AFTER HANDLING MISSING VALUE==========')
print('Training Data Shape: ', train_ms.shape)
print('Testing Data Shape: ', test_ms.shape)

# # Save file into CSV
# # module.save_csv(train, test, 'train_new.csv', 'test_new.csv')
# Plotting kde_plot for some of derived variables
module.kde_target('prev_app_NAME_CONTRACT_STATUS_Refused_count_norm', train_ms)
module.kde_target('bureau_CREDIT_ACTIVE_Active_count_norm', train_ms)

# Remove low correlated features in each dataset
# corr_selection(df1, df2, threshold : correlation for each independent variables
train_co, test_co = module.corr_selection(train_ms, test_ms, 0.7)

# Visualization comparing before & after Multicollinearity handling
print('==========BEFORE HANDLING MULTICOLLINEARITY==========')
print('Training Data Shape: ', train_ms.shape)
print('Testing Data Shape: ', test_ms.shape)
print('===========AFTER HANDLING MULTICOLLINEARITY==========')
print('Training Data Shape: ', train_co.shape)
print('Testing Data Shape: ', test_co.shape)



# # Save file into CSV
# # module.save_csv

# Encoding & Scaling for the final dataset
# feature_engineering(df1, df2, ['LabelEncoder', 'OneHotEncoder'], ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'])
train_fe, test_fe = module.feature_engineering(train_co, test_co, 'OneHotEncoder', 'StandardScaler')

# Visualization comparing before & after Feature engineering
print('==========BEFORE FEATURE ENGINEERING==========')
print('Training Data Shape: ', train_co.shape)
print('Testing Data Shape: ', test_co.shape)
print('===========AFTER FEATURE ENGINEERING==========')
print('Training Data Shape: ', train_fe.shape)
print('Testing Data Shape: ', test_fe.shape)
# Save file into csv
module.save_csv(train_fe, test_fe, 'train_final.csv', 'test_final.csv')

# Setting dataframe's column names with regular expresssion(Excluding special characters) for lightGBM model
train_fe = train_fe.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
test_fe = test_fe.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Train final dataset into model
# model(df1, df2, id, target, number of folds
module.model(train_fe, test_fe, id, label, 5)








