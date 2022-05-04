import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

import gc

warnings.filterwarnings('ignore')

# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label='TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label='TARGET == 1')

    # label the plot
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()
    plt.show()

    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)

"""Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.

    Parameters
    --------
        df (dataframe):
            the dataframe to calculate the statistics on
        group_var (string):
            the variable by which to group df
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated for
            all numeric columns. Each instance of the grouping variable will have
            the statistics (mean, min, max, sum; currently supported) calculated.
            The columns are also renamed to keep track of features created.

    """
def agg_numeric(df, group_var, df_name):
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg


# Function to calculate correlations with the target for a dataframe
def target_corrs(df):
    # List of correlations
    corrs = []

    # Iterate through the columns 
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))

    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)

    return corrs

"""Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable

    Parameters
    --------
    df : dataframe
        The dataframe to calculate the value counts for.

    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row

    df_name : string
        Variable added to the front of column names to keep track of columns


    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.

    """
def count_categorical(df, group_var, df_name):
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    return categorical


# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +" columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Function for merging train dataset with sub data groupby id
def merge_df(df1_train, df1_test, df2, standard, df2_col, df2_name):
    # Calling agg_numeric function to set sub data's numeric columns and store into dataframe
    agg_new = agg_numeric(df2.drop(columns=[df2_col]), group_var=standard, df_name=df2_name)

    # Merge with the training data and sub data's transformed numeric dataframe
    train = df1_train.merge(agg_new, on=standard, how='left')

    # Calling count_categorical function to set sub data's categorical columns and store into dataframe
    counts = count_categorical(df2, group_var=standard, df_name= df2_name)

    # Merge with the training data and sub data's transformed numeric datafram
    train = train.merge(counts, left_on=standard, right_index=True, how='left')

    # Same process with test data
    test = df1_test.merge(agg_new, on=standard, how='left')
    test = test.merge(counts, left_on=standard, right_index=True, how='left')

    train_labels = train['TARGET']

    # Align the dataframes, this will remove the 'TARGET' column
    train, test = train.align(test, join='inner', axis=1)

    train['TARGET'] = train_labels
    
    return train, test

# Function for dropping columns that explored missing value is over threshold : percentage
def handle_missing_values(train, test, percentage):
    # Handling missing values
    # Train dataset missing values
    missing_train = missing_values_table(train)
    print(missing_train.head(10))
    missing_train_vars = list(missing_train.index[missing_train['% of Total Values'] > percentage])
    print(len(missing_train_vars))

    # Test dataset missing values
    missing_test = missing_values_table(test)
    print(missing_test.head(10))
    missing_test_vars = list(missing_test.index[missing_test['% of Total Values'] > percentage])
    print(len(missing_test_vars))

    missing_columns = list(set(missing_test_vars + missing_train_vars))
    print("There are %d columns with more than %d%% missing in either the training or testing data." % (len(
        missing_columns), percentage))

    # Drop the missing columns
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)
    
    return train, test

# Function for saving dataframes into csv
def save_csv(train, test, train_name, test_name):
    train.to_csv(train_name, index=False)
    test.to_csv(test_name, index=False)

# Function for calculating multicollinearity and dropping columns with over the threshold. (Avoiding duplication)
def corr_selection(train, test, threshold):
    # Calculate all correlations in dataframe
    corrs = train.corr()

    corrs = corrs.sort_values('TARGET', ascending=False)

    # Ten most positive correlations
    print(pd.DataFrame(corrs['TARGET'].head(10)))

    # Ten most negative correlations
    print(pd.DataFrame(corrs['TARGET'].dropna().tail(10)))

    # Set the threshold
    threshold = threshold

    # Empty dictionary to hold correlated variables
    above_threshold_vars = {}

    # For each column, record the variables that are above the threshold
    for col in corrs:
        above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

    # Track columns to remove and columns already examined
    cols_to_remove = []
    cols_seen = []
    cols_to_remove_pair = []

    # Iterate through columns and correlated columns
    for key, value in above_threshold_vars.items():
        # Keep track of columns already examined
        cols_seen.append(key)
        for x in value:
            if x == key:
                next
            else:
                # Only want to remove one in a pair
                if x not in cols_seen:
                    cols_to_remove.append(x)
                    cols_to_remove_pair.append(key)

    cols_to_remove = list(set(cols_to_remove))
    print('Number of columns to remove: ', len(cols_to_remove))

    train_corrs_removed = train.drop(columns=cols_to_remove)
    test_corrs_removed = test.drop(columns=cols_to_remove)
    
    return train_corrs_removed, test_corrs_removed


# Function for encoding & scaling with variety combinations of encoders and scalers
def feature_engineering(train, test, encoder, scaler):

    # Set ids and target and store
    label = train['TARGET']
    id_train = train['SK_ID_CURR']
    id_test = test['SK_ID_CURR']
    # Remove the ids and target
    train = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    test = test.drop(columns=['SK_ID_CURR'])

    train, test = train.align(test, join='inner', axis=1)

    # Setting division for train and test dataset
    train['training_set'] = True
    test['training_set'] = False

    # Concat train & test
    df_full = pd.concat([train, test])

    # Function for encoding categorical features with Label Encoder
    # Parameter : df -> Missing value handling, feature selection completed dataset
    def labelEnc(df):
        label = LabelEncoder()
        # select only columns with values are in object type and make new dataframe
        categorical_df = df.select_dtypes(include='object')
        numerical_df = df.select_dtypes(exclude='object')
        # encode all categorical columns with Label Encoder
        for i in range(0, len(categorical_df.columns)):
            df[categorical_df.columns[i]] = label.fit_transform(categorical_df.iloc[:, [i]].values)

        return df

    # Function for encoding categorical features with One-Hot Encoder
    def OneHotEnc(df):
        df = pd.get_dummies(df)
        return df

    # Function for showing heatmap of correlation between each attributes
    # Parameter : df -> Missing value handling, feature selection completed dataset
    def showHeatmap(df):
        df = labelEnc(df)
        heatmap_data = df
        colormap = plt.cm.PuBu
        plt.figure(figsize=(15, 15))
        plt.title("Correlation of Features", y=1.05, size=15)
        sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
                    annot=True, annot_kws={"size": 8})
        plt.show()

    # Function for scaling dataset
    def scaling(scaler, X_train, X_test):

        feature_names = X_train.columns.values.tolist()
        # Make scaler with the parameter inputted with below condition
        Scaler = scaler
        if scaler == 'MinMaxScaler':
            Scaler = MinMaxScaler()
        elif scaler == 'MaxAbsScaler':
            Scaler = MaxAbsScaler()
        elif scaler == 'StandardScaler':
            Scaler = StandardScaler()
        elif scaler == 'RobustScaler':
            Scaler = RobustScaler()

        # Scale the datasets with each scaler chosen with condition above
        X_train_scale = Scaler.fit_transform(X_train)
        X_test_scale = Scaler.transform(X_test)
        train = pd.DataFrame(X_train_scale, columns= feature_names)
        test = pd.DataFrame(X_test_scale, columns= feature_names)

        return train, test

    # Choosing which encoder and scaler will user use
    if encoder =='LabelEncoder':
        # Do label encoding with all categorical features
        df_full = labelEnc(df_full)
        # After encoding split into train & test set with division value
        train = df_full[df_full['training_set'] == True]
        train = train.drop('training_set', axis=1)
        test = df_full[df_full['training_set'] == False]
        test = test.drop('training_set', axis=1)


        # When it meets MinMaxScaler
        if scaler == 'MinMaxScaler':
            # MinMax Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

        # When it meets MaxAbsScaler
        elif scaler == 'MaxAbsScaler':
            # MaxAbs Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

        # When it meets StandardScaler
        elif scaler == 'StandardScaler':
            # Standard Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

        # When it meets RobustScaler
        elif scaler == 'RobustScaler':
            # Robust Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

    elif encoder =='OneHotEncoder':
        # Do label encoding with all categorical features
        df_full = OneHotEnc(df_full)
        # After encoding split into train & test set with division value
        train = df_full[df_full['training_set']==True]
        train = train.drop('training_set', axis=1)
        test = df_full[df_full['training_set'] == False]
        test = test.drop('training_set', axis=1)

        # When it meets MinMaxScaler
        if scaler == 'MinMaxScaler':
            # MinMax Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

        # When it meets MaxAbsScaler
        elif scaler == 'MaxAbsScaler':
            # MaxAbs Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

        # When it meets StandardScaler
        elif scaler == 'StandardScaler':
            # Standard Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

        # When it meets RobustScaler
        elif scaler == 'RobustScaler':
            # Robust Scaling using scaling function defined above
            train, test = scaling(scaler, train, test)

    # After all feature-engineering concat id, label with train/ concat id with test
    train = pd.concat([train, label], axis=1)
    train = pd.concat([id_train, train], axis=1)
    test = pd.concat([id_test, test], axis=1)
    return train, test

# Function for training data with model
def model(train, test, id, label, n_folds=5):

    features = np.array(train)
    test_features = np.array(test)

    # Extract feature names
    feature_names = list(train.columns)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Create the kfold object
    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=50)
    X = train.drop([id], 1)
    X = train.drop([label], 1)
    y = train[label]

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=121)

    eval_set = [(X_test, y_test)]

    # Create the model
    model = lgb.LGBMClassifier(num_leaves=36, bagging_fraction=0.5101, max_depth=8, lambda_l1=3.891, lambda_l2=2.61,
                               min_split_gain=0.05, min_child_weight=40.96, colsample_bytree=0.933)

    model.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=eval_set)

    print('========================= LGB Classifier ==========================')

    # predict y
    lgb_y_pred = model.predict(X_test)
    feature_importance_values = model.feature_importances_
    # predict proba y
    # 서브미션시 사용할 변수
    lgb_y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_prediction = model.predict_proba(test_features)[:, 1]

    # precision, recall, f1
    lgb_s = round(accuracy_score(y_test, lgb_y_pred), 6)
    print("accuracy score :", lgb_s)
    lgb_p = round(precision_score(y_test, lgb_y_pred), 6)
    print("precision score :", lgb_p)
    lgb_r = round(recall_score(y_test, lgb_y_pred), 6)
    print("recall score :", lgb_r)
    lgb_f = round(f1_score(y_test, lgb_y_pred), 6)
    print("F1 score :", lgb_f)
    lgb_roc_auc = roc_auc_score(y_test, lgb_y_pred_proba)
    print("AUC :", lgb_roc_auc)

    # Make confusion matrix
    lgb_cf = confusion_matrix(y_test, lgb_y_pred)
    lgb_total = np.sum(lgb_cf, axis=1)
    lgb_cf = lgb_cf / lgb_total[:, None]
    lgb_cf = pd.DataFrame(lgb_cf)

    # visualization
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix with LGB")
    sns.heatmap(lgb_cf, annot=True, annot_kws={"size": 20})
    plt.savefig('lgbm_confusion_matrix-01.png')
    plt.show()

    def rocvis(y_test, pred_proba, label):
        # FPR, TPR values are returned according to the threshold
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
        # Draw roc curve with plot
        plt.plot(fpr, tpr, label=label)
        # Draw diagonal straight line
        plt.plot([0, 1], [0, 1], linestyle='--')

    rocvis(y_test, lgb_y_pred_proba, 'LGB Classifier')
    # Setting FPR axis of X and scaling into unit of 0.1, labels
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend(fontsize=18)
    plt.title("Roc Curve", fontsize=25)
    plt.savefig('lgbm_roc-01.png')
    plt.show()

    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test[id], 'TARGET': test_prediction})
    submission.to_csv('prediction.csv', index=False)

    # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(30))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_feature_importances-01.png')
    plt.show()
