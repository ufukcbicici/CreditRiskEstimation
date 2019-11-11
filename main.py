import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import expon

# Payment table
payment_column_names = ["id", "OVD_t1", "OVD_t2", "OVD_t3", "OVD_sum", "pay_normal", "prod_code", "prod_limit",
                        "update_date",
                        "new_balance", "highest_balance", "report_date"]
payment_ignore_columns = {"prod_limit", "report_date"}
payment_column_types = \
    {"OVD_t1": "numerical",
     "OVD_t2": "numerical",
     "OVD_t3": "numerical",
     "OVD_sum": "numerical",
     "pay_normal": "numerical",
     "new_balance": "numerical",
     "highest_balance": "numerical",
     "prod_limit": "numerical",
     "prod_code": "categorical",
     "update_date": "date",
     "report_date": "date"}
t0_moment = pd.Timestamp(year=2004, month=1, day=1, hour=0)
payment_date_bin_count = 8

# Customer table
customer_column_names = ["label", "id", "fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7", "fea_8", "fea_9",
                         "fea_10", "fea_11"]
customer_ignore_columns = {}

customer_column_types = \
    {"fea_1": "categorical",
     "fea_2": "numerical",
     "fea_3": "categorical",
     "fea_4": "numerical",
     "fea_5": "categorical",
     "fea_6": "numerical",
     "fea_7": "categorical",
     "fea_8": "numerical",
     "fea_9": "categorical",
     "fea_10": "numerical",
     "fea_11": "numerical"}


def count_missing_entries(data):
    # Get the count of missing values in every column
    nan_counts_arr = pd.isna(data)
    nan_counts_dict = {}
    nan_percents_dict = {}
    row_count = data.shape[0]
    for col_name in data.columns:
        nan_counts_dict[col_name] = np.sum(nan_counts_arr[col_name])
        nan_percents_dict[col_name] = nan_counts_dict[col_name] / row_count
    print(nan_counts_dict)
    print(nan_percents_dict)
    # df = pd.DataFrame.from_dict({k: [v] for k, v in nan_counts_dict.items()})
    # display("Nan Counts Per Column")
    # display(pd.DataFrame.from_dict({k: [v] for k, v in nan_counts_dict.items()}))
    # display("Nan Percents Per Column")
    # display(pd.DataFrame.from_dict(({k: [100.0 * v] for k, v in nan_percents_dict.items()}))


def impute_missing_value(data, col_name, col_type):
    if col_type == "numerical":
        strategy = "median"
    elif col_type == "categorical":
        strategy = "most_frequent"
    else:
        raise NotImplementedError()
    col_values = data[col_name]
    imputer = SimpleImputer(strategy=strategy, copy=True)
    values = col_values.to_numpy(copy=True)
    values = np.expand_dims(values, axis=1)
    imputer.fit(values)
    imputed_values = imputer.transform(values)
    imputed_values = np.reshape(imputed_values, newshape=(imputed_values.shape[0],))
    data[col_name] = imputed_values


def analyze_column(data, col_name, col_types):
    column = data.loc[:, col_name]

    print("Column:{0}".format(col_name))
    # Step 0: Get column type
    print("Type:{0}".format(column.dtypes))
    if col_types[col_name] == "numerical":
        # Step 1: Get total number of unique values
        unique_count = column.nunique()
        unique_ratio = unique_count / column.shape[0]
        # Step 2: Get the data distribution
        max_val = column.max(axis=0)
        min_val = column.min(axis=0)
        mean = column.mean(axis=0)
        std = column.std(axis=0)
        stats_df = pd.DataFrame({"max_val": [max_val], "min_val": [min_val], "mean": [mean], "std": [std],
                                 "unique_count": [unique_count],
                                 "unique_ratio": [unique_ratio]})
        print(stats_df)
        column.hist(bins=100)
        plt.title(col_name)
        plt.show()
    elif col_types[col_name] == "date":
        data[col_name] = pd.to_datetime(data[col_name])
        column = data.loc[:, col_name]
        column.hist(bins=100)
        plt.title(col_name)
        plt.show()
    elif col_types[col_name] == "categorical":
        # Get unique values of the categorical variable
        val_counts = column.value_counts()
        print(val_counts)


def preprocess_payment_data(data, ignore_colums, col_types):
    # Step 1: Select valid columns
    valid_columns = [col_name for col_name in data.columns if col_name not in ignore_colums]
    data = data.loc[:, valid_columns]
    # Step 2: Analyze each column
    for col in valid_columns:
        if col == "id" or col == "label":
            continue
        analyze_column(data, col, col_types)
    # Step 3: Remove rows with missing "update_date" column
    data = data[~data["update_date"].isnull()]
    # Step 4: Impute missing values for "highest_balance" column; use median
    impute_missing_value(data, col_name="highest_balance", col_type=col_types["highest_balance"])
    # Step 5: Replace categorical variables with one-hot encoding
    # unencoded_data = data.copy()
    for col, col_type in col_types.items():
        if col_type == "categorical":
            data = pd.get_dummies(data, columns=[col], prefix=col)
    return data


def preprocess_customer_data(data, ignore_colums, col_types):
    # Step 1: Select valid columns
    valid_columns = [col_name for col_name in data.columns if col_name not in ignore_colums]
    data = data.loc[:, valid_columns]
    # Step 2: Analyze each column
    for col in valid_columns:
        if col == "id" or col == "label":
            continue
        analyze_column(data, col, col_types)
    # Step 3: Impute missing values if any of the feature columns contains one.
    for col in valid_columns:
        # id and label columns MUST not contain any missing value.
        if col == "id" or col == "label":
            assert data[col].isna().sum() == 0
        else:
            if data[col].isna().sum() > 0:
                impute_missing_value(data, col_name=col, col_type=col_types[col])
    # Step 4: Replace categorical variables with one-hot encoding
    for col, col_type in col_types.items():
        if col_type == "categorical":
            data = pd.get_dummies(data, columns=[col], prefix=col, drop_first=True)
    return data


def aggregate_time_data(data, binning_start_date, time_column, bin_count):
    # There is more than one line per user in the payment data, which belong to different update_date values.
    # We need a fixed size feature vector for classification algorithms. What we are going to do is to bin the
    # span of the dates in the "update_date" column and aggregate each data row which belong to a specific customer
    # in the same date bin. The histogram of the "update_date" column shows there is very small amount of data before
    # 2004 and the distribution is heavily skewed toward new years. So, we group each entry before 2004, and then
    # apply binning for years after 2004, with a given time period (for example two years). The mean and std. of each
    # numerical column for a time bin is calculated as new features. Note that at this point we don't have any categorical
    # values, all of them have been one-hot encoded.

    # Create date bins
    t0_date = binning_start_date
    max_date = data[time_column].max(axis=0) + pd.DateOffset(days=1)
    min_date = data[time_column].min(axis=0) - pd.DateOffset(days=1)
    d_range = pd.DatetimeIndex([min_date])
    d_range_new = pd.date_range(start=t0_date, end=max_date, periods=bin_count)
    d_range = d_range.append(d_range_new)
    data_parts = []
    parts_count = []
    data_aggregated = None
    for idx, t in enumerate(range(d_range.shape[0] - 1)):
        t0 = d_range[t]
        t1 = d_range[t + 1]
        data_subset = data.loc[(t0 <= data[time_column]) & (data[time_column] <= t1)]
        parts_count.append(data_subset.shape[0])
        # Aggregate all data in the bin according to customer ids
        data_subset_aggregated = data_subset.groupby("id", as_index=False).mean()
        # Rename columns according to the bins
        new_col_names = {col: "{0}_t{1}".format(col, idx) for col in data_subset_aggregated.columns if col != "id"}
        data_subset_aggregated.rename(columns=new_col_names, inplace=True)
        # Merge each bin; make it outer join to account for missing customer ids in each bin
        if data_aggregated is None:
            data_aggregated = data_subset_aggregated
        else:
            df_merged = pd.merge(data_aggregated, data_subset_aggregated, left_on='id', right_on='id', how='outer',
                                 suffixes=('', ''))
            data_aggregated = df_merged
    data_aggregated.fillna(0.0, inplace=True)
    # At this stage, we have fixed length payment data summarized in an single vector for every customer. Return the
    # data frame.
    return data_aggregated


payment_data = pd.read_csv("payment_data_ratio20.csv")
customer_data = pd.read_csv("customer_data_ratio20.csv")
print(payment_data.shape)
print(customer_data.shape)
count_missing_entries(payment_data)
count_missing_entries(customer_data)

# Get payment features
payment_data = preprocess_payment_data(payment_data, payment_ignore_columns, payment_column_types)
payment_features = aggregate_time_data(payment_data, t0_moment, "update_date", payment_date_bin_count)

# Get customer features
customer_features = preprocess_customer_data(customer_data, customer_ignore_columns, customer_column_types)

# Merge both features
complete_features = pd.merge(customer_features, payment_features, left_on='id', right_on='id', how='outer',
                             suffixes=('', ''))
# Be sure that there is no Nan entry
assert complete_features.isna().sum(axis=0).sum() == 0

# Extact the label data. Drop the "id" column.
y_pd = complete_features.loc[:, "label"]
X_pd = complete_features.loc[:, [col_name != "id" and col_name != "label" for col_name in complete_features.columns]]
y = y_pd.to_numpy(copy=True)
X = X_pd.to_numpy(copy=True)

# Classification Part

# We are going to use Random Decision Forests as the classifier. For the following reasons:
# 1- Resistant to overfitting; as the number of trees goes up, the test accuracy asymptotically converges.
# 2- Can handle unscaled data well due to the feature selection criterion.
# 3- Fast training (Can train trees in parallel, compared to the iterative Gradient Boosting Trees).
# 4- Can learn very nonlinear boundaries in the feature space.

# Pick a held-out test set, we are going to test our final on this.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Subsample
# X_train_0 = X_train[y_train == 0]
# X_train_1 = X_train[y_train == 1]
# subset_indices = np.random.choice(X_train_0.shape[0], X_train_1.shape[0], replace=False)
# X_subsampled_train_0 = X_train_0[subset_indices, :]
#
# X_train = np.concatenate([X_subsampled_train_0, X_train_1], axis=0)
# y_train = np.ones_like(y_train)[0: X_train.shape[0]]
# y_train[0: X_subsampled_train_0.shape[0]] = 0

# Logistic Regression
# pca = PCA()
# logistic = LogisticRegression()
# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
# # Hyperparameter grid
# exponential_distribution = expon(scale=100)
# all_regularizer_values = exponential_distribution.rvs(10).tolist()
# lesser_than_one = np.linspace(0.00001, 1.0, 11)
# all_regularizer_values.extend(lesser_than_one)
# all_regularizer_values.extend([10, 100, 1000, 10000])
# param_grid = [
#     #     # {
#     #     #     'pca__n_components': [5, 20, 30, 40, 50, 64, 128, 150, 200],
#     #     #     'svm__kernel': ['rbf'],
#     #     #     'svm__gamma': [5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
#     #     #     'svm__C': all_regularizer_values,
#     #     #     'svm__class_weight': [None, "balanced"]
#     #     # },
#     {
#         'pca__n_components': [5, 20, 30, 40, 50, 64, 128, 150, 200],
#         'logistic__solver': ['newton-cg', 'liblinear'],
#         'logistic__C': all_regularizer_values,
#         'logistic__class_weight': [None, "balanced"],
#         'logistic__fit_intercept': [True, False],
#         'logistic__max_iter': [1000],
#     }]

# SVM
# pca = PCA()
# svm = SVC()
# pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
# # Hyperparameter grid
# exponential_distribution = expon(scale=100)
# all_regularizer_values = exponential_distribution.rvs(10).tolist()
# lesser_than_one = np.linspace(0.00001, 1.0, 11)
# all_regularizer_values.extend(lesser_than_one)
# all_regularizer_values.extend([10, 100, 1000, 10000])
# # param_grid = [
# #     # {
# #     #     'pca__n_components': [5, 20, 30, 40, 50, 64, 128, 150, 200],
# #     #     'svm__kernel': ['rbf'],
# #     #     'svm__gamma': [5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
# #     #     'svm__C': all_regularizer_values,
# #     #     'svm__class_weight': [None, "balanced"]
# #     # },
# #     {
#         'pca__n_components': [5, 20, 30, 40, 50, 64, 128, 150, 200],
#         'svm__kernel': ['linear'],
#         'svm__C': all_regularizer_values,
#         'svm__class_weight': [None, "balanced"]
#     }]

# RDF
pca = PCA()
rdf = RandomForestClassifier()
pipe = Pipeline(steps=[('pca', pca), ('rdf', rdf)])

# Hyperparameter grid
param_grid = {
    'pca__n_components': [5, 20, 30, 40, 50, 64, 128, 150, 200],
    'rdf__n_estimators': [100],
    'rdf__max_depth': [5, 10, 15, 20, 25, 30],
    'rdf__bootstrap': [False, True],
    # 'rdf__class_weight': [{0: 1.0, 1: 1.0}, {0: 1.0, 1: 5.0}, {0: 1.0, 1: 10.0}, {0: 1.0, 1: 100.0}]
    'rdf__class_weight': [None, "balanced", "balanced_subsample"]
}

grid_search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=8, refit=True, verbose=5)
grid_search.fit(X=X_train, y=y_train)
best_model = grid_search.best_estimator_
print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
print(grid_search.best_params_)

# Training Results
print("Training Results")
mean_training_accuracy = best_model.score(X=X_train, y=y_train)
y_hat_train = best_model.predict(X=X_train)
print("Mean Training Accuracy={0}".format(mean_training_accuracy))
report_training = classification_report(y_true=y_train, y_pred=y_hat_train, target_names=["0", "1"])
print(report_training)

# Test Results
print("Test Results")
mean_test_accuracy = best_model.score(X=X_test, y=y_test)
y_hat_test = best_model.predict(X=X_test)
print("Mean Test Accuracy={0}".format(mean_test_accuracy))
report_test = classification_report(y_true=y_test, y_pred=y_hat_test, target_names=["0", "1"])
print(report_test)
