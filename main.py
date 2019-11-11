import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import os

# Payment table
payment_column_names = ["id", "OVD_t1", "OVD_t2", "OVD_t3", "OVD_sum", "pay_normal", "prod_code", "prod_limit",
                        "update_date",
                        "new_balance", "highest_balance", "report_date"]
payment_ignore_columns = {"prod_limit", "report_date"}
payment_columns_types = {"numerical": {"OVD_t1", "OVD_t2", "OVD_t3", "OVD_sum", "pay_normal",
                                       "new_balance", "highest_balance", "prod_limit"},
                         "categorical": {"prod_code"},
                         "date": {"update_date", "report_date"}}
date_bin_count = 8


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


def analyze_column(data, col_name, col_types):
    column = data.loc[:, col_name]

    print("Column:{0}".format(col_name))
    # Step 0: Get column type
    print("Type:{0}".format(column.dtypes))
    if col_name in col_types["numerical"]:
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
        # plt.show()
    elif col_name in col_types["date"]:
        data[col_name] = pd.to_datetime(data[col_name])
        column = data.loc[:, col_name]
        column.hist(bins=100)
        plt.title(col_name)
        # plt.show()
    elif col_name in col_types["categorical"]:
        # Get unique values of the categorical variable
        val_counts = column.value_counts()
        print(val_counts)


def preprocess_payment_data(data):
    # Step 1: Select valid columns
    valid_columns = [col_name for col_name in data.columns if col_name not in payment_ignore_columns]
    data = data.loc[:, valid_columns]
    # Step 2: Analyze each column
    for col in valid_columns:
        if col == "id":
            continue
        analyze_column(data, col, payment_columns_types)
    # Step 3: Remove rows with missing "update_date" column
    data = data[~data["update_date"].isnull()]
    # Step 4: Impute missing values for "highest_balance" column; use median
    highest_balance_values = data["highest_balance"]
    imputer = SimpleImputer(strategy="median", copy=True)
    values = highest_balance_values.to_numpy(copy=True)
    values = np.expand_dims(values, axis=1)
    imputer.fit(values)
    imputed_values = imputer.transform(values)
    imputed_values = np.reshape(imputed_values, newshape=(imputed_values.shape[0], ))
    data["highest_balance"] = imputed_values
    # Step 5: Replace categorical variables with one-hot encoding
    # unencoded_data = data.copy()
    for categorical_col in payment_columns_types["categorical"]:
        data = pd.get_dummies(data, columns=[categorical_col], prefix=categorical_col)
    # Step 6: There is more than one line per user in the payment data, which belong to different update_date values.
    # We need a fixed size feature vector for classification algorithms. What we are going to do is to bin the
    # span of the dates in the "update_date" column and aggregate each data row which belong to a specific customer
    # in the same date bin. The histogram of the "update_date" column shows there is very small amount of data before
    # 2004 and the distribution is heavily skewed toward new years. So, we group each entry before 2004, and then
    # apply binning for years after 2004, with a given time period (for example two years). The mean and std. of each
    # numerical column for a time bin is calculated as new features. Note that at this point we don't have any categorical
    # values, all of them have been one-hot encoded.

    # Create date bins
    t0_date = pd.Timestamp(year=2004, month=1, day=1, hour=0)
    max_date = data["update_date"].max(axis=0) + pd.DateOffset(days=1)
    min_date = data["update_date"].min(axis=0) - pd.DateOffset(days=1)
    d_range = pd.DatetimeIndex([min_date])
    d_range_new = pd.date_range(start=t0_date, end=max_date, periods=date_bin_count)
    d_range = d_range.append(d_range_new)
    data_parts = []
    parts_count = []
    data_aggregated = None
    for idx, t in enumerate(range(d_range.shape[0] - 1)):
        t0 = d_range[t]
        t1 = d_range[t + 1]
        data_subset = data.loc[(t0 <= data["update_date"]) & (data["update_date"] <= t1)]
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

    print("X")

    # curr_date = date_bins[0]
    # while True:
    #     next_date = curr_date + pd.DateOffset(months=date_bin_length_in_months)
    #     date_bins.append(next_date)
    #     curr_date = next_date
    #     if curr_date >= max_date:
    #         break

    grouped = data.groupby("id").count()
    print("X")





payment_data = pd.read_csv("payment_data_ratio20.csv")
customer_data = pd.read_csv("customer_data_ratio20.csv")
print(payment_data.shape)
print(customer_data.shape)
count_missing_entries(payment_data)
count_missing_entries(customer_data)
print(customer_data["id"].nunique())
preprocess_payment_data(payment_data)

# payment_data["OVD_t1"].hist(bins=100)

# statistics_dict = {"values_1": np.array([12.0, 13.0, 19.0, 123.0, 54.0])}
# df = pd.DataFrame.from_dict(statistics_dict)
# # df['values_1'] = df['values_1'].astype(float)
# df.hist(bins=5)

# df2 = pd.DataFrame({'length': [1.5, 0.5, 1.2, 0.9, 3], 'width': [0.7, 0.2, 0.15, 0.2, 1.1]},
#                    index=['pig', 'rabbit', 'duck', 'chicken', 'horse'])
# df2.hist(bins=3)
plt.show()
print("X")
# preprocess_payment_data(payment_data)
