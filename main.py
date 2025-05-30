# general
import io

# data
import numpy as np
import pandas as pd

# machine learning
import keras

# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

# @title -Read dataset
chicago_taxi_dataset = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
)
# print(f"Data {chicago_taxi_dataset}")


# Updates dataframe to use specific columns
training_df = chicago_taxi_dataset[
    ["TRIP_MILES", "TRIP_SECONDS", "FARE", "COMPANY", "PAYMENT_TYPE", "TIP_RATE"]
]

print("Read dataset completed sucessfully")
print("Total number of rows:{0}\n\n".format(len(training_df.index)))
# print(training_df.head(200))
# print(training_df.describe(include="all"))
#

# What is the maximum fare?
max_fare = training_df["FARE"].max()
print("What is the maximum fare? \t\t\t\tAnswer: ${fare:.2f}".format(fare=max_fare))

# What is the mean distance across all trips?
mean_distance = training_df["TRIP_MILES"].mean()
print(
    "What is the mean distance across all trips? \t\tAnswer: {mean:.4f} miles".format(
        mean=mean_distance
    )
)

# How many cab companies are in the dataset?
num_unique_companies = training_df["COMPANY"].nunique()
print(
    "How many cab companies are in the dataset? \t\tAnswer: {number}".format(
        number=num_unique_companies
    )
)

# What is the most frequent payment type?
most_freq_payment_type = training_df["PAYMENT_TYPE"].value_counts().idxmax()
print(
    "What is the most frequent payment type? \t\tAnswer: {type}".format(
        type=most_freq_payment_type
    )
)

# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print(
    "Are any features missing data? \t\t\t\tAnswer:",
    "No" if missing_values == 0 else "Yes",
)
