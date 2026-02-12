# %% [markdown]
# Step 1: Identify a question that the dataset can answer
#
# College Completion Dataset:
# Assuming Flagship Universities are the best, can we use awards per 100
# graduates to determine what schools are flagship?
#
# Job Placement Dataset:
# Is there a correlation between performance on standardized tests and
# degree performance percentile and whether you have gotten a job?


# %% [markdown]
# Step 2: Independent Business Metric and Data Cleaning
#
# College Completion Dataset:
# Independent Business Metric:
# `high_award` = Whether or not the institution has a high award status
# (1 = high award status, 0 = low award status). `high_award` is
# determined by whether the institution's awards per state value is above
# the Q3 value of awards per 100 graduates across all institutions.
#
# Job Placement Dataset:
# Independent Business Metric:
# `status` = Whether or not the candidate is employed or not (1 = employed,
# 0 = unemployed).


# %%
"""Data preparation and cleaning steps for the two datasets."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import pandas as pd


# %%
# Data Preparation and Cleaning for College Completion Dataset
COLLEGE = pd.read_csv("cc_institution_details.csv")
COLLEGE.info()


# %%
# Turning necessary variables into categorical variables
cat_cols = ["state", "level", "control"]

for col in cat_cols:
    COLLEGE[col] = COLLEGE[col].astype("category")

COLLEGE.info()


# %%
# Turning Boolean columns into 0s and 1s
bool_cols = ["flagship"]

for col in bool_cols:
    if col in COLLEGE.columns:
        COLLEGE[col] = (COLLEGE[col] == "X").astype(int)


# %%
# One-hot encoding categorical variables
one_hot_columns = ["level", "control"]

cols_to_encode = [c for c in one_hot_columns if c in COLLEGE.columns]

if cols_to_encode:
    COLLEGE = pd.get_dummies(COLLEGE, columns=cols_to_encode, dtype=int)


# %%
high_threshold = COLLEGE["awards_per_state_value"].quantile(0.75)

COLLEGE["high_award"] = (
    COLLEGE["awards_per_state_value"] > high_threshold
).astype(int)


# %%
# Standardizing and scaling numerical columns
columns = [
    "aid_percentile",
    "aid_value",
    "endow_value",
    "endow_percentile",
    "grad_100_value",
    "grad_100_percentile",
    "grad_150_value",
    "grad_150_percentile",
    "exp_award_percentile",
    "exp_award_value",
    "exp_award_state_value",
    "exp_award_natl_value",
    "ft_pct",
    "fte_value",
    "fte_percentile",
    "med_sat_value",
    "med_sat_percentile",
    "student_count",
    "awards_per_value",
    "awards_per_state_value",
    "awards_per_natl_value"
]

scaler = MinMaxScaler()

for col in columns:
    if col in columns:
        COLLEGE[[col]] = scaler.fit_transform(COLLEGE[[col]])


# %%
# Dropping unnecessary columns
cols_by_name = [
    "site",
    "long_x",
    "lat_y",
    "med_sat_percentile",
    "med_sat_value",
    "endow_value",
    "basic",
    "endow_percentile",
    "unitid",
    "city",
    "grad_100_value",
    "grad_100_percentile",
    "grad_150_value",
    "grad_150_percentile",
    "hbcu",
    "chronname",
    "state",
    "awards_per_value",
    "awards_per_state_value",
    "awards_per_natl_value",
]

cols_by_index = COLLEGE.columns[34:63]

cols_to_drop = [c for c in cols_by_name + list(cols_by_index) if c in COLLEGE.columns]

if cols_to_drop:
    COLLEGE_dt = COLLEGE.drop(columns=cols_to_drop)


# %%
# Calculating prevalence
total_rows = len(COLLEGE_dt)
prevalence = COLLEGE_dt["high_award"].sum() / total_rows
print(f"Prevalence of High Award Status: {prevalence:.2f}")


# %%
# Splitting the dataset into Train and Test sets
Train, Test = train_test_split(COLLEGE_dt, train_size=2700, stratify=COLLEGE_dt.high_award)
print(Train.shape)
print(Test.shape)


# %%
# Create tuning set
Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test.high_award)


# %%
# Calculating prevalence for Train, Tune, and Test sets
train_prev = Train.high_award.value_counts() / len(Train)
tune_prev = Tune.high_award.value_counts() / len(Tune)
test_prev = Test.high_award.value_counts() / len(Test)

print(
    f"""Training Prevalence: {train_prev[1]:.2f}
Tuning Prevalence: {tune_prev[1]:.2f}
Test Prevalence: {test_prev[1]:.2f}
"""
)


########################################################################################
########################################################################################


# %%
# Data Preparation and Cleaning for Job Placement Dataset
JOB = pd.read_csv("Placement_Data_Full_Class.csv")
JOB.info()


# %%
# Turning necessary variables into categorical variables
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "specialisation"]

for col in cat_cols:
    JOB[col] = JOB[col].astype("category")

JOB.info()


# %%
# One-hot encoding categorical variables
one_hot_columns = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "specialisation",
    "status",
    "workex",
]

cols_to_encode = [c for c in one_hot_columns if c in JOB.columns]

if cols_to_encode:
    JOB = pd.get_dummies(JOB, columns=cols_to_encode, dtype=int)


# %%
# Standardizing and scaling numerical columns
columns = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
scaler = MinMaxScaler()

for col in columns:
    if col in JOB.columns:
        JOB[[col]] = scaler.fit_transform(JOB[[col]])


# %%
# Dropping unnecessary columns
cols_by_name = ["salary"]

cols_to_drop = [c for c in cols_by_name if c in JOB.columns]

if cols_to_drop:
    JOB_dt = JOB.drop(columns=cols_to_drop)


# %%
# Calculating prevalence
total_rows = len(JOB_dt)
prevalence = JOB_dt["status_Placed"].sum() / total_rows
print(f"Prevalence of Employed Status: {prevalence:.2f}")


# %%
# Splitting the dataset into Train and Test sets
Train, Test = train_test_split(JOB_dt, train_size=145, stratify=JOB_dt.status_Placed)
print(Train.shape)
print(Test.shape)


# %%
# Create tuning set
Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test.status_Placed)


# %%
# Calculating prevalence for Train, Tune, and Test sets
train_prev = Train.status_Placed.value_counts() / len(Train)
tune_prev = Tune.status_Placed.value_counts() / len(Tune)
test_prev = Test.status_Placed.value_counts() / len(Test)

print(
    f"""Training Prevalence: {train_prev[1]:.2f}
Tuning Prevalence: {tune_prev[1]:.2f}
Test Prevalence: {test_prev[1]:.2f}
"""
)


# %% [markdown]
## Step three:
## What do your instincts tell you about the data. Can it address your
## problem, what areas/items are you worried about?
##
## College Completion Dataset:
# The dataset seems to have a good amount of data available to answer my
# question. The only concern I have is that there are only 50 flagship
# universities because each state only has one flagship university which
# might not be enough data to make a strong conclusion; however I will
# proceed with the analysis.
##
## Job Placement Dataset:
# This dataset seems much more organized and straightforward. I think that
# it will be able to address my problem well. My only concern is that I
# did not interpret the columns correctly and that might skew my answer.
# It is a little odd that there is not a data dictionary available. I did
# my best to interpret the columns correctly.
