
"""
Graduation Lab: Week 6


Instructions:

Let's build a kNN model using the college completion data.
The data is messy and you have a degrees of freedom problem, as in, we have too many features.

You've done most of the hard work already, so you should be ready to move forward with building your model.

1. Use the question/target variable you submitted and
build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary).

2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning
if needed changed the target variable.

3. Create a dataframe that includes the test target values, test predicted values,
and test probabilities of the positive class.

4. No code question: If you adjusted the k hyperparameter what do you think would
happen to the threshold function? Would the confusion look the same at the same threshold
levels or not? Why or why not?

5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what
concerns or positive elements do you have about the model as it relates to your question?

6. Create two functions: One that cleans the data & splits into training|test and one that
allows you to train and test the model with different k and threshold values, then use them to
optimize your model (test your model with several k and threshold combinations). Try not to use variable names
in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
function just run them separately.)

7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not?

8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
step 7.

"""

# %%
import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %% [markdown]
# Question 1
# **Question**
# Is there a correlation between grad_100 and other college graduation and scholarship
# metrics and the whether a college can be considered "high_award"?

###
# **IBM** `high_award` = Whether or not the institution has a high award status (1 = high award status, 0 = low award status). `high_award` is determined by whether the institution's awards per state value is above the Q3 value of awards per 100 graduates across all institutions.

# %% [markdown]
# Question 2

# %%
# Data Preparation and Cleaning for College Completion Dataset
COLLEGE = pd.read_csv("cc_institution_details.csv")
COLLEGE.info()

# %%
# Create the "high_threshold" variable/column
high_threshold = COLLEGE["awards_per_state_value"].quantile(0.75)

COLLEGE["high_award"] = (
    COLLEGE["awards_per_state_value"] > high_threshold
).astype(int)

# %%
# Turning "chronname" into type object
COLLEGE["chronname"] = COLLEGE["chronname"].astype(object)

# Turning necessary variables into categorical variables
cat_cols = ["state", "level", "control", "high_award"]

for col in cat_cols:
    COLLEGE[col] = COLLEGE[col].astype("category")

COLLEGE.info()

# %%
# One-hot encoding categorical variables
one_hot_columns = ["level", "control", "high_award"]

cols_to_encode = [c for c in one_hot_columns if c in COLLEGE.columns]

if cols_to_encode:
    COLLEGE = pd.get_dummies(COLLEGE, columns=cols_to_encode)

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
    "hbcu",
    "flagship",
    "state",
    "awards_per_value",
    "awards_per_state_value",
    "awards_per_natl_value",
    "index",
    "chronname"
]

cols_by_index = COLLEGE.columns[34:63]

cols_to_drop = [c for c in cols_by_name +
                list(cols_by_index) if c in COLLEGE.columns]

COLLEGE_dt = COLLEGE.drop(columns=cols_to_drop)

COLLEGE_dt.info()

# %%
sns.displot(
    data=COLLEGE_dt.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
plt.show()

# %%
# Drop all rows with missing data
COLLEGE_dt.dropna(axis=0, how='any', inplace=True)

sns.displot(
    data=COLLEGE_dt.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
plt.show()
# %%
Train, Test = train_test_split(
    COLLEGE_dt, train_size=0.4, stratify=COLLEGE_dt["high_award_1"])
Test, Val = train_test_split(
    Test, test_size=0.5, stratify=Test["high_award_1"])

# %%
# This is the KNN model with k=3
# kNN is a random algorithm, so we use `random.seed(x)` to make results
# repeatable
random.seed(1984)

X_train = Train.drop(['high_award_1'], axis=1).values
y_train = Train['high_award_1'].values

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# %%
# now, we check the model's accuracy on the test data:

X_test = Test.drop(['high_award_1'], axis=1).values
y_test = Test['high_award_1'].values

neigh.score(X_test, y_test)

# %%
# now, we test the accuracy on our validation data.

X_val = Val.drop(['high_award_1'], axis=1).values
y_val = Val['high_award_1'].values

neigh.score(X_val, y_val)

# %%
ConfusionMatrixDisplay.from_estimator(
    neigh,
    X_val,
    y_val,
    cmap='Purples'
)

plt.show()

# %%
y_val_pred = neigh.predict(X_val)
print(classification_report(y_val_pred, y_val))
# %%


# %% [markdown]
# Question 3
# Create a dataframe that includes the test target values, test predicted values,
# and test probabilities of the positive class.
# %%
test_probs = neigh.predict_proba(X_test)
test_preds = neigh.predict(X_test)

# %%
test_probabilities = pd.DataFrame(
    test_probs,
    columns=[
        'not_high_award_prob',
        'high_award_prob'])

final_model = pd.DataFrame(
    {
        "actual_class": y_test.tolist(),
        "pred_class": test_preds.tolist(),
        "pred_prob": [
            test_probabilities["high_award_prob"][i]
            if test_preds[i] == 1
            else test_probabilities["not_high_award_prob"][i]
            for i in range(len(test_preds))
        ],
    }
)

final_model

# %% [markdown]
# # Question 4
#  No code question: If you adjusted the k hyperparameter what do you think would
# happen to the threshold function? Would the confusion look the same at the same threshold
# levels or not? Why or why not?

# ## Answer:
# Adjusting the k hyperparameter would impact the treshold function by the number of possible probability values that can be generated.
# This would smooth out the threshold function as k increases because as more neighbors are considered, the lable becomes closer to the average of the overall dataset.
# Since there are more possible probability values, the confusion matrix at the same threshold levels would likely differ as k changes. At K = 3 there are only 4
# possible probability values (0, 1/3, 2/3, 1) but at K = 5 there are 6 possible probability values (0, 1/5, 2/5, 3/5, 4/5, 1). Meaning that at the same threshold of 0.33, the k = 3
# might classify them as 0.33 but the k = 5 might classify them as 0.4 and
# 0.2, which would change the confusion matrix at the same threshold
# level.


# %% [markdown]
# Question 5
# 5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what
# concerns or positive elements do you have about the model as it relates
# to your question?

# %%
ConfusionMatrixDisplay.from_estimator(
    neigh,
    X_val,
    y_val,
    cmap='Purples'
)

plt.show()


# %% [markdown]
# Putting the confusion matrix into context, there are a total of 1040 colleges in this dataset
# There are 810 colleges that are not high award and 230 colleges that are high award.
# Of the 810 colleges that are not high award, the model correctly predicted 793 colleges as being
# not high award and incorrectly predicted 17 colleges as being high award. This means that
# the model has a false positive rate of 2.1% (17/810). Of the 230 colleges that are considered to be
# high award, the model correctly predicted 180 colleges as being high award and incorrectly predicted 50
# colleges as being not high award when they in fact were high award. This means that the model has a
# false negative rate of 21.7% (50/230).
#
# Now to put this into the context of the question, which is to determine if you can correctlt predict
# if a college is high award based on other featurs of the college, like graduation and scholarship metrics.
# If the model predicts that a college is high award, there is a 91.4% (180/197) chance that the college is actually high award.
# If the model predicts that a college is not high award, there is a 94.7% (793/843) chance that the college is actually not high award.
# All of this comes to say, that if you are using this model to predict whether it is likely that you will receive in or above the
# 75th percentile of awards per 100 graduates, then this model will be correct 91.4% of the time when predicting that the college will he "high_award".
# Similarly, this model will be correct 94.7% of the time when predicting that the college will not be "high_award". A concern about this
# model is that it is only right 90-95% of the time, which means that
# there is a 5-10% chance that the model will be wrong when predicting
# whether a college is "high_award" or not.

# %% [markdown]
# Question 6:
# Create two functions: One that cleans the data & splits into training|test and one that
# allows you to train and test the model with different k and threshold values, then use them to
# optimize your model (test your model with several k and threshold combinations). Try not to use variable names
# in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
# function just run them separately.)

# %%


def prepare_and_split_college(
    df,
    target_col=None,
    cat_cols=None,
    one_hot_cols=None,
    scale_cols=None,
    drop_cols_by_name=None,
    drop_col_index_range=(0, 0),
    train_size=0.4,
    val_size_of_remainder=0.5,
    random_state=1984,
    drop_na=True,
):
    """
    Clean the dataset and split into Train / Test / Val.

    Cleaning:
      1) Cast cat_cols to category
      2) One-hot encode one_hot_cols
      3) MinMax scale scale_cols
      4) Drop columns by name + by index slice
      5) Drop rows with any missing values (optional)

    Splitting:
      - Train gets train_size (stratified)
      - Remainder splits into Test and Val (stratified)
    """
    df = df.copy()

    cat_cols = cat_cols or []
    one_hot_cols = one_hot_cols or []
    scale_cols = scale_cols or []
    drop_cols_by_name = drop_cols_by_name or []

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    cols_to_encode = [c for c in one_hot_cols if c in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode)

    scaler = MinMaxScaler()
    for col in scale_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[[col]] = scaler.fit_transform(df[[col]])

    start, end = drop_col_index_range
    drop_by_index = list(df.columns[start:end]) if end > start else []
    cols_to_drop = [
        c for c in (
            drop_cols_by_name +
            drop_by_index) if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if drop_na:
        df = df.dropna(axis=0, how="any")

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found after cleaning.")

    train_df, remainder = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    test_df, val_df = train_test_split(
        remainder,
        test_size=val_size_of_remainder,
        stratify=remainder[target_col],
        random_state=random_state,
    )

    return train_df, test_df, val_df


def knn_grid_search(
    train_df,
    val_df,
    target_col=None,
    k_values=None,
    thresholds=(),
):
    """
    Iterate through all combinations of k and threshold values.
    Evaluate accuracy on the validation set.

    Returns a DataFrame sorted by best accuracy first.
    """
    x_train = train_df.drop(columns=[target_col]).to_numpy()
    y_train = train_df[target_col].to_numpy()

    x_val = val_df.drop(columns=[target_col]).to_numpy()
    y_val = val_df[target_col].to_numpy()

    results = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=int(k))
        model.fit(x_train, y_train)

        probs = model.predict_proba(x_val)[:, 1]

        for th in thresholds:
            preds = (probs > float(th)).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
            acc = (tp + tn) / (tp + tn + fp + fn)

            results.append(
                {
                    "k": int(k),
                    "threshold": float(th),
                    "accuracy": acc,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )

    return (
        pd.DataFrame(results)
        .sort_values(by="accuracy", ascending=False)
        .reset_index(drop=True)
    )


# %%
# These cell must be run before the functions can be used.
# The data must be loaded in, and the target variable must be created.
COLLEGE = pd.read_csv("cc_institution_details.csv")
# Create the "high_threshold" variable/column
high_threshold = COLLEGE["awards_per_state_value"].quantile(0.75)

COLLEGE["high_award"] = (
    COLLEGE["awards_per_state_value"] > high_threshold
).astype(int)

# %%
train_df, test_df, val_df = prepare_and_split_college(
    df=COLLEGE,
    target_col="high_award_1",
    cat_cols=["state", "level", "control", "high_award"],
    one_hot_cols=["level", "control", "high_award"],
    scale_cols=[
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
    ],
    drop_cols_by_name=[
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
        "hbcu",
        "flagship",
        "state",
        "awards_per_value",
        "awards_per_state_value",
        "awards_per_natl_value",
        "index",
        "chronname",
        "similar",
        "counted_pct",
        "nicknames"
    ]
)

results_df = knn_grid_search(
    train_df=train_df,
    val_df=val_df,
    target_col="high_award_1",
    k_values=range(1, 22, 2),
    thresholds=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                0.7, 0.75, 0.8, 0.85, 0.9, 0.95
                )
)


# %%
results_df.head(10)
results_df.sort_values(by=['accuracy'], ascending=False)

# %% [markdown]
# The best combination is k = 21 and threshold = 0.25, which gives an accuracy of 0.914.
# %% [markdown]
# 7. How well does the model perform? Did the interaction
# of the adjusted thresholds and k values help the model? Why or why not?
#
# The model performs well, with the best combination of k and threshold giving
# an accuracy of 0.914. The interaction of the adjusted thresholds and k values didn't really help the model
# because the overall accruacy didn't change much from our original model with k = 3 and threshold = 0.5.
# However, I think most of that was just luck. In a normal situation, this interaction would be necessary
# as it allows us to find the optimal combination that maximized accuracy on the validation set.
# By testing different k values, we were able to find the right balance between bias and variance,
# while adjusting the threshold allowed us to optimize the classification decision boundary for our
# specific dataset and target variable.
# %% [Markdown]
# 8. Choose another variable as the target in the dataset and
# create another kNN model using the two functions you created in
# step 7.

# %%
COLLEGE = pd.read_csv("cc_institution_details.csv")
high_threshold = COLLEGE["grad_100_value"].quantile(0.75)
COLLEGE["high_grad_100"] = (
    COLLEGE["grad_100_value"] > high_threshold
).astype(int)

# %%

train_df, test_df, val_df = prepare_and_split_college(
    df=COLLEGE,
    target_col="high_grad_100_1",
    cat_cols=["state", "level", "control", "high_grad_100"],
    one_hot_cols=["level", "control", "high_grad_100"],
    scale_cols=[
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
    ],
    drop_cols_by_name=[
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
        "hbcu",
        "state",
        "index",
        "chronname",
        "similar",
        "counted_pct",
        "nicknames",
        "flagship"
    ]
)

results_df = knn_grid_search(
    train_df=train_df,
    val_df=val_df,
    target_col="high_grad_100_1",
    k_values=range(1, 22, 2),
    thresholds=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                0.7, 0.75, 0.8, 0.85, 0.9, 0.95
                )
)


# %%
results_df.sort_values(by=['accuracy'], ascending=False)


# %% [markdown]
# The best combination is k = 5 and threshold = 0.75, which gives an accuracy of 0.913. The question is switched to predict
# which schools are in the top 25% of graduation rates within 100% of normal time. This will help people decide which schools
# have been able to attract students that are more likely to finish school
# on time.
