import os
import json

import numpy as np
import pandas as pd

from xgboost import XGBRanker


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "task_machine_candidates_v2.csv"

MODEL_DIR = "models"
MODEL_FILE = "models/xgboost_ranker_v3.json"
METRICS_FILE = "ml_ranker_v3_metrics.json"
PREDICTIONS_FILE = "ml_ranker_v3_test_predictions.csv"

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("===================================")
print("XGBoost Learning-to-Rank v3")
print("===================================")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Tasks:", df["instance_name"].nunique())


# ============================================================
# FEATURES
#
# Only columns that actually exist in
# task_machine_candidates_v2.csv.
#
# Deliberately excluded:
#   candidate_rank
#   machine_id
#   instance_name
#   task_name
#   job_name
#   estimated_cpu_pressure
#   estimated_mem_pressure
# ============================================================

FEATURES = [
    "task_type",

    "cpu_avg",
    "cpu_max",

    "mem_avg",
    "mem_max",

    "task_duration",

    "cpu_demand_ratio",
    "mem_demand_ratio",

    "log_cpu_demand",
    "log_mem_demand",

    "cpu_num",
    "mem_size",

    "cpu_util",
    "mem_util",

    "cpu_pressure",
    "mem_pressure",

    "bottleneck_pressure"
]


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features:\n"
        + "\n".join(missing_features)
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=FEATURES
).reset_index(drop=True)


print(
    "Rows after cleaning:",
    len(df)
)


# ============================================================
# CREATE RANKING SCORE
#
# IMPORTANT:
#
# This score is used ONLY to generate training labels.
#
# It is NOT included as a model feature.
#
# Lower score = better candidate.
# ============================================================

df["ranking_score"] = np.maximum(
    df["bottleneck_pressure"],
    df["estimated_cpu_pressure"]
)


# ============================================================
# CREATE RELEVANCE LABEL
#
# Best candidate:
#     relevance = 4
#
# Worst candidate:
#     relevance = 0
#
# XGBRanker tries to put higher-relevance candidates first.
# ============================================================

df["relevance"] = (
    df.groupby("instance_name")["ranking_score"]
    .rank(
        ascending=True,
        method="first"
    )
)

df["relevance"] = (
    5 - df["relevance"]
)

df["relevance"] = (
    df["relevance"]
    .round()
    .astype(int)
    .clip(0, 4)
)


# ============================================================
# VERIFY LABELS
# ============================================================

print(
    "\n==================================="
)

print(
    "Ranking Label Distribution"
)

print(
    "==================================="
)

print(
    df["relevance"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# VERIFY FIVE CANDIDATES PER TASK
# ============================================================

rows_per_task = (
    df.groupby("instance_name")
    .size()
)

print(
    "\nCandidates per task:"
)

print(
    rows_per_task
    .value_counts()
    .sort_index()
    .to_string()
)

assert np.all(
    rows_per_task.values == 5
)


# ============================================================
# TASK-LEVEL SPLIT
# ============================================================

unique_tasks = (
    df["instance_name"]
    .drop_duplicates()
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED
    )
    .tolist()
)

num_tasks = len(unique_tasks)

train_end = int(
    0.70 * num_tasks
)

validation_end = int(
    0.85 * num_tasks
)


train_tasks = set(
    unique_tasks[:train_end]
)

validation_tasks = set(
    unique_tasks[
        train_end:validation_end
    ]
)

test_tasks = set(
    unique_tasks[
        validation_end:
    ]
)


# ============================================================
# CREATE SPLITS
# ============================================================

train_df = df[
    df["instance_name"].isin(train_tasks)
].copy()

validation_df = df[
    df["instance_name"].isin(validation_tasks)
].copy()

test_df = df[
    df["instance_name"].isin(test_tasks)
].copy()


# ============================================================
# SORT BY TASK
# ============================================================

train_df = train_df.sort_values(
    "instance_name"
).reset_index(drop=True)

validation_df = validation_df.sort_values(
    "instance_name"
).reset_index(drop=True)

test_df = test_df.sort_values(
    "instance_name"
).reset_index(drop=True)


# ============================================================
# VERIFY NO TASK LEAKAGE
# ============================================================

assert not (
    train_tasks & validation_tasks
)

assert not (
    train_tasks & test_tasks
)

assert not (
    validation_tasks & test_tasks
)


# ============================================================
# GROUP SIZES
# ============================================================

train_groups = (
    train_df
    .groupby("instance_name")
    .size()
    .to_numpy()
)

validation_groups = (
    validation_df
    .groupby("instance_name")
    .size()
    .to_numpy()
)

test_groups = (
    test_df
    .groupby("instance_name")
    .size()
    .to_numpy()
)


assert np.all(
    train_groups == 5
)

assert np.all(
    validation_groups == 5
)

assert np.all(
    test_groups == 5
)


# ============================================================
# PRINT SPLIT
# ============================================================

print(
    "\n==================================="
)

print(
    "Task-Level Ranking Split"
)

print(
    "==================================="
)

print(
    "Training tasks:",
    len(train_tasks)
)

print(
    "Validation tasks:",
    len(validation_tasks)
)

print(
    "Test tasks:",
    len(test_tasks)
)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(validation_df)
)

print(
    "Test rows:",
    len(test_df)
)


# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["relevance"]

X_validation = validation_df[FEATURES]
y_validation = validation_df["relevance"]

X_test = test_df[FEATURES]
y_test = test_df["relevance"]


# ============================================================
# CREATE RANKER
# ============================================================

print(
    "\n==================================="
)

print(
    "Creating XGBoost Ranker"
)

print(
    "==================================="
)

model = XGBRanker(

    objective="rank:pairwise",

    n_estimators=600,

    learning_rate=0.05,

    max_depth=6,

    min_child_weight=5,

    subsample=0.85,

    colsample_bytree=0.85,

    reg_alpha=0.1,

    reg_lambda=1.5,

    random_state=RANDOM_SEED,

    n_jobs=-1,

    tree_method="hist"
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nStarting ranking training..."
)

model.fit(

    X_train,

    y_train,

    group=train_groups,

    eval_set=[
        (
            X_validation,
            y_validation
        )
    ],

    eval_group=[
        validation_groups
    ],

    verbose=False
)

print(
    "Ranking training complete."
)


# ============================================================
# PREDICTIONS
# ============================================================

train_df[
    "ranking_prediction"
] = model.predict(X_train)

validation_df[
    "ranking_prediction"
] = model.predict(
    X_validation
)

test_df[
    "ranking_prediction"
] = model.predict(
    X_test
)


# ============================================================
# TOP-1 ACCURACY
# ============================================================

def top1_accuracy(data):

    predicted = (
        data
        .sort_values(
            [
                "instance_name",
                "ranking_prediction"
            ],
            ascending=[
                True,
                False
            ]
        )
        .groupby(
            "instance_name"
        )
        .first()
    )

    actual = (
        data
        .sort_values(
            [
                "instance_name",
                "ranking_score"
            ],
            ascending=[
                True,
                True
            ]
        )
        .groupby(
            "instance_name"
        )
        .first()
    )

    matches = (
        predicted["machine_id"].values
        ==
        actual["machine_id"].values
    )

    return (
        matches.sum()
        /
        len(matches)
    )


# ============================================================
# CALCULATE TOP-1
# ============================================================

train_top1 = top1_accuracy(
    train_df
)

validation_top1 = top1_accuracy(
    validation_df
)

test_top1 = top1_accuracy(
    test_df
)


# ============================================================
# PRINT PERFORMANCE
# ============================================================

print(
    "\n==================================="
)

print(
    "Ranking Performance"
)

print(
    "==================================="
)

print(
    "Training Top-1:",
    round(
        train_top1 * 100,
        2
    ),
    "%"
)

print(
    "Validation Top-1:",
    round(
        validation_top1 * 100,
        2
    ),
    "%"
)

print(
    "Test Top-1:",
    round(
        test_top1 * 100,
        2
    ),
    "%"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance":
            model.feature_importances_
    }
)

importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    "\n==================================="
)

print(
    "Top Feature Importance"
)

print(
    "==================================="
)

print(
    importance
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save_model(
    MODEL_FILE
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_df[
    [
        "instance_name",
        "event_time",
        "machine_id",
        "candidate_rank",
        "ranking_score",
        "relevance",
        "ranking_prediction"
    ]
].to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "XGBRanker",

    "objective": "rank:pairwise",

    "features": FEATURES,

    "random_seed": RANDOM_SEED,

    "train_tasks": len(train_tasks),

    "validation_tasks": len(validation_tasks),

    "test_tasks": len(test_tasks),

    "train_top1_accuracy": float(
        train_top1
    ),

    "validation_top1_accuracy": float(
        validation_top1
    ),

    "test_top1_accuracy": float(
        test_top1
    )
}


with open(
    METRICS_FILE,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n==================================="
)

print(
    "ML Ranker v3 Complete"
)

print(
    "==================================="
)

print(
    "Model:",
    MODEL_FILE
)

print(
    "Test predictions:",
    PREDICTIONS_FILE
)

print(
    "Metrics:",
    METRICS_FILE
)