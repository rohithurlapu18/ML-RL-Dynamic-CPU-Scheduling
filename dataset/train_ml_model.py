import os
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "ml_assignment_dataset.csv"

MODEL_DIR = "models"

MODEL_FILE = (
    "models/xgboost_assignment_cost.json"
)

TEST_PREDICTIONS_FILE = (
    "ml_test_predictions.csv"
)

METRICS_FILE = (
    "ml_model_metrics.json"
)

RANDOM_SEED = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("===================================")
print("XGBoost ML Model Training")
print("===================================")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Rows:",
    len(df)
)

print(
    "Tasks:",
    df["instance_name"].nunique()
)


# ============================================================
# TARGET
# ============================================================

TARGET = "assignment_cost"


# ============================================================
# FEATURES
#
# IMPORTANT:
# Do NOT use:
#   instance_name
#   machine_id
#   candidate_rank
#
# Otherwise the model can memorize the dataset structure.
# ============================================================

FEATURES = [

    # Task characteristics
    "task_type",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
    "task_duration",

    # Task demand
    "cpu_demand_ratio",
    "mem_demand_ratio",
    "log_cpu_demand",
    "log_mem_demand",

    # Machine characteristics
    "cpu_num",
    "mem_size",

    # Current machine utilization
    "cpu_util",
    "mem_util",
    "cpu_pressure",
    "mem_pressure",
    "bottleneck_pressure",

    # Machine headroom
    "cpu_headroom",
    "mem_headroom",

    # Task-machine interaction
    "cpu_fit_ratio",
    "mem_fit_ratio",

    # Estimated post-placement pressure
    "estimated_cpu_pressure",
    "estimated_mem_pressure",
    "log_estimated_cpu_pressure",
    "log_estimated_mem_pressure",
    "estimated_bottleneck",

    # Normalized features
    "normalized_estimated_cpu",
    "normalized_duration",
    "normalized_cpu_demand",
    "normalized_mem_demand",

    # Feasibility
    "cpu_feasible",
    "mem_feasible",
    "feasible"
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
# REMOVE INVALID TARGET ROWS
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=FEATURES + [TARGET]
).reset_index(drop=True)


print(
    "Rows after cleaning:",
    len(df)
)


# ============================================================
# TASK-LEVEL SPLIT
#
# IMPORTANT:
# All 5 candidates belonging to the same task
# must stay in the same split.
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

num_tasks = len(
    unique_tasks
)

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


train_df = df[
    df["instance_name"].isin(
        train_tasks
    )
].copy()

validation_df = df[
    df["instance_name"].isin(
        validation_tasks
    )
].copy()

test_df = df[
    df["instance_name"].isin(
        test_tasks
    )
].copy()


# ============================================================
# SPLIT VALIDATION
# ============================================================

assert (
    len(
        train_tasks
        & validation_tasks
    )
    == 0
)

assert (
    len(
        train_tasks
        & test_tasks
    )
    == 0
)

assert (
    len(
        validation_tasks
        & test_tasks
    )
    == 0
)


print(
    "\n==================================="
)

print(
    "Task-Level Dataset Split"
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

X_train = train_df[
    FEATURES
]

y_train = train_df[
    TARGET
]

X_validation = validation_df[
    FEATURES
]

y_validation = validation_df[
    TARGET
]

X_test = test_df[
    FEATURES
]

y_test = test_df[
    TARGET
]


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

print(
    "\n==================================="
)

print(
    "Creating XGBoost Model"
)

print(
    "==================================="
)

model = XGBRegressor(

    objective="reg:squarederror",

    n_estimators=600,

    learning_rate=0.05,

    max_depth=7,

    min_child_weight=3,

    subsample=0.85,

    colsample_bytree=0.85,

    reg_alpha=0.05,

    reg_lambda=1.0,

    random_state=RANDOM_SEED,

    n_jobs=-1,

    tree_method="hist"
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nStarting training..."
)

model.fit(

    X_train,

    y_train,

    eval_set=[
        (
            X_train,
            y_train
        ),
        (
            X_validation,
            y_validation
        )
    ],

    verbose=False
)


print(
    "Training complete."
)


# ============================================================
# PREDICTIONS
# ============================================================

train_predictions = (
    model.predict(
        X_train
    )
)

validation_predictions = (
    model.predict(
        X_validation
    )
)

test_predictions = (
    model.predict(
        X_test
    )
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ============================================================
# CALCULATE METRICS
# ============================================================

train_metrics = calculate_metrics(
    y_train,
    train_predictions
)

validation_metrics = calculate_metrics(
    y_validation,
    validation_predictions
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions
)


# ============================================================
# PRINT METRICS
# ============================================================

print(
    "\n==================================="
)

print(
    "ML Model Performance"
)

print(
    "==================================="
)

print(
    "\nTraining:"
)

print(
    "MAE:",
    round(
        train_metrics["MAE"],
        6
    )
)

print(
    "RMSE:",
    round(
        train_metrics["RMSE"],
        6
    )
)

print(
    "R²:",
    round(
        train_metrics["R2"],
        6
    )
)


print(
    "\nValidation:"
)

print(
    "MAE:",
    round(
        validation_metrics["MAE"],
        6
    )
)

print(
    "RMSE:",
    round(
        validation_metrics["RMSE"],
        6
    )
)

print(
    "R²:",
    round(
        validation_metrics["R2"],
        6
    )
)


print(
    "\nTest:"
)

print(
    "MAE:",
    round(
        test_metrics["MAE"],
        6
    )
)

print(
    "RMSE:",
    round(
        test_metrics["RMSE"],
        6
    )
)

print(
    "R²:",
    round(
        test_metrics["R2"],
        6
    )
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_output = test_df[
    [
        "instance_name",
        "event_time",
        "machine_id",
        "candidate_rank",
        TARGET,
        "bottleneck_pressure",
        "estimated_cpu_pressure",
        "estimated_mem_pressure",
        "cpu_util",
        "mem_util",
        "cpu_num",
        "mem_size"
    ]
].copy()

test_output[
    "predicted_assignment_cost"
] = test_predictions

test_output[
    "prediction_error"
] = (
    test_output[
        "predicted_assignment_cost"
    ]
    -
    test_output[TARGET]
)

test_output.to_csv(
    TEST_PREDICTIONS_FILE,
    index=False
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

model.save_model(
    MODEL_FILE
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "XGBRegressor",

    "target": TARGET,

    "features": FEATURES,

    "random_seed": RANDOM_SEED,

    "num_tasks": num_tasks,

    "train_tasks": len(train_tasks),

    "validation_tasks":
        len(validation_tasks),

    "test_tasks":
        len(test_tasks),

    "train_rows":
        len(train_df),

    "validation_rows":
        len(validation_df),

    "test_rows":
        len(test_df),

    "training_metrics":
        train_metrics,

    "validation_metrics":
        validation_metrics,

    "test_metrics":
        test_metrics
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
    "ML Training Complete"
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
    TEST_PREDICTIONS_FILE
)

print(
    "Metrics:",
    METRICS_FILE
)