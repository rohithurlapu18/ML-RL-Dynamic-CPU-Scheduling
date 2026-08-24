import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "ml_assignment_dataset.csv"

MODEL_FILE = "models/xgboost_assignment_cost_v2.json"

OUTPUT_FILE = "ml_scheduler_results.csv"


# ============================================================
# FEATURES
# Must exactly match ML v2 training
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
    "bottleneck_pressure",

    "cpu_headroom",
    "mem_headroom",

    "cpu_fit_ratio",
    "mem_fit_ratio"
]


# ============================================================
# LOAD DATA
# ============================================================

print("===================================")
print("ML Scheduler Evaluation")
print("===================================")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))
print("Total tasks:", df["instance_name"].nunique())


# ============================================================
# RECREATE SAME TASK-LEVEL TEST SPLIT
# ============================================================

RANDOM_SEED = 42

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

test_tasks = set(
    unique_tasks[
        validation_end:
    ]
)

test_df = df[
    df["instance_name"].isin(test_tasks)
].copy()


print(
    "\nTest tasks:",
    test_df["instance_name"].nunique()
)

print(
    "Test rows:",
    len(test_df)
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading XGBoost v2 model..."
)

model = XGBRegressor()

model.load_model(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)


# ============================================================
# PREDICT COST FOR EVERY CANDIDATE
# ============================================================

print(
    "\nPredicting assignment cost..."
)

test_df[
    "predicted_assignment_cost"
] = model.predict(
    test_df[FEATURES]
)


# ============================================================
# ML SCHEDULER
#
# For each task:
# choose candidate with minimum predicted cost.
# ============================================================

selected_ml = (
    test_df
    .sort_values(
        [
            "instance_name",
            "predicted_assignment_cost"
        ]
    )
    .groupby(
        "instance_name",
        as_index=False
    )
    .first()
)


# ============================================================
# TRADITIONAL BASELINE
#
# Candidate rank 1 represents the traditional
# lowest-pressure selection.
# ============================================================

baseline = (
    test_df[
        test_df["candidate_rank"] == 1
    ]
    .copy()
)


# There should be exactly one rank-1
# candidate for every test task.

baseline = (
    baseline
    .sort_values(
        "instance_name"
    )
    .drop_duplicates(
        "instance_name"
    )
)


# ============================================================
# ML METRICS
# ============================================================

ml_pressure = (
    selected_ml["bottleneck_pressure"]
)

baseline_pressure = (
    baseline["bottleneck_pressure"]
)


print(
    "\n==================================="
)

print(
    "ML Scheduler Results"
)

print(
    "==================================="
)

print(
    "Tasks evaluated:",
    len(selected_ml)
)

print(
    "Average pressure:",
    round(
        ml_pressure.mean(),
        4
    )
)

print(
    "Median pressure:",
    ml_pressure.median()
)

print(
    "Pressure >= 90%:",
    (
        ml_pressure >= 0.90
    ).sum()
)

print(
    "Pressure >= 95%:",
    (
        ml_pressure >= 0.95
    ).sum()
)

print(
    "Pressure >= 98%:",
    (
        ml_pressure >= 0.98
    ).sum()
)

print(
    "Average CPU utilization (%):",
    round(
        selected_ml["cpu_util"].mean(),
        2
    )
)

print(
    "Average memory utilization (%):",
    round(
        selected_ml["mem_util"].mean(),
        2
    )
)


# ============================================================
# BASELINE ON SAME TEST TASKS
# ============================================================

print(
    "\n==================================="
)

print(
    "Traditional Baseline"
)

print(
    "==================================="
)

print(
    "Average pressure:",
    round(
        baseline_pressure.mean(),
        4
    )
)

print(
    "Median pressure:",
    baseline_pressure.median()
)

print(
    "Pressure >= 90%:",
    (
        baseline_pressure >= 0.90
    ).sum()
)

print(
    "Pressure >= 95%:",
    (
        baseline_pressure >= 0.95
    ).sum()
)

print(
    "Pressure >= 98%:",
    (
        baseline_pressure >= 0.98
    ).sum()
)

print(
    "Average CPU utilization (%):",
    round(
        baseline["cpu_util"].mean(),
        2
    )
)

print(
    "Average memory utilization (%):",
    round(
        baseline["mem_util"].mean(),
        2
    )
)


# ============================================================
# IMPROVEMENT
# ============================================================

baseline_avg = (
    baseline_pressure.mean()
)

ml_avg = (
    ml_pressure.mean()
)

if baseline_avg != 0:

    improvement = (
        (
            baseline_avg - ml_avg
        )
        /
        baseline_avg
    )* 100

else:

    improvement = 0.0


print(
    "\n==================================="
)

print(
    "ML vs Traditional"
)

print(
    "==================================="
)

print(
    "Pressure improvement (%):",
    round(
        improvement,
        2
    )
)


# ============================================================
# AGREEMENT
# ============================================================

agreement = (
    selected_ml["candidate_rank"]
    == 1
)

print(
    "ML selected traditional rank-1:",
    int(agreement.sum())
)

print(
    "ML selected different candidate:",
    int((~agreement).sum())
)

print(
    "Agreement (%):",
    round(
        agreement.mean() * 100,
        2
    )
)


# ============================================================
# CANDIDATE DISTRIBUTION
# ============================================================

print(
    "\n==================================="
)

print(
    "ML Candidate Distribution"
)

print(
    "==================================="
)

print(
    selected_ml[
        "candidate_rank"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# MACHINE DISTRIBUTION
# ============================================================

print(
    "\nUnique machines selected:",
    selected_ml[
        "machine_id"
    ].nunique()
)

print(
    "\nTop selected machines:"
)

print(
    selected_ml[
        "machine_id"
    ]
    .value_counts()
    .head(10)
    .to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_columns = [
    "instance_name",
    "event_time",
    "machine_id",
    "candidate_rank",

    "cpu_util",
    "mem_util",
    "bottleneck_pressure",

    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",

    "task_duration",

    "predicted_assignment_cost",

    "assignment_cost"
]

selected_ml[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n==================================="
)

print(
    "ML Scheduler Evaluation Complete"
)

print(
    "==================================="
)

print(
    "Saved:",
    OUTPUT_FILE
)