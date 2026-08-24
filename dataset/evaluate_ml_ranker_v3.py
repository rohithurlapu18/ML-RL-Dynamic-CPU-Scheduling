import numpy as np
import pandas as pd
from xgboost import XGBRanker


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "task_machine_candidates_v2.csv"

MODEL_FILE = "models/xgboost_ranker_v3.json"

OUTPUT_FILE = "ml_ranker_v3_scheduler_results.csv"

RANDOM_SEED = 42


# ============================================================
# FEATURES
# Must exactly match the v3 training features
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
# LOAD DATA
# ============================================================

print("===================================")
print("ML Ranker v3 Scheduler Evaluation")
print("===================================")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))
print("Total tasks:", df["instance_name"].nunique())


# ============================================================
# RECREATE EXACT TEST SPLIT
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

test_tasks = set(
    unique_tasks[
        validation_end:
    ]
)


test_df = df[
    df["instance_name"].isin(
        test_tasks
    )
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
# LOAD RANKER
# ============================================================

print(
    "\nLoading XGBoost Ranker v3..."
)

model = XGBRanker()

model.load_model(
    MODEL_FILE
)

print(
    "Ranker loaded successfully."
)


# ============================================================
# PREDICT RANKING SCORE
# ============================================================

print(
    "\nPredicting candidate rankings..."
)

test_df[
    "ranking_prediction"
] = model.predict(
    test_df[FEATURES]
)


# ============================================================
# ML SCHEDULER
#
# XGBRanker gives a larger score to the preferred
# candidate, therefore choose the MAXIMUM prediction.
# ============================================================

ml_selected = (
    test_df
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
        "instance_name",
        as_index=False
    )
    .first()
)


# ============================================================
# TRADITIONAL BASELINE
#
# Candidate rank 1.
# ============================================================

baseline = (
    test_df[
        test_df["candidate_rank"] == 1
    ]
    .sort_values(
        "instance_name"
    )
    .drop_duplicates(
        "instance_name"
    )
)


# ============================================================
# ML RESULTS
# ============================================================

ml_pressure = (
    ml_selected[
        "bottleneck_pressure"
    ]
)

baseline_pressure = (
    baseline[
        "bottleneck_pressure"
    ]
)


print(
    "\n==================================="
)

print(
    "ML Ranker v3 Results"
)

print(
    "==================================="
)

print(
    "Tasks evaluated:",
    len(ml_selected)
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
    int(
        (
            ml_pressure >= 0.90
        ).sum()
    )
)

print(
    "Pressure >= 95%:",
    int(
        (
            ml_pressure >= 0.95
        ).sum()
    )
)

print(
    "Pressure >= 98%:",
    int(
        (
            ml_pressure >= 0.98
        ).sum()
    )
)

print(
    "Average CPU utilization (%):",
    round(
        ml_selected[
            "cpu_util"
        ].mean(),
        2
    )
)

print(
    "Average memory utilization (%):",
    round(
        ml_selected[
            "mem_util"
        ].mean(),
        2
    )
)


# ============================================================
# TRADITIONAL RESULTS
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
    "Tasks evaluated:",
    len(baseline)
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
    int(
        (
            baseline_pressure >= 0.90
        ).sum()
    )
)

print(
    "Pressure >= 95%:",
    int(
        (
            baseline_pressure >= 0.95
        ).sum()
    )
)

print(
    "Pressure >= 98%:",
    int(
        (
            baseline_pressure >= 0.98
        ).sum()
    )
)

print(
    "Average CPU utilization (%):",
    round(
        baseline[
            "cpu_util"
        ].mean(),
        2
    )
)

print(
    "Average memory utilization (%):",
    round(
        baseline[
            "mem_util"
        ].mean(),
        2
    )
)


# ============================================================
# PRESSURE IMPROVEMENT
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
    ) * 100

else:

    improvement = 0.0


print(
    "\n==================================="
)

print(
    "ML v3 vs Traditional"
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
# AGREEMENT WITH BASELINE
# ============================================================

agreement = (
    ml_selected[
        "candidate_rank"
    ].values
    ==
    1
)


print(
    "ML selected rank-1:",
    int(
        agreement.sum()
    )
)

print(
    "ML selected different candidate:",
    int(
        (~agreement).sum()
    )
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
    ml_selected[
        "candidate_rank"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# UNIQUE MACHINES
# ============================================================

print(
    "\nUnique machines selected:",
    ml_selected[
        "machine_id"
    ].nunique()
)


print(
    "\nTop selected machines:"
)

print(
    ml_selected[
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

    "ranking_prediction"
]


ml_selected[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n==================================="
)

print(
    "ML Ranker v3 Evaluation Complete"
)

print(
    "==================================="
)

print(
    "Saved:",
    OUTPUT_FILE
)