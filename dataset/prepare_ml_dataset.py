import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "task_machine_candidates_v2.csv"

OUTPUT_FILE = "ml_assignment_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("===================================")
print("ML Dataset Preparation")
print("===================================")

df = pd.read_csv(INPUT_FILE)

print(
    "Input rows:",
    len(df)
)

print(
    "Input columns:",
    len(df.columns)
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "instance_name",
    "event_time",
    "machine_id",

    "task_type",

    "cpu_num",
    "mem_size",

    "cpu_util",
    "mem_util",

    "cpu_pressure",
    "mem_pressure",
    "bottleneck_pressure",

    "cpu_avg",
    "cpu_max",

    "mem_avg",
    "mem_max",

    "task_duration",

    "cpu_demand_ratio",
    "mem_demand_ratio",

    "log_cpu_demand",
    "log_mem_demand",

    "estimated_cpu_pressure",
    "estimated_mem_pressure"
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.copy()

df = df.drop_duplicates(
    subset=[
        "instance_name",
        "machine_id"
    ]
)

print(
    "Rows after duplicate removal:",
    len(df)
)


# ============================================================
# RESOURCE HEADROOM
# ============================================================

df["cpu_headroom"] = (
    df["cpu_num"]
    *
    (
        1.0 -
        df["cpu_util"] / 100.0
    )
)

df["mem_headroom"] = (
    df["mem_size"]
    *
    (
        1.0 -
        df["mem_util"] / 100.0
    )
)


# ============================================================
# SAFE HEADROOM
# ============================================================

df["cpu_headroom_safe"] = (
    df["cpu_headroom"]
    .clip(lower=1e-6)
)

df["mem_headroom_safe"] = (
    df["mem_headroom"]
    .clip(lower=1e-6)
)


# ============================================================
# TASK / MACHINE INTERACTION FEATURES
# ============================================================

df["cpu_fit_ratio"] = (
    df["cpu_max"]
    /
    df["cpu_headroom_safe"]
)

df["mem_fit_ratio"] = (
    df["mem_max"]
    /
    df["mem_headroom_safe"]
)


# ============================================================
# ESTIMATED PRESSURE TRANSFORMATION
# ============================================================

df["log_estimated_cpu_pressure"] = np.log1p(
    df["estimated_cpu_pressure"]
)

df["log_estimated_mem_pressure"] = np.log1p(
    df["estimated_mem_pressure"]
)


# ============================================================
# ESTIMATED BOTTLENECK
# ============================================================

df["estimated_bottleneck"] = (
    df[
        [
            "estimated_cpu_pressure",
            "estimated_mem_pressure"
        ]
    ]
    .max(axis=1)
)


# ============================================================
# NORMALIZED ESTIMATED CPU PRESSURE
# ============================================================

# Dataset 95th percentile observed earlier:
# approximately 12.83

CPU_PRESSURE_SCALE = 12.83

df["normalized_estimated_cpu"] = np.clip(
    df["estimated_cpu_pressure"]
    /
    CPU_PRESSURE_SCALE,
    0.0,
    1.0
)


# ============================================================
# NORMALIZED DURATION
# ============================================================

df["normalized_duration"] = np.clip(
    df["task_duration"]
    /
    599.0,
    0.0,
    1.0
)


# ============================================================
# NORMALIZED RESOURCE DEMAND
# ============================================================

df["normalized_cpu_demand"] = np.clip(
    np.log1p(
        df["cpu_demand_ratio"]
    )
    /
    np.log1p(828.0),
    0.0,
    1.0
)

df["normalized_mem_demand"] = np.clip(
    np.log1p(
        df["mem_demand_ratio"]
    )
    /
    np.log1p(1.0),
    0.0,
    1.0
)


# ============================================================
# FEASIBILITY
# ============================================================

df["cpu_feasible"] = (
    df["cpu_max"]
    <=
    df["cpu_headroom"]
)

df["mem_feasible"] = (
    df["mem_max"]
    <=
    df["mem_headroom"]
)

df["feasible"] = (
    df["cpu_feasible"]
    &
    df["mem_feasible"]
)


# ============================================================
# ML TARGET
# ============================================================

# Lower assignment cost is better.

df["assignment_cost"] = (

    0.40
    *
    df["normalized_estimated_cpu"]

    +

    0.25
    *
    df["estimated_mem_pressure"].clip(
        0.0,
        1.0
    )

    +

    0.15
    *
    df["normalized_duration"]

    +

    0.10
    *
    df["normalized_cpu_demand"]

    +

    0.05
    *
    df["normalized_mem_demand"]

    +

    0.05
    *
    df["bottleneck_pressure"]

)


# ============================================================
# STRONG INFEASIBILITY PENALTY
# ============================================================

df.loc[
    ~df["feasible"],
    "assignment_cost"
] += 2.0


# ============================================================
# FINAL TARGET CLIPPING
# ============================================================

df["assignment_cost"] = (
    df["assignment_cost"]
    .clip(
        lower=0.0
    )
)


# ============================================================
# TARGET STATISTICS
# ============================================================

print(
    "\n==================================="
)

print(
    "Assignment Cost Statistics"
)

print(
    "==================================="
)

print(
    df["assignment_cost"]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .to_string()
)


# ============================================================
# FEASIBILITY STATISTICS
# ============================================================

print(
    "\n==================================="
)

print(
    "Feasibility"
)

print(
    "==================================="
)

print(
    "CPU feasible:",
    int(df["cpu_feasible"].sum())
)

print(
    "Memory feasible:",
    int(df["mem_feasible"].sum())
)

print(
    "Both feasible:",
    int(df["feasible"].sum())
)

print(
    "Infeasible:",
    int((~df["feasible"]).sum())
)


# ============================================================
# CANDIDATE COST BY RANK
# ============================================================

print(
    "\n==================================="
)

print(
    "Assignment Cost by Candidate Rank"
)

print(
    "==================================="
)

print(
    df.groupby(
        "candidate_rank"
    )[
        "assignment_cost"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
    .round(4)
    .to_string()
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n==================================="
)

print(
    "ML Dataset Preparation Complete"
)

print(
    "==================================="
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "Output:",
    OUTPUT_FILE
)