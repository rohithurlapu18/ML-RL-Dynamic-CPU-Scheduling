import pandas as pd


INPUT_FILE = "task_machine_candidates_v2.csv"
OUTPUT_FILE = "baseline_results.csv"


# ============================================================
# LOAD CANDIDATES
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("===================================")
print("Baseline Scheduler")
print("===================================")

print("Candidate rows:", len(df))
print("Tasks:", df["instance_name"].nunique())


# ============================================================
# CALCULATE CURRENT MACHINE PRESSURE
# ============================================================

df["pressure"] = df[
    [
        "cpu_pressure",
        "mem_pressure"
    ]
].max(axis=1)


# ============================================================
# BASELINE POLICY
#
# Select the candidate machine with the lowest
# current resource pressure.
# ============================================================

selected = (
    df.sort_values(
        [
            "instance_name",
            "pressure",
            "candidate_rank"
        ]
    )
    .groupby(
        "instance_name",
        as_index=False
    )
    .first()
)


# ============================================================
# METRICS
# ============================================================

print("\n===================================")
print("Baseline Results")
print("===================================")

print(
    "Tasks:",
    len(selected)
)

print(
    "Average pressure:",
    round(
        selected["pressure"].mean(),
        4
    )
)

print(
    "Median pressure:",
    selected["pressure"].median()
)

print(
    "Pressure >= 90%:",
    (
        selected["pressure"] >= 0.90
    ).sum()
)

print(
    "Pressure >= 95%:",
    (
        selected["pressure"] >= 0.95
    ).sum()
)

print(
    "Pressure >= 98%:",
    (
        selected["pressure"] >= 0.98
    ).sum()
)

print(
    "Average CPU utilization:",
    round(
        selected["cpu_util"].mean(),
        4
    )
)

print(
    "Average memory utilization:",
    round(
        selected["mem_util"].mean(),
        4
    )
)


# ============================================================
# RESOURCE UTILIZATION PERCENTAGES
# ============================================================

print(
    "Average CPU utilization (%):",
    round(
        selected["cpu_util"].mean(),
        2
    )
)

print(
    "Average memory utilization (%):",
    round(
        selected["mem_util"].mean(),
        2
    )
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

    "cpu_pressure",
    "mem_pressure",
    "pressure",

    "cpu_num",
    "mem_size",

    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",

    "task_duration",

    "cpu_demand_ratio",
    "mem_demand_ratio",

    "log_cpu_demand",
    "log_mem_demand"
]


selected[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:",
    OUTPUT_FILE
)