import pandas as pd
import numpy as np

TASK_FILE = "resource_allocation_dataset.csv"
STATE_FILE = "machine_state_events.csv"
OUTPUT_FILE = "task_machine_candidates.csv"

TOP_K = 5

# --------------------------------------------------
# Load task data
# --------------------------------------------------

tasks = pd.read_csv(TASK_FILE)

# Event time = task arrival
tasks = tasks.rename(
    columns={"start_time": "event_time"}
)

task_columns = [
    "instance_name",
    "event_time",
    "task_name",
    "job_name",
    "task_type",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
    "task_duration",
    "cpu_num",
    "mem_size"
]

tasks = tasks[task_columns]

print("Tasks:", len(tasks))

# --------------------------------------------------
# Load machine states
# --------------------------------------------------

states = pd.read_csv(STATE_FILE)

states["cpu_pressure"] = (
    states["cpu_util"] / 100.0
)

states["mem_pressure"] = (
    states["mem_util"] / 100.0
)

states["bottleneck_pressure"] = states[
    ["cpu_pressure", "mem_pressure"]
].max(axis=1)

print("Machine-state rows:", len(states))

# --------------------------------------------------
# Select top-K least pressured machines per event
# --------------------------------------------------

states = states.sort_values(
    [
        "event_time",
        "bottleneck_pressure",
        "cpu_pressure",
        "mem_pressure"
    ]
)

states["candidate_rank"] = (
    states
    .groupby("event_time")
    .cumcount()
    + 1
)

candidates = states[
    states["candidate_rank"] <= TOP_K
].copy()

# --------------------------------------------------
# Join task information
# --------------------------------------------------

candidates = candidates.merge(
    tasks,
    on="event_time",
    how="inner"
)

# --------------------------------------------------
# Save
# --------------------------------------------------

columns = [
    "instance_name",
    "event_time",
    "candidate_rank",
    "machine_id",
    "cpu_num",
    "mem_size",
    "cpu_util",
    "mem_util",
    "cpu_pressure",
    "mem_pressure",
    "bottleneck_pressure",
    "task_name",
    "job_name",
    "task_type",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
    "task_duration"
]

candidates = candidates[columns]

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===================================")
print("Candidate generation complete")
print("===================================")

print("Rows:", len(candidates))
print(
    "Tasks represented:",
    candidates["instance_name"].nunique()
)

print(
    "Candidates per task:",
    candidates.groupby(
        "instance_name"
    ).size().describe().to_string()
)

print("Output:", OUTPUT_FILE)