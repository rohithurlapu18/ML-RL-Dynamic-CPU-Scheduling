import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TASK_FILE = "resource_allocation_dataset.csv"
STATE_FILE = "machine_state_events.csv"
OUTPUT_FILE = "task_machine_candidates_v2.csv"

# Five different resource-pressure levels
PERCENTILES = [0.10, 0.30, 0.50, 0.70, 0.90]


# ============================================================
# 1. LOAD ORIGINAL TASK DATA
# ============================================================

print("Loading task dataset...")

raw_tasks = pd.read_csv(TASK_FILE)

print("Original task rows:", len(raw_tasks))


# ============================================================
# 2. EXTRACT MACHINE METADATA
# ============================================================

machine_meta = (
    raw_tasks[
        [
            "machine_id",
            "cpu_num",
            "mem_size"
        ]
    ]
    .drop_duplicates("machine_id")
    .copy()
)

print("Unique machines:", len(machine_meta))


# ============================================================
# 3. PREPARE TASK DATA
# ============================================================

tasks = raw_tasks.rename(
    columns={
        "start_time": "event_time"
    }
).copy()

tasks = tasks[
    [
        "instance_name",
        "event_time",
        "task_name",
        "job_name",
        "task_type",
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max",
        "task_duration"
    ]
].copy()

print("Tasks prepared:", len(tasks))


# ============================================================
# 4. LOAD MACHINE STATES
# ============================================================

print("\nLoading machine states...")

states = pd.read_csv(
    STATE_FILE
)

print("Machine-state rows:", len(states))


# ============================================================
# 5. ADD MACHINE CAPACITY
# ============================================================

states = states.merge(
    machine_meta,
    on="machine_id",
    how="left"
)

missing_capacity = states[
    ["cpu_num", "mem_size"]
].isna().sum()

print("\nMissing machine metadata:")
print(missing_capacity.to_string())


# ============================================================
# 6. CALCULATE MACHINE RESOURCE PRESSURE
# ============================================================

states["cpu_pressure"] = (
    states["cpu_util"] / 100.0
)

states["mem_pressure"] = (
    states["mem_util"] / 100.0
)

states["bottleneck_pressure"] = states[
    [
        "cpu_pressure",
        "mem_pressure"
    ]
].max(axis=1)


# ============================================================
# 7. SELECT FIVE DISTINCT MACHINES PER EVENT
# ============================================================

print("\nSelecting stratified candidates...")

candidate_rows = []

event_count = states["event_time"].nunique()

processed_events = 0


for event_time, group in states.groupby(
    "event_time",
    sort=True
):

    # --------------------------------------------------------
    # Sort machines from lowest to highest pressure
    # --------------------------------------------------------

    group = group.sort_values(
        [
            "bottleneck_pressure",
            "cpu_pressure",
            "mem_pressure",
            "machine_id"
        ]
    ).reset_index(drop=True)

    n = len(group)

    selected_indices = []

    # --------------------------------------------------------
    # Select approximately:
    #
    # 10th percentile
    # 30th percentile
    # 50th percentile
    # 70th percentile
    # 90th percentile
    #
    # while guaranteeing unique machines.
    # --------------------------------------------------------

    for q in PERCENTILES:

        target = int(
            round(
                q * (n - 1)
            )
        )

        distance = 0

        while True:

            possible_indices = []

            left = target - distance
            right = target + distance

            if left >= 0:
                possible_indices.append(left)

            if (
                right < n
                and right != left
            ):
                possible_indices.append(right)

            selected = None

            for idx in possible_indices:

                if idx not in selected_indices:

                    selected = idx
                    break

            if selected is not None:

                selected_indices.append(
                    selected
                )

                break

            distance += 1

    # --------------------------------------------------------
    # Save selected machines
    # --------------------------------------------------------

    for rank, idx in enumerate(
        selected_indices,
        start=1
    ):

        row = group.iloc[idx]

        candidate_rows.append(
            {
                "event_time":
                    event_time,

                "candidate_rank":
                    rank,

                "machine_id":
                    row["machine_id"],

                "cpu_num":
                    row["cpu_num"],

                "mem_size":
                    row["mem_size"],

                "cpu_util":
                    row["cpu_util"],

                "mem_util":
                    row["mem_util"],

                "cpu_pressure":
                    row["cpu_pressure"],

                "mem_pressure":
                    row["mem_pressure"],

                "bottleneck_pressure":
                    row["bottleneck_pressure"]
            }
        )

    processed_events += 1

    if processed_events % 20 == 0:

        print(
            f"Processed events: "
            f"{processed_events}/{event_count}"
        )


# ============================================================
# 8. CREATE CANDIDATE DATAFRAME
# ============================================================

candidates = pd.DataFrame(
    candidate_rows
)

print(
    "\nCandidate machine rows:",
    len(candidates)
)


# ============================================================
# 9. JOIN TASK INFORMATION
# ============================================================

print("Joining task information...")

candidates = candidates.merge(
    tasks,
    on="event_time",
    how="inner"
)


# ============================================================
# 10. CALCULATE TASK DEMAND FEATURES
# ============================================================

# Protect against division by zero.

candidates["cpu_demand_ratio"] = (
    candidates["cpu_max"]
    /
    candidates["cpu_num"].clip(lower=1)
)

candidates["mem_demand_ratio"] = (
    candidates["mem_max"]
    /
    candidates["mem_size"].clip(lower=1)
)


# Log transformation is useful because CPU demand
# has a very heavy-tailed distribution.

candidates["log_cpu_demand"] = np.log1p(
    candidates["cpu_demand_ratio"]
)

candidates["log_mem_demand"] = np.log1p(
    candidates["mem_demand_ratio"]
)


# ============================================================
# 11. ADD ESTIMATED RESOURCE PRESSURE
# ============================================================

# These are NOT hard feasibility constraints.
# They are features for the RL model.

candidates["estimated_cpu_pressure"] = (
    candidates["cpu_pressure"]
    +
    candidates["cpu_demand_ratio"]
)

candidates["estimated_mem_pressure"] = (
    candidates["mem_pressure"]
    +
    candidates["mem_demand_ratio"]
)


# ============================================================
# 12. VALIDATION
# ============================================================

print("\n===================================")
print("Candidate generation validation")
print("===================================")

print(
    "Total rows:",
    len(candidates)
)

print(
    "Unique tasks:",
    candidates["instance_name"].nunique()
)

print(
    "Unique events:",
    candidates["event_time"].nunique()
)

print(
    "Unique machines:",
    candidates["machine_id"].nunique()
)


# ------------------------------------------------------------
# Candidates per task
# ------------------------------------------------------------

candidate_counts = (
    candidates
    .groupby("instance_name")
    .size()
)

print("\nCandidates per task:")

print(
    candidate_counts
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Unique candidate machines per task
# ------------------------------------------------------------

unique_candidate_counts = (
    candidates
    .groupby("instance_name")["machine_id"]
    .nunique()
)

tasks_with_5_unique = (
    unique_candidate_counts == 5
).sum()

tasks_with_duplicates = (
    unique_candidate_counts < 5
).sum()

print(
    "\nTasks with 5 unique candidates:",
    tasks_with_5_unique
)

print(
    "Tasks with duplicate candidates:",
    tasks_with_duplicates
)


# ------------------------------------------------------------
# Check candidate ranks
# ------------------------------------------------------------

rank_counts = (
    candidates
    .groupby("instance_name")["candidate_rank"]
    .nunique()
)

print(
    "Tasks with 5 candidate ranks:",
    (rank_counts == 5).sum()
)


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\nMissing values:")

print(
    candidates.isna()
    .sum()
    .to_string()
)


# ------------------------------------------------------------
# Duplicate task-machine combinations
# ------------------------------------------------------------

duplicates = candidates.duplicated(
    [
        "instance_name",
        "machine_id"
    ]
).sum()

print(
    "\nDuplicate task-machine pairs:",
    duplicates
)


# ============================================================
# 13. SAVE FINAL CANDIDATE DATASET
# ============================================================

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===================================")
print("Candidate generation complete")
print("===================================")

print(
    "Saved:",
    OUTPUT_FILE
)