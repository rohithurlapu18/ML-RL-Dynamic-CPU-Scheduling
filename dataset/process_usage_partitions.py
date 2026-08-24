import pandas as pd
import numpy as np
import os

PARTITION_DIR = "usage_partitions"
EVENT_FILE = "scheduling_events.csv"
OUTPUT_FILE = "machine_state_events.csv"

# --------------------------------------------------
# Load event timestamps
# --------------------------------------------------

events = pd.read_csv(EVENT_FILE)

event_times = np.sort(
    events["start_time"].unique()
)

print("Events:", len(event_times))

# --------------------------------------------------
# Output
# --------------------------------------------------

first_file = True
total_rows = 0
total_machines = set()

# --------------------------------------------------
# Process one partition at a time
# --------------------------------------------------

partition_files = sorted(
    f for f in os.listdir(PARTITION_DIR)
    if f.endswith(".csv")
)

print("Partitions found:", len(partition_files))

for number, filename in enumerate(partition_files, 1):

    path = os.path.join(
        PARTITION_DIR,
        filename
    )

    print(
        f"\n[{number}/{len(partition_files)}] "
        f"Processing {filename}"
    )

    df = pd.read_csv(
        path,
        header=None,
        names=[
            "machine_id",
            "timestamp",
            "cpu_util",
            "mem_util"
        ]
    )

    if df.empty:
        continue

    # --------------------------------------------------
    # Sort chronologically for each machine
    # --------------------------------------------------

    df = df.sort_values(
        ["machine_id", "timestamp"]
    )

    # --------------------------------------------------
    # Process machines in this partition
    # --------------------------------------------------

    partition_results = []

    for machine_id, group in df.groupby(
        "machine_id",
        sort=False
    ):

        timestamps = (
            group["timestamp"]
            .to_numpy()
        )

        cpu = (
            group["cpu_util"]
            .to_numpy()
        )

        mem = (
            group["mem_util"]
            .to_numpy()
        )

        # Find latest usage <= each event
        indices = np.searchsorted(
            timestamps,
            event_times,
            side="right"
        ) - 1

        valid = indices >= 0

        if not valid.any():
            continue

        machine_events = pd.DataFrame({
            "event_time":
                event_times[valid],

            "machine_id":
                machine_id,

            "usage_timestamp":
                timestamps[indices[valid]],

            "cpu_util":
                cpu[indices[valid]],

            "mem_util":
                mem[indices[valid]]
        })

        partition_results.append(
            machine_events
        )

        total_machines.add(machine_id)

    # --------------------------------------------------
    # Write this partition's result
    # --------------------------------------------------

    if partition_results:

        result = pd.concat(
            partition_results,
            ignore_index=True
        )

        result.to_csv(
            OUTPUT_FILE,
            mode="w" if first_file else "a",
            header=first_file,
            index=False
        )

        first_file = False

        total_rows += len(result)

        print(
            "State rows written:",
            f"{total_rows:,}"
        )

    # Release memory
    del df
    del partition_results

# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\n===================================")
print("Machine-state extraction complete")
print("===================================")

print(
    "State rows:",
    f"{total_rows:,}"
)

print(
    "Machines:",
    len(total_machines)
)

print(
    "Events:",
    len(event_times)
)

print(
    "Expected maximum rows:",
    f"{len(event_times) * len(total_machines):,}"
)

print(
    "Output:",
    OUTPUT_FILE
)