import pandas as pd

# --------------------------------------------------
# 1. Load instances
# --------------------------------------------------

instances = pd.read_csv("instance_machine_usage_ready.csv")

instances["start_time"] = pd.to_numeric(
    instances["start_time"]
)

print("Instances:", len(instances))
print("Machines:", instances["machine_id"].nunique())


# --------------------------------------------------
# 2. Usage columns
# --------------------------------------------------

usage_columns = [
    "machine_id",
    "timestamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent"
]


# --------------------------------------------------
# 3. Load usage data in chunks
# --------------------------------------------------

chunk_size = 1_000_000

best_matches = {}

reader = pd.read_csv(
    "required_machine_usage.csv",
    names=usage_columns,
    header=0,
    chunksize=chunk_size
)

processed = 0

for chunk in reader:

    processed += len(chunk)

    print("Processed usage rows:", processed)

    chunk["timestamp"] = pd.to_numeric(
        chunk["timestamp"],
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["timestamp"]
    )

    chunk["timestamp"] = chunk["timestamp"].astype(int)

    # --------------------------------------------------
    # Only machines needed by our instances
    # --------------------------------------------------

    needed = set(instances["machine_id"])

    chunk = chunk[
        chunk["machine_id"].isin(needed)
    ]

    if chunk.empty:
        continue

    # --------------------------------------------------
    # Process each machine in this chunk
    # --------------------------------------------------

    for machine_id, usage in chunk.groupby("machine_id"):

        usage = usage.sort_values(
            "timestamp"
        )

        # Instances belonging to this machine
        task = instances[
            instances["machine_id"] == machine_id
        ][
            ["instance_name", "start_time"]
        ].copy()

        if task.empty:
            continue

        task = task.sort_values(
            "start_time"
        )

        # --------------------------------------------------
        # Time-aware matching inside this chunk
        # --------------------------------------------------

        matched = pd.merge_asof(
            task,
            usage[
                [
                    "timestamp",
                    "cpu_util_percent",
                    "mem_util_percent"
                ]
            ],
            left_on="start_time",
            right_on="timestamp",
            direction="backward"
        )

        # --------------------------------------------------
        # Keep only valid matches
        # --------------------------------------------------

        matched = matched.dropna(
            subset=["timestamp"]
        )

        # --------------------------------------------------
        # Update global best match
        #
        # If another chunk has a later valid timestamp,
        # replace the previous match.
        # --------------------------------------------------

        for _, row in matched.iterrows():

            instance_name = row["instance_name"]

            timestamp = int(row["timestamp"])

            if (
                instance_name not in best_matches
                or timestamp > best_matches[
                    instance_name
                ]["usage_timestamp"]
            ):

                best_matches[instance_name] = {
                    "usage_timestamp": timestamp,
                    "machine_cpu_util": row[
                        "cpu_util_percent"
                    ],
                    "machine_mem_util": row[
                        "mem_util_percent"
                    ]
                }


# --------------------------------------------------
# 4. Convert matches to DataFrame
# --------------------------------------------------

print("\nCreating final dataset...")

matches = pd.DataFrame.from_dict(
    best_matches,
    orient="index"
)

matches.index.name = "instance_name"

matches = matches.reset_index()


# --------------------------------------------------
# 5. Merge with original instances
# --------------------------------------------------

final = instances.merge(
    matches,
    on="instance_name",
    how="left"
)


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

final.to_csv(
    "final_resource_dataset.csv",
    index=False
)


# --------------------------------------------------
# 7. Report
# --------------------------------------------------

print("\n===================================")
print("Finished!")
print("===================================")

print("Final rows:", len(final))

print("\nMissing usage values:")

print(
    final[
        [
            "usage_timestamp",
            "machine_cpu_util",
            "machine_mem_util"
        ]
    ].isna().sum()
)

print("\nMatched instances:", len(matches))

print("\nSaved:")
print("final_resource_dataset.csv")