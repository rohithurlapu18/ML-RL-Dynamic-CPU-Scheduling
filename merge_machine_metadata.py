import pandas as pd

# Load instance data
instances = pd.read_csv("batch_instance_clean.csv")

# Load machine metadata
meta_columns = [
    "machine_id",
    "timestamp",
    "cpu_num",
    "mem_size",
    "status",
    "cpu_capacity",
    "mem_capacity"
]

meta = pd.read_csv(
    "machine_meta_relevant.csv",
    header=None,
    names=meta_columns
)

# Convert timestamps to numeric
instances["start_time"] = pd.to_numeric(instances["start_time"])
meta["timestamp"] = pd.to_numeric(meta["timestamp"])

# Sort before time-aware merge
instances = instances.sort_values(["machine_id", "start_time"])
meta = meta.sort_values(["machine_id", "timestamp"])

# Match each instance with the latest metadata
# available at or before its start time
merged = pd.merge_asof(
    instances,
    meta,
    left_on="start_time",
    right_on="timestamp",
    by="machine_id",
    direction="backward"
)

print("Original instances:", len(instances))
print("Merged instances:", len(merged))

print("\nMissing machine metadata after merge:")
print(merged[
    ["cpu_num", "mem_size", "cpu_capacity", "mem_capacity"]
].isna().sum())

merged.to_csv(
    "instance_with_machine.csv",
    index=False
)

print("\nSaved: instance_with_machine.csv")