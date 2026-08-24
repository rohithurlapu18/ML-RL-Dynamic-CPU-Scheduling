import csv
import os
import shutil
from collections import defaultdict

INPUT_FILE = "required_machine_usage.csv"
OUTPUT_DIR = "usage_partitions"

# Number of partitions.
# 64 gives reasonably sized files while keeping
# the number of files manageable.
NUM_PARTITIONS = 64

# --------------------------------------------------
# Load required machines
# --------------------------------------------------

import pandas as pd

instances = pd.read_csv(
    "resource_allocation_dataset.csv",
    usecols=["machine_id"]
)

required_machines = set(
    instances["machine_id"].unique()
)

print("Required machines:", len(required_machines))

# --------------------------------------------------
# Recreate output directory
# --------------------------------------------------

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)

# --------------------------------------------------
# Open partition files
# --------------------------------------------------

files = {}
writers = {}

for i in range(NUM_PARTITIONS):

    path = os.path.join(
        OUTPUT_DIR,
        f"part_{i:02d}.csv"
    )

    f = open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    )

    files[i] = f
    writers[i] = csv.writer(f)

# --------------------------------------------------
# Stream usage data
# --------------------------------------------------

processed = 0
kept = 0

with open(
    INPUT_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.reader(f)

    for row in reader:

        if len(row) < 4:
            continue

        machine_id = row[0]

        if machine_id not in required_machines:
            continue

        try:
            timestamp = int(row[1])
            cpu = float(row[2])
            mem = float(row[3])
        except ValueError:
            continue

        # Partition based on machine ID.
        partition = hash(machine_id) % NUM_PARTITIONS

        writers[partition].writerow([
            machine_id,
            timestamp,
            cpu,
            mem
        ])

        kept += 1

        if kept % 1_000_000 == 0:
            print(
                f"Saved relevant rows: {kept:,}"
            )

# --------------------------------------------------
# Close files
# --------------------------------------------------

for f in files.values():
    f.close()

print("\n===================================")
print("Partitioning complete")
print("===================================")
print("Relevant rows:", f"{kept:,}")
print("Partitions:", NUM_PARTITIONS)
print("Output directory:", OUTPUT_DIR)