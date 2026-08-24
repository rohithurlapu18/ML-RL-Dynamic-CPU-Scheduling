import pandas as pd
import tarfile
import csv

# Load our prepared instance dataset
instances = pd.read_csv("instance_machine_usage_ready.csv")

# Convert start_time to integer
instances["start_time"] = instances["start_time"].astype(int)

# Machines we actually need
needed_machines = set(instances["machine_id"])

print("Instances:", len(instances))
print("Machines needed:", len(needed_machines))

# Store the latest usage record found for each machine
latest_usage = {}

# Open the compressed machine usage dataset
tar = tarfile.open("machine_usage.tar.gz", "r:gz")
file = tar.extractfile("machine_usage.csv")

print("Machine usage archive opened successfully.")

# Read machine usage records
reader = csv.reader(
    line.decode().strip()
    for line in file
)

count = 0

for row in reader:
    machine_id = row[0]

    # Ignore machines we don't need
    if machine_id not in needed_machines:
        continue

    timestamp = int(row[1])

    cpu_util = float(row[2]) if row[2] else None
    mem_util = float(row[3]) if row[3] else None

    # Keep the latest usage record for this machine
    latest_usage[machine_id] = {
        "timestamp": timestamp,
        "cpu_util": cpu_util,
        "mem_util": mem_util
    }

    count += 1

    if count % 1000000 == 0:
        print("Processed usage rows:", count)

print("Relevant usage rows processed:", count)

tar.close()
print("Relevant usage rows processed:", count)

tar.close()