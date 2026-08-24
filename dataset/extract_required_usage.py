import pandas as pd
import tarfile
import csv

# Load our instances
instances = pd.read_csv("instance_machine_usage_ready.csv")

# For every machine, we need the earliest task start time
# because we will keep usage records up to that point and
# then perform the exact time-aware matching later.
machine_start = (
    instances.groupby("machine_id")["start_time"]
    .max()
    .to_dict()
)

needed_machines = set(machine_start.keys())

print("Instances:", len(instances))
print("Machines needed:", len(needed_machines))

# Open Alibaba machine usage archive
tar = tarfile.open("machine_usage.tar.gz", "r:gz")
file = tar.extractfile("machine_usage.csv")

# Output only relevant usage records
out = open("required_machine_usage.csv", "w", newline="")

writer = csv.writer(out)

# Original machine_usage columns
writer.writerow([
    "machine_id",
    "timestamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent"
])

count = 0

reader = csv.reader(
    line.decode().strip()
    for line in file
)

for row in reader:

    machine_id = row[0]

    # Ignore machines we don't need
    if machine_id not in needed_machines:
        continue

    timestamp = int(row[1])

    # We only need usage at or before the earliest
    # instance start time for this machine.
    if timestamp > machine_start[machine_id]:
        continue

    writer.writerow(row)

    count += 1

    if count % 1_000_000 == 0:
        print("Saved usage rows:", count)

out.close()
tar.close()

print("Finished.")
print("Relevant usage rows saved:", count)
print("Output: required_machine_usage.csv")