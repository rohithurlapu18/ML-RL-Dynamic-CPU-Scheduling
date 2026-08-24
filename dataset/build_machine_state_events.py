import pandas as pd
import csv

USAGE_FILE = "required_machine_usage.csv"
EVENT_FILE = "scheduling_events.csv"
INSTANCE_FILE = "resource_allocation_dataset.csv"
OUTPUT_FILE = "machine_state_events.csv"

# --------------------------------------------------
# Load events and machines
# --------------------------------------------------

events = pd.read_csv(EVENT_FILE)
event_times = sorted(events["start_time"].unique())

instances = pd.read_csv(
    INSTANCE_FILE,
    usecols=["machine_id"]
)

machine_ids = set(instances["machine_id"].unique())

print("Events:", len(event_times))
print("Machines:", len(machine_ids))
print("Target rows:", len(event_times) * len(machine_ids))

# --------------------------------------------------
# We maintain the latest known usage for each
# machine.
# --------------------------------------------------

latest = {}

# Results are stored only for the 180 event times.
results = []

event_index = 0
next_event = event_times[event_index]

processed = 0

# --------------------------------------------------
# IMPORTANT:
# We cannot assume the usage file is globally sorted.
#
# Therefore, this streaming method is safe only if
# we process the complete file and update the latest
# record for every machine.
# --------------------------------------------------

with open(
    USAGE_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.reader(f)

    for row in reader:

        if len(row) < 4:
            continue

        machine_id = row[0]

        if machine_id not in machine_ids:
            continue

        try:
            timestamp = int(row[1])
            cpu = float(row[2])
            mem = float(row[3])
        except ValueError:
            continue

        # Ignore records after final event
        if timestamp > event_times[-1]:
            continue

        previous = latest.get(machine_id)

        if previous is None or timestamp > previous[0]:

            latest[machine_id] = (
                timestamp,
                cpu,
                mem
            )

        processed += 1

        if processed % 1_000_000 == 0:
            print(
                f"Processed relevant rows: {processed:,}"
            )

# --------------------------------------------------
# Build event snapshots
# --------------------------------------------------

print("\nCreating event snapshots...")

for event_time in event_times:

    for machine_id in machine_ids:

        state = latest.get(machine_id)

        if state is None:
            continue

        timestamp, cpu, mem = state

        # IMPORTANT:
        # The latest global state may be AFTER the
        # current event, because usage rows aren't
        # globally sorted.
        #
        # Therefore this result is NOT safe for
        # historical event lookup.