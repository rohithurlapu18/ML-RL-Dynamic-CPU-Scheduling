import sqlite3
import csv
import time

USAGE_FILE = "required_machine_usage.csv"
DB_FILE = "machine_usage.db"

conn = sqlite3.connect(DB_FILE)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS usage (
    machine_id TEXT,
    timestamp INTEGER,
    cpu_util REAL,
    mem_util REAL
)
""")

conn.commit()

print("Reading usage data...")

start = time.time()
count = 0

with open(
    USAGE_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.reader(f)

    batch = []

    for row in reader:

        # Skip malformed rows
        if len(row) < 4:
            continue

        machine_id = row[0]

        try:
            timestamp = int(row[1])
            cpu_util = float(row[2])
            mem_util = float(row[3])
        except ValueError:
            continue

        batch.append(
            (
                machine_id,
                timestamp,
                cpu_util,
                mem_util
            )
        )

        if len(batch) >= 100_000:

            cur.executemany(
                """
                INSERT INTO usage
                VALUES (?, ?, ?, ?)
                """,
                batch
            )

            conn.commit()

            count += len(batch)

            print(
                f"Inserted {count:,} rows"
            )

            batch.clear()

    if batch:

        cur.executemany(
            """
            INSERT INTO usage
            VALUES (?, ?, ?, ?)
            """,
            batch
        )

        conn.commit()

        count += len(batch)

print("\nCreating index...")

cur.execute("""
CREATE INDEX IF NOT EXISTS
idx_usage_machine_time
ON usage(machine_id, timestamp)
""")

conn.commit()

elapsed = time.time() - start

print("\n===================================")
print("Finished!")
print("===================================")
print("Rows:", f"{count:,}")
print("Time:", round(elapsed / 60, 2), "minutes")
print("Database:", DB_FILE)

conn.close()