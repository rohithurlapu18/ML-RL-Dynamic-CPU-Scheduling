import pandas as pd

columns = [
    "instance_name",
    "task_name",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "machine_id",
    "seq_no",
    "total_seq_no",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max"
]

df = pd.read_csv(
    "batch_instance_sample.csv",
    header=None,
    names=columns
)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum())