import io
import zipfile
import torch
import pandas as pd

from stable_baselines3 import PPO

from rl_scheduler_env import DynamicCPUSchedulingEnv


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = "models/ppo_scheduler_500000_steps.zip"
CANDIDATE_FILE = "task_machine_candidates_v2.csv"
OUTPUT_FILE = "ppo_results.csv"


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

print("===================================")
print("PPO Policy Recovery")
print("===================================")

env = DynamicCPUSchedulingEnv(
    candidate_file=CANDIDATE_FILE
)

print("Tasks:", env.num_tasks)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


# ============================================================
# CREATE SAME PPO ARCHITECTURE
# ============================================================

print("\nCreating PPO architecture...")

model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,
    vf_coef=0.5,

    max_grad_norm=0.5,

    verbose=0,

    device="cpu"
)

print("PPO architecture created.")


# ============================================================
# EXTRACT TRAINED POLICY WEIGHTS
# ============================================================

print("\nReading trained policy weights...")

with zipfile.ZipFile(
    CHECKPOINT,
    "r"
) as z:

    policy_bytes = z.read(
        "policy.pth"
    )


# ============================================================
# LOAD POLICY STATE DICT
# ============================================================

trained_policy = torch.load(
    io.BytesIO(policy_bytes),
    map_location="cpu",
    weights_only=True
)

print(
    "Policy tensors:",
    len(trained_policy)
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.policy.load_state_dict(
    trained_policy
)

print("Trained policy weights loaded successfully.")


# ============================================================
# RESET ENVIRONMENT
# ============================================================

observation, info = env.reset(
    seed=42
)


# ============================================================
# EVALUATION
# ============================================================

print("\n===================================")
print("Evaluating recovered PPO policy")
print("===================================\n")

results = []


for task_index in range(env.num_tasks):

    # --------------------------------------------------------
    # PPO prediction
    # --------------------------------------------------------

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action = int(action)

    # --------------------------------------------------------
    # Current task
    # --------------------------------------------------------

    task = env.current_task

    selected_machine = task.iloc[action]

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "instance_name":
                selected_machine["instance_name"],

            "event_time":
                int(selected_machine["event_time"]),

            "machine_id":
                selected_machine["machine_id"],

            "candidate_rank":
                int(selected_machine["candidate_rank"]),

            "cpu_num":
                selected_machine["cpu_num"],

            "mem_size":
                selected_machine["mem_size"],

            "cpu_util":
                selected_machine["cpu_util"],

            "mem_util":
                selected_machine["mem_util"],

            "cpu_pressure":
                selected_machine["cpu_pressure"],

            "mem_pressure":
                selected_machine["mem_pressure"],

            "pressure":
                selected_machine["bottleneck_pressure"],

            "cpu_avg":
                selected_machine["cpu_avg"],

            "cpu_max":
                selected_machine["cpu_max"],

            "mem_avg":
                selected_machine["mem_avg"],

            "mem_max":
                selected_machine["mem_max"],

            "task_duration":
                selected_machine["task_duration"],

            "cpu_demand_ratio":
                selected_machine["cpu_demand_ratio"],

            "mem_demand_ratio":
                selected_machine["mem_demand_ratio"]
        }
    )

    # --------------------------------------------------------
    # Environment step
    # --------------------------------------------------------

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (task_index + 1) % 1000 == 0:

        print(
            f"Evaluated "
            f"{task_index + 1}/{env.num_tasks}"
        )

    if terminated or truncated:
        break


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# METRICS
# ============================================================

print("\n===================================")
print("PPO Evaluation Results")
print("===================================")

print(
    "Tasks evaluated:",
    len(results_df)
)

print(
    "Average pressure:",
    round(
        results_df["pressure"].mean(),
        4
    )
)

print(
    "Median pressure:",
    results_df["pressure"].median()
)

print(
    "Pressure >= 90%:",
    (
        results_df["pressure"] >= 0.90
    ).sum()
)

print(
    "Pressure >= 95%:",
    (
        results_df["pressure"] >= 0.95
    ).sum()
)

print(
    "Pressure >= 98%:",
    (
        results_df["pressure"] >= 0.98
    ).sum()
)

print(
    "Average CPU utilization (%):",
    round(
        results_df["cpu_util"].mean(),
        2
    )
)

print(
    "Average memory utilization (%):",
    round(
        results_df["mem_util"].mean(),
        2
    )
)


# ============================================================
# MACHINE USAGE
# ============================================================

print(
    "\nUnique machines selected:",
    results_df["machine_id"].nunique()
)

print("\nTop selected machines:")

print(
    results_df[
        "machine_id"
    ]
    .value_counts()
    .head(10)
    .to_string()
)


# ============================================================
# FINISH
# ============================================================

print("\n===================================")
print("Evaluation Complete")
print("===================================")

print(
    "Saved:",
    OUTPUT_FILE
)

env.close()