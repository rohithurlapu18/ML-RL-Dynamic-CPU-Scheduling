import os
import io
import zipfile

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO

from rl_scheduler_env import DynamicCPUSchedulingEnv


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_FILE = "task_machine_candidates_v2.csv"

MODEL_FILE = "models/ppo_scheduler_v2_final.zip"

OUTPUT_FILE = "ppo_v2_results.csv"


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

print("===================================")
print("PPO v2 Scheduler Evaluation")
print("===================================")

env = DynamicCPUSchedulingEnv(
    candidate_file=CANDIDATE_FILE
)

print("Tasks:", env.num_tasks)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


# ============================================================
# CREATE PPO ARCHITECTURE
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
    device="cpu",
    seed=42
)

print("PPO architecture created.")


# ============================================================
# READ TRAINED POLICY
# ============================================================

print("\nReading trained policy weights...")

with zipfile.ZipFile(
    MODEL_FILE,
    "r"
) as z:

    policy_bytes = z.read(
        "policy.pth"
    )

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
# LOAD POLICY WEIGHTS
# ============================================================

model.policy.load_state_dict(
    trained_policy
)

model.policy.eval()

print(
    "Trained policy weights loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n==================================="
)

print(
    "Evaluating PPO v2 policy"
)

print(
    "==================================="
)


results = []


for task_index in range(
    env.num_tasks
):

    task = env.task_groups[
        task_index
    ]

    # --------------------------------------------------------
    # Build observation
    # --------------------------------------------------------

    observation = (
        env._get_observation(task)
    )

    # --------------------------------------------------------
    # PPO prediction
    # --------------------------------------------------------

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action = int(action)

    # --------------------------------------------------------
    # Selected candidate
    # --------------------------------------------------------

    selected = task.iloc[action]

    # --------------------------------------------------------
    # Calculate reward
    # --------------------------------------------------------

    reward = env._calculate_reward(
        task,
        action
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "instance_name":
                selected["instance_name"],

            "event_time":
                int(
                    selected["event_time"]
                ),

            "machine_id":
                selected["machine_id"],

            "candidate_rank":
                int(
                    selected["candidate_rank"]
                ),

            "cpu_util":
                float(
                    selected["cpu_util"]
                ),

            "mem_util":
                float(
                    selected["mem_util"]
                ),

            "pressure":
                float(
                    selected["bottleneck_pressure"]
                ),

            "cpu_avg":
                float(
                    selected["cpu_avg"]
                ),

            "cpu_max":
                float(
                    selected["cpu_max"]
                ),

            "mem_avg":
                float(
                    selected["mem_avg"]
                ),

            "mem_max":
                float(
                    selected["mem_max"]
                ),

            "task_duration":
                float(
                    selected["task_duration"]
                ),

            "reward":
                float(reward),

            "estimated_cpu_pressure":
                float(
                    selected[
                        "estimated_cpu_pressure"
                    ]
                ),

            "estimated_mem_pressure":
                float(
                    selected[
                        "estimated_mem_pressure"
                    ]
                )
        }
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        (task_index + 1) % 1000 == 0
        or
        task_index + 1 == env.num_tasks
    ):

        print(
            f"Evaluated "
            f"{task_index + 1}/"
            f"{env.num_tasks}"
        )


# ============================================================
# CREATE RESULTS DATAFRAME
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

pressure = (
    results_df["pressure"]
)

cpu_util = (
    results_df["cpu_util"]
)

mem_util = (
    results_df["mem_util"]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n==================================="
)

print(
    "PPO v2 Evaluation Results"
)

print(
    "==================================="
)

print(
    "Tasks evaluated:",
    len(results_df)
)

print(
    "Average pressure:",
    round(
        pressure.mean(),
        4
    )
)

print(
    "Median pressure:",
    pressure.median()
)

print(
    "Pressure >= 90%:",
    (
        pressure >= 0.90
    ).sum()
)

print(
    "Pressure >= 95%:",
    (
        pressure >= 0.95
    ).sum()
)

print(
    "Pressure >= 98%:",
    (
        pressure >= 0.98
    ).sum()
)

print(
    "Average CPU utilization (%):",
    round(
        cpu_util.mean(),
        2
    )
)

print(
    "Average memory utilization (%):",
    round(
        mem_util.mean(),
        2
    )
)

# ============================================================
# CANDIDATE DISTRIBUTION
# ============================================================

print(
    "\nCandidate rank distribution:"
)

print(
    results_df[
        "candidate_rank"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# UNIQUE MACHINES
# ============================================================

print(
    "\nUnique machines selected:",
    results_df[
        "machine_id"
    ].nunique()
)


print(
    "\nTop selected machines:"
)

print(
    results_df[
        "machine_id"
    ]
    .value_counts()
    .head(10)
    .to_string()
)


# ============================================================
# REWARD
# ============================================================

print(
    "\nAverage reward:",
    round(
        results_df["reward"].mean(),
        4
    )
)

print(
    "Median reward:",
    round(
        results_df["reward"].median(),
        4
    )
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n==================================="
)

print(
    "Evaluation Complete"
)

print(
    "==================================="
)

print(
    "Saved:",
    OUTPUT_FILE
)


env.close()