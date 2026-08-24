import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_scheduler_env import DynamicCPUSchedulingEnv


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_FILE = (
    "task_machine_candidates_v2.csv"
)

MODEL_DIR = "models"

LOG_DIR = "logs"

FINAL_MODEL = (
    "models/ppo_scheduler_v2_final"
)

TOTAL_TIMESTEPS = 500_000


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

print(
    "==================================="
)

print(
    "PPO Training - Version 2"
)

print(
    "==================================="
)

env = DynamicCPUSchedulingEnv(
    candidate_file=CANDIDATE_FILE
)

print(
    "Tasks:",
    env.num_tasks
)

print(
    "Action space:",
    env.action_space
)

print(
    "Observation space:",
    env.observation_space
)


# ============================================================
# CHECKPOINT CALLBACK
# ============================================================

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_scheduler_v2"
)


# ============================================================
# CREATE PPO MODEL
# ============================================================

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

    verbose=1,

    tensorboard_log=LOG_DIR,

    device="cpu",

    seed=42
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nStarting PPO training..."
)

print(
    "Total timesteps:",
    TOTAL_TIMESTEPS
)

print(
    "\nTraining...\n"
)


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,

    callback=checkpoint_callback,

    progress_bar=False,

    reset_num_timesteps=True
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL
)


# ============================================================
# FINISH
# ============================================================

print(
    "\n==================================="
)

print(
    "Training Complete"
)

print(
    "==================================="
)

print(
    "Saved model:",
    FINAL_MODEL + ".zip"
)

print(
    "Total timesteps:",
    model.num_timesteps
)


env.close()