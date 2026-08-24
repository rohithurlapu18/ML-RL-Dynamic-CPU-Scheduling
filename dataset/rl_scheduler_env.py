import numpy as np
import pandas as pd
import gymnasium as gym

from gymnasium import spaces


class DynamicCPUSchedulingEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        candidate_file="task_machine_candidates_v2.csv",
        render_mode=None
    ):

        super().__init__()

        self.render_mode = render_mode

        # --------------------------------------------------------
        # Load candidate dataset
        # --------------------------------------------------------

        self.df = pd.read_csv(candidate_file)

        # --------------------------------------------------------
        # Sort correctly
        # --------------------------------------------------------

        self.df = self.df.sort_values(
            [
                "event_time",
                "instance_name",
                "candidate_rank"
            ]
        ).reset_index(drop=True)

        # --------------------------------------------------------
        # Group 5 candidates per task
        # --------------------------------------------------------

        self.task_groups = [
            group.reset_index(drop=True)
            for _, group
            in self.df.groupby(
                "instance_name",
                sort=False
            )
        ]

        # --------------------------------------------------------
        # Validate candidate count
        # --------------------------------------------------------

        invalid_groups = [
            len(group)
            for group in self.task_groups
            if len(group) != 5
        ]

        if invalid_groups:

            raise ValueError(
                "Every task must have exactly "
                "5 candidates."
            )

        # --------------------------------------------------------
        # Number of tasks
        # --------------------------------------------------------

        self.num_tasks = len(
            self.task_groups
        )

        # --------------------------------------------------------
        # Action space
        #
        # 0 -> candidate rank 1
        # 1 -> candidate rank 2
        # 2 -> candidate rank 3
        # 3 -> candidate rank 4
        # 4 -> candidate rank 5
        # --------------------------------------------------------

        self.action_space = spaces.Discrete(5)

        # --------------------------------------------------------
        # Observation
        #
        # Task features:
        #
        # 1. CPU demand
        # 2. Memory demand
        # 3. CPU average
        # 4. Memory average
        # 5. Duration
        #
        # Candidate features:
        #
        # For each of 5 machines:
        #
        # 1. CPU pressure
        # 2. Memory pressure
        # 3. Bottleneck pressure
        # 4. Estimated CPU pressure
        # 5. Estimated memory pressure
        #
        # 5 + (5 × 5) = 30
        # --------------------------------------------------------

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(30,),
            dtype=np.float32
        )

        # --------------------------------------------------------
        # Episode state
        # --------------------------------------------------------

        self.current_task_index = 0
        self.current_task = None
        self.last_action = None
        self.last_reward = None

    # ============================================================
    # NORMALIZATION FUNCTIONS
    # ============================================================

    @staticmethod
    def normalize_cpu_avg(value):

        value = float(value)

        return np.clip(
            value / 232.0,
            0.0,
            1.0
        )

    @staticmethod
    def normalize_mem_avg(value):

        value = float(value)

        return np.clip(
            value / 0.92,
            0.0,
            1.0
        )

    @staticmethod
    def normalize_duration(value):

        value = float(value)

        return np.clip(
            value / 599.0,
            0.0,
            1.0
        )

    @staticmethod
    def normalize_cpu_capacity(value):

        value = float(value)

        return np.clip(
            value / 292.0,
            0.0,
            1.0
        )

    @staticmethod
    def normalize_memory_capacity(value):

        value = float(value)

        return np.clip(
            value / 20.0,
            0.0,
            1.0
        )

    @staticmethod
    def normalize_cpu_demand(value):

        value = float(value)

        return np.clip(
            value / np.log1p(828.0),
            0.0,
            1.0
        )

    @staticmethod
    def normalize_memory_demand(value):

        value = float(value)

        return np.clip(
            value / np.log1p(1.0),
            0.0,
            1.0
        )

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_observation(self, task):

        observation = []

        # --------------------------------------------------------
        # Task-level information
        # --------------------------------------------------------

        task_info = task.iloc[0]

        log_cpu_demand = (
            self.normalize_cpu_demand(
                task_info["log_cpu_demand"]
            )
        )

        log_mem_demand = (
            self.normalize_memory_demand(
                task_info["log_mem_demand"]
            )
        )

        cpu_avg = (
            self.normalize_cpu_avg(
                task_info["cpu_avg"]
            )
        )

        mem_avg = (
            self.normalize_mem_avg(
                task_info["mem_avg"]
            )
        )

        duration = (
            self.normalize_duration(
                task_info["task_duration"]
            )
        )

        observation.extend(
            [
                float(log_cpu_demand),
                float(log_mem_demand),
                float(cpu_avg),
                float(mem_avg),
                float(duration)
            ]
        )

        # --------------------------------------------------------
        # Candidate machine information
        # --------------------------------------------------------

        for _, machine in task.iterrows():

            cpu_pressure = np.clip(
                float(machine["cpu_pressure"]),
                0.0,
                1.0
            )

            mem_pressure = np.clip(
                float(machine["mem_pressure"]),
                0.0,
                1.0
            )

            bottleneck_pressure = np.clip(
                float(machine["bottleneck_pressure"]),
                0.0,
                1.0
            )

            # ----------------------------------------------------
            # Estimated pressure
            #
            # CPU pressure is extremely heavy-tailed.
            # Dataset 95th percentile ≈ 12.83.
            #
            # Values above 12.83 are treated as severe pressure.
            # ----------------------------------------------------

            estimated_cpu_pressure = np.clip(
                float(
                    machine[
                        "estimated_cpu_pressure"
                    ]
                ) / 12.83,
                0.0,
                1.0
            )

            estimated_mem_pressure = np.clip(
                float(
                    machine[
                        "estimated_mem_pressure"
                    ]
                ),
                0.0,
                1.0
            )

            observation.extend(
                [
                    float(cpu_pressure),
                    float(mem_pressure),
                    float(bottleneck_pressure),
                    float(estimated_cpu_pressure),
                    float(estimated_mem_pressure)
                ]
            )

        # --------------------------------------------------------
        # Convert to float32
        # --------------------------------------------------------

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if observation.shape != (30,):

            raise ValueError(
                f"Invalid observation shape: "
                f"{observation.shape}"
            )

        observation = np.clip(
            observation,
            0.0,
            1.0
        ).astype(
            np.float32
        )

        return observation

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self.current_task_index = 0

        self.last_action = None
        self.last_reward = None

        self.current_task = (
            self.task_groups[
                self.current_task_index
            ]
        )

        observation = self._get_observation(
            self.current_task
        )

        info = {
            "task_index":
                self.current_task_index,

            "instance_name":
                self.current_task.iloc[0][
                    "instance_name"
                ],

            "event_time":
                int(
                    self.current_task.iloc[0][
                        "event_time"
                    ]
                )
        }

        return observation, info

    # ============================================================
    # REWARD FUNCTION
    # ============================================================

    def _calculate_reward(
        self,
        task,
        action
    ):

        selected_machine = task.iloc[action]

        # --------------------------------------------------------
        # Actual current pressure
        # --------------------------------------------------------

        cpu_pressure = float(
            selected_machine[
                "cpu_pressure"
            ]
        )

        mem_pressure = float(
            selected_machine[
                "mem_pressure"
            ]
        )

        actual_bottleneck = max(
            cpu_pressure,
            mem_pressure
        )

        # --------------------------------------------------------
        # Estimated post-placement pressure
        # --------------------------------------------------------

        estimated_cpu = float(
            selected_machine[
                "estimated_cpu_pressure"
            ]
        )

        estimated_mem = float(
            selected_machine[
                "estimated_mem_pressure"
            ]
        )

        normalized_estimated_cpu = np.clip(
            estimated_cpu / 12.83,
            0.0,
            1.0
        )

        normalized_estimated_mem = np.clip(
            estimated_mem,
            0.0,
            1.0
        )

        estimated_bottleneck = max(
            normalized_estimated_cpu,
            normalized_estimated_mem
        )

        # --------------------------------------------------------
        # Duration
        # --------------------------------------------------------

        duration = float(
            selected_machine[
                "task_duration"
            ]
        )

        normalized_duration = np.clip(
            duration / 599.0,
            0.0,
            1.0
        )

        # --------------------------------------------------------
        # Resource demand
        # --------------------------------------------------------

        cpu_demand = float(
            selected_machine[
                "cpu_demand_ratio"
            ]
        )

        mem_demand = float(
            selected_machine[
                "mem_demand_ratio"
            ]
        )

        cpu_demand_penalty = np.clip(
            np.log1p(cpu_demand)
            / np.log1p(828.0),
            0.0,
            1.0
        )

        mem_demand_penalty = np.clip(
            np.log1p(mem_demand)
            / np.log1p(1.0),
            0.0,
            1.0
        )

        # --------------------------------------------------------
        # Main reward
        #
        # Lower pressure is better.
        # Lower predicted pressure is better.
        # Shorter tasks are preferred.
        # Lower resource demand is preferred.
        # --------------------------------------------------------

        reward = 0.0

        reward -= (
            4.0 *
            actual_bottleneck
        )

        reward -= (
            2.0 *
            estimated_bottleneck
        )

        reward -= (
            1.0 *
            normalized_duration
        )

        reward -= (
            0.5 *
            cpu_demand_penalty
        )

        reward -= (
            0.25 *
            mem_demand_penalty
        )

        # --------------------------------------------------------
        # Overload penalties
        # --------------------------------------------------------

        if actual_bottleneck >= 0.90:

            reward -= 1.0

        if actual_bottleneck >= 0.95:

            reward -= 2.0

        if actual_bottleneck >= 0.98:

            reward -= 3.0

        if estimated_bottleneck >= 1.0:

            reward -= 3.0

        # --------------------------------------------------------
        # Safe-pressure bonus
        # --------------------------------------------------------

        if actual_bottleneck < 0.80:

            reward += 1.0

        elif actual_bottleneck < 0.90:

            reward += 0.5

        return float(reward)

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action):

        action = int(action)

        if not self.action_space.contains(
            action
        ):

            raise ValueError(
                f"Invalid action: {action}"
            )

        # --------------------------------------------------------
        # Calculate reward
        # --------------------------------------------------------

        reward = self._calculate_reward(
            self.current_task,
            action
        )

        # --------------------------------------------------------
        # Selected machine
        # --------------------------------------------------------

        selected_machine = (
            self.current_task.iloc[action]
        )

        # --------------------------------------------------------
        # Save state
        # --------------------------------------------------------

        self.last_action = action

        self.last_reward = reward

        # --------------------------------------------------------
        # Move to next task
        # --------------------------------------------------------

        self.current_task_index += 1

        terminated = (
            self.current_task_index
            >= self.num_tasks
        )

        truncated = False

        # --------------------------------------------------------
        # Next observation
        # --------------------------------------------------------

        if terminated:

            observation = np.zeros(
                30,
                dtype=np.float32
            )

            info = {
                "task_index":
                    self.current_task_index,

                "instance_name":
                    selected_machine[
                        "instance_name"
                    ],

                "event_time":
                    int(
                        selected_machine[
                            "event_time"
                        ]
                    ),

                "selected_machine":
                    selected_machine[
                        "machine_id"
                    ]
            }

        else:

            self.current_task = (
                self.task_groups[
                    self.current_task_index
                ]
            )

            observation = (
                self._get_observation(
                    self.current_task
                )
            )

            info = {
                "task_index":
                    self.current_task_index,

                "instance_name":
                    self.current_task.iloc[0][
                        "instance_name"
                    ],

                "event_time":
                    int(
                        self.current_task.iloc[0][
                            "event_time"
                        ]
                    ),

                "selected_machine":
                    selected_machine[
                        "machine_id"
                    ]
            }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ============================================================
    # RENDER
    # ============================================================

    def render(self):

        if self.current_task is None:

            return

        print(
            "Task:",
            self.current_task_index
        )

        print(
            "Last action:",
            self.last_action
        )

        print(
            "Last reward:",
            self.last_reward
        )


# ================================================================
# ENVIRONMENT TEST
# ================================================================

if __name__ == "__main__":

    print(
        "==================================="
    )

    print(
        "Testing Dynamic CPU Scheduling Env"
    )

    print(
        "==================================="
    )

    env = DynamicCPUSchedulingEnv(
        candidate_file=
        "task_machine_candidates_v2.csv"
    )

    print(
        "\nEnvironment tasks:",
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

    print(
        "Observation size:",
        env.observation_space.shape
    )

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------

    observation, info = env.reset(
        seed=42
    )

    print(
        "\nInitial observation shape:",
        observation.shape
    )

    print(
        "Initial observation dtype:",
        observation.dtype
    )

    print(
        "Initial observation min:",
        observation.min()
    )

    print(
        "Initial observation max:",
        observation.max()
    )

    print(
        "Initial info:",
        info
    )

    # ------------------------------------------------------------
    # Random actions
    # ------------------------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "Testing random actions"
    )

    print(
        "==================================="
    )

    for step_number in range(5):

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        print(
            f"\nStep {step_number + 1}"
        )

        print(
            "Action:",
            action
        )

        print(
            "Reward:",
            round(reward, 4)
        )

        print(
            "Observation shape:",
            observation.shape
        )

        print(
            "Terminated:",
            terminated
        )

        print(
            "Truncated:",
            truncated
        )

        print(
            "Selected machine:",
            info.get(
                "selected_machine",
                "N/A"
            )
        )

    # ------------------------------------------------------------
    # Gymnasium validation
    # ------------------------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "Environment validation"
    )

    print(
        "==================================="
    )

    try:

        from gymnasium.utils.env_checker import (
            check_env
        )

        check_env(
            DynamicCPUSchedulingEnv(
                candidate_file=
                "task_machine_candidates_v2.csv"
            )
        )

        print(
            "\nGymnasium environment check PASSED."
        )

    except Exception as e:

        print(
            "\nGymnasium environment check FAILED."
        )

        print(
            type(e).__name__,
            ":",
            e
        )

    env.close()

    print(
        "\n==================================="
    )

    print(
        "Environment test complete"
    )

    print(
        "==================================="
    )