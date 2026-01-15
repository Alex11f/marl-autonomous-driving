import os
import gymnasium
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from custom_env import CustomHighwayEnv, ENV_CONFIG, MultiAgentWrapper, SingleAgentStubEnv


EPS_START = 1.0
EPS_END = 0.1
EPS_FRACTION = 0.75  # Decay over 75% of TOTAL_STEPS
TOTAL_STEPS = 75000

files_to_load = {
    # "Multi-independet 100k": "results_multi-independent_1768320846.csv",
    # "Single 100k": "results_single_1768319780.csv",
    "single 50k": "results_single_1768241107.csv",
    "Single 100k2": "results_single_1768343914.csv",
    "shared 50k": "results_multi-shared_1768337662.csv",
    # "shared 100k": "results_multi-shared_1768321130.csv",
    "shared 100k2": "results_multi-shared_100000_1768390911.csv",
    "independet 50k": "results_multi-independent_1768341042.csv",
    "independet 100k2": "results_multi-independent_1768347379.csv",
}


def _fmt_mean(values) -> str:
    if not values:
        return "N/A"
    return f"{float(np.mean(values)):.2f}"

def _to_scalar_reward(r) -> float:
    try:
        return float(r)
    except Exception:
        return float(np.mean(r))
    


# Training helper functions
def _epsilon(step: int) -> float:
    decay_steps = max(1, int(EPS_FRACTION * TOTAL_STEPS))
    if step >= decay_steps:
        return EPS_END
    t = step / decay_steps
    return EPS_START + t * (EPS_END - EPS_START)

def _select_action(model: DQN, env: gymnasium.Env, obs_arr: np.ndarray, eps: float, m_num: int) -> int:
    if np.random.rand() < eps:
        return int(env.action_space[m_num].sample())
    a, _ = model.predict(obs_arr, deterministic=True)
    return int(a)