import os
from dataclasses import dataclass, field

# ------------------------
# --- Plot information ---
# ------------------------
ENV_INFO = """Environment:
- Duration: 120s
- Vehicles: 20
- Density: 0.3
- Collision Reward: -1.5
- Speed Reward Range: [0, 35]"""

MODEL_INFO = """Model (DQN):
- Hidden Layers: [256, 256]
- Buffer Size: 50k
- Learning Rate: 1e-4
- Gamma: 0.95"""


# ----------------------------------
# --- Environment configurations ---
# ----------------------------------
ENV_CONFIG = {
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "flatten": False,
        },
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        },
    },
    "controlled_vehicles": 2,
    # Environment parameters - lane and traffic
    "lanes_count": 4,
    "vehicles_count": 20,
    "vehicles_density": 0.3,
    "offroad_terminal": True,
    "initial_spacing": 2000,
    # Agent parameters
    "ego_spacing": 50.0,
    "ego_initial_pos": 100.0,
    "initial_lane_id": None,
    # Rendering parameters
    "duration": 10,
    "screen_width": 1200,
    "screen_height": 300,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "render_agent": True,
    "simulation_frequency": 15, 
    # Reward parameters
    "reward_speed_range": [0, 35],
    "normalize_reward": False,
    # The following two are not used in the base env, so atm are commented out
    "collision_reward": -1.50,
    "right_lane_reward": 0.00,
}

# -----------------------------------------
# --- Model and training configurations ---
# -----------------------------------------
@dataclass
class ExperimentConfig:
    mode: str = ""  # Options: "single", "multi-independent", "multi-shared"
    train: bool = True
    test: bool = True

    loading_steps: int = 0 # Step count of the model to load (0 for scratch)
    total_steps: int = 50000 #Total training steps in the training the final value is 100000
    
    test_episodes: int = 10
    render_test: bool = True  # Set to True to watch the agent play
    seed: int = None
    n_agents: int = 2 
    
    results_dir: str = os.path.join("results", "highway_dqn_results")
    checkpoint_dir: str = os.path.join("results", "checkpoints", "highway_dqn")
    
    dqn_params: dict = field(default_factory=lambda: dict(
        policy='MlpPolicy',
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=0
    ))
