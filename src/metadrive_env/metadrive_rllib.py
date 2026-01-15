import os
import glob
import csv
import time
import shutil
import argparse
import datetime
import numpy as np
import pandas as pd
import cv2
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.stopper import Stopper
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from metadrive.envs.marl_envs import MultiAgentMetaDrive

# --- CONFIGURATION MANAGEMENT ---

@dataclass
class ExperimentConfig:
    # Experiment Mode
    mode: str = "multi"  # "single" or "multi"
    train: bool = True
    test: bool = True
    resume: bool = True # Resume from latest checkpoint if available
    
    # Environment Settings
    n_agents: int = 2
    map_config_type: str = "block_num" # "block_num" or "block_sequence"
    map_config_args: int = 3 # Number of blocks
    start_seed: int = 5000 # Fixed seed for map generation (Ensure same map)
    num_scenarios: int = 1 # Number of scenarios (1 = fixed map)
    traffic_density: float = 0.1 # Density of background traffic
    num_workers: int = 2 # Number of parallel environment runners
    
    # Training Settings
    total_steps: int = 500_000
    batch_size: int = 4096
    lr: float = 1e-4
    
    # Testing Settings
    test_episodes: int = 10
    render_test: bool = True
    
    # Paths
    experiment_name: str = "PPO_MetaDrive_Experiment"
    results_dir: str = os.path.abspath("results/rllib_metadrive")
    
    def __post_init__(self):
        if self.mode == "single":
            self.n_agents = 1
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)

def get_experiment_config() -> ExperimentConfig:
    # You can modify defaults here or use argument parsing
    return ExperimentConfig(
        mode="single", # Change to 'single' for single agent
        train=True,
        test=True,
        resume=True, # Resume from latest checkpoint if available
        n_agents=2,
        total_steps=500_000, # This defines ADDITIONAL steps. 500k prev + 500k new = 1M total.
        start_seed=5000,
        num_scenarios=50, # Train on multiple scenarios for generalization
        test_episodes=10, # Explicitly set test episodes
        traffic_density=0.1,
        num_workers=1 # Parallel workers
    )

# --- UTILS ---

class MaxIterationsStopper(Stopper):
    def __init__(self, max_iterations):
        self._max_iterations = max_iterations
        self._iterations = 0

    def __call__(self, trial_id, result):
        self._iterations += 1
        return self._iterations >= self._max_iterations

    def stop_all(self):
        return False

def get_latest_checkpoint(experiment_path, experiment_name):
    if not os.path.exists(experiment_path):
        return None
    
    search_pattern = os.path.join(experiment_path, f"{experiment_name}*", "**", "checkpoint_*")
    checkpoints = glob.glob(search_pattern, recursive=True)
    
    if not checkpoints:
        return None
        
    checkpoints = [ckpt for ckpt in checkpoints if os.path.isdir(ckpt)]
    if not checkpoints:
        return None

    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

# --- ENVIRONMENT WRAPPER ---

class RLLibMetaDriveWrapper(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        # Create a copy to avoid modifying the original config used by Ray
        config_copy = env_config.copy()
        
        # Clean up config for MetaDrive
        config_copy.pop("metadata", None)
        
        # Initialize MetaDrive MultiAgent Environment
        self.env = MultiAgentMetaDrive(config_copy)
        
        # RLlib requires these to be set
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._agent_ids = set(self.env.agents.keys())

    def reset(self, *, seed=None, options=None):
        # MetaDrive handles seeding via config, but we can pass seed if needed
        # Note: MetaDrive's reset typically doesn't take 'options'
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = self.env.step(action_dict)
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        return self.env.render(mode="top_down", film_size=(800, 800))
        
    def close(self):
        self.env.close()

# --- TRAIN & TEST FUNCTIONS ---

def train(config: ExperimentConfig):
    print(f"--- Starting Training: {config.mode} Agent(s) ---")
    print(f"Map Config: {config.num_scenarios} scenario(s) starting at seed {config.start_seed}")

    # Environment Config for MetaDrive
    env_config = dict(
        allow_respawn=True,
        num_agents=config.n_agents,
        crash_done=True,
        delay_done=0,
        use_render=False, # No render during training
        num_scenarios=config.num_scenarios, 
        start_seed=config.start_seed,
        traffic_density=config.traffic_density,
        map_config={
            "type": config.map_config_type,
            "config": config.map_config_args,
            "lane_num": 2
        },
        metadata={
            "experiment_name": config.experiment_name
        }
    )

    # Register Environment
    register_env("meta_drive_train", lambda c: RLLibMetaDriveWrapper(c))

    # Retrieve spaces for policy definition
    temp_env = RLLibMetaDriveWrapper(env_config)
    obs_space = temp_env.observation_space["agent0"]
    act_space = temp_env.action_space["agent0"]
    temp_env.close()

    # PPO Config
    algo_config = (
        PPOConfig()
        .environment(env="meta_drive_train", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=0) # Set to 1 if you have a GPU available
        .env_runners(num_env_runners=config.num_workers) # Parallel workers
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {})
            },
            # Map all agents to the same shared policy
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        .training(
            train_batch_size=config.batch_size,
            lr=config.lr,
            grad_clip=0.5,
            model={"fcnet_hiddens": [256, 256]} 
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )
    
    # Set PPO specific parameters directly to avoid version mismatch issues
    algo_config.sgd_minibatch_size = 256
    algo_config.num_sgd_iter = 10 # num_epochs
    algo_config.clip_param = 0.2
    algo_config.kl_coeff = 0.2
    algo_config.lambda_ = 0.95

    # Calculate iterations
    iterations = int(config.total_steps / config.batch_size)
    stopper = MaxIterationsStopper(max_iterations=iterations)

    # Resume Logic
    restore_path = None
    run_name = config.experiment_name
    
    if config.resume:
        latest_ckpt = get_latest_checkpoint(config.results_dir, config.experiment_name)
        if latest_ckpt:
            print(f"Resuming from checkpoint: {latest_ckpt}")
            restore_path = latest_ckpt
            run_name = f"{config.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            print("No checkpoint found. Starting fresh.")

    print(f"Training for {iterations} iterations (approx {config.total_steps} steps).")
    
    tune.run(
        "PPO",
        name=run_name,
        config=algo_config.to_dict(),
        stop=stopper,
        checkpoint_freq=5, # Save every 5 iterations
        storage_path=config.results_dir,
        verbose=1,
        restore=restore_path,
        resume=False
    )
    print("Training Finished.")


def test(config: ExperimentConfig):
    print(f"--- Starting Testing: {config.mode} Agent(s) ---")
    print(f"Test Episodes: {config.test_episodes}")
    
    # Environment Config (Same as training but with render capability if needed)
    env_config = dict(
        allow_respawn=True,
        num_agents=config.n_agents,
        crash_done=True,
        delay_done=0,
        use_render=False, # We use manual cv2 render
        num_scenarios=config.test_episodes,
        start_seed=config.start_seed,
        traffic_density=config.traffic_density,
        map_config={
            "type": config.map_config_type,
            "config": config.map_config_args,
            "lane_num": 2
        }
    )
    
    # Build Algorithm to load weights
    # We re-create the config structure to match training
    temp_env = RLLibMetaDriveWrapper(env_config)
    obs_space = temp_env.observation_space["agent0"]
    act_space = temp_env.action_space["agent0"]
    temp_env.close()

    algo_config = (
        PPOConfig()
        .environment(env="meta_drive_test", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=0)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda x, *args, **kwargs: "shared_policy"
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )
    
    register_env("meta_drive_test", lambda c: RLLibMetaDriveWrapper(c))
    algo = algo_config.build()
    
    # Load Checkpoint
    checkpoint_to_load = get_latest_checkpoint(config.results_dir, config.experiment_name)
    if checkpoint_to_load:
        print(f"Restoring checkpoint: {checkpoint_to_load}")
        algo.restore(checkpoint_to_load)
    else:
        print("Warning: No checkpoint found! Testing with random weights.")

    # Run Episodes
    env = RLLibMetaDriveWrapper(env_config)
    
    results = []
    
    try:
        for ep in range(config.test_episodes):
            print(f"Starting Episode {ep+1}/{config.test_episodes}")
            obs, info = env.reset(seed=config.start_seed + ep) # Optionally vary seed for testing
            done = False
            steps = 0
            
            ep_rewards = {a_id: 0.0 for a_id in obs.keys()}
            
            while not done:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    actions[agent_id] = algo.compute_single_action(
                        agent_obs, 
                        policy_id="shared_policy", 
                        explore=False
                    )
                
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                
                for agent_id, r in rewards.items():
                    ep_rewards[agent_id] = ep_rewards.get(agent_id, 0) + r
                
                if config.render_test:
                    img = env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Add text overlay
                    cv2.putText(img, f"Ep: {ep} Step: {steps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imshow("MetaDrive Test", img)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("Test Interrupted by User")
                        env.close()
                        cv2.destroyAllWindows()
                        return

                done = terminateds["__all__"] or truncateds["__all__"]
                steps += 1
            
            # Record Results
            avg_reward = sum(ep_rewards.values()) / len(ep_rewards) if ep_rewards else 0
            print(f"Episode {ep} finished. Steps: {steps}. Avg Reward: {avg_reward:.2f}")
            
            for agent_id, reward in ep_rewards.items():
                results.append({
                    "episode": ep,
                    "agent_id": agent_id,
                    "reward": reward,
                    "steps": steps
                })
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
            
    env.close()
    if config.render_test:
        cv2.destroyAllWindows()
        # Save to CSV
        if results:
            csv_path = os.path.join(config.results_dir, f"test_results_{int(time.time())}.csv")
            pd.DataFrame(results).to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

def main():
    # Optimization for Ray memory
    os.environ["RAY_memory_usage_threshold"] = "0.99"
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    config = get_experiment_config()
    
    if config.train:
        train(config)
    
    if config.test:
        test(config)
    
    ray.shutdown()

if __name__ == "__main__":
    main()