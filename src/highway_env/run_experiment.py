import os
import time
import numpy as np
import pandas as pd
import gymnasium
from dataclasses import dataclass, field
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

# Local imports
from custom_env import CustomHighwayEnv, ENV_CONFIG, MultiAgentWrapper, SingleAgentStubEnv
from utils import _epsilon

# --- CONFIGURATION MANAGEMENT ---

@dataclass
class ExperimentConfig:
    mode: str = "single"  # Options: "single", "multi-independent", "multi-shared"
    train: bool = True
    test: bool = True

    loading_steps: int = 0 # Step count of the model to load (0 for scratch)
    total_steps: int = 50000 # Target total steps (end of training / model to test)
    
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

def get_experiment_config() -> ExperimentConfig:
    """
    Returns the configuration for the experiment.
    This function allows loading parameters as a standalone part.
    """
    config = ExperimentConfig(
        mode="single",
        train=True,
        test=True,
        loading_steps=0,
        total_steps=50000,
        test_episodes=10,
        render_test=True,
        seed=None
    )
    
    # Ensure directories exist
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    return config

# ----------------------------

def get_env(config: ExperimentConfig, render_mode="rgb_array"):
    env_config = ENV_CONFIG.copy()
    env_config["duration"] = 75 
    
    if config.mode == "single":
        env_config["controlled_vehicles"] = 1
        # Override observation type for single agent for compatibility
        env_config["observation"]["type"] = "Kinematics" 
        env_config["action"]["type"] = "DiscreteMetaAction" 
        env = CustomHighwayEnv(render_mode=render_mode, config=env_config)
        return env, 1
    
    else:
        # Multi-agent
        env_config["controlled_vehicles"] = config.n_agents 
        # config is already set for MultiAgentObservation in ENV_CONFIG
        base_env = CustomHighwayEnv(render_mode=render_mode, config=env_config)
        env = MultiAgentWrapper(base_env)
        return env, config.n_agents

def load_model(config: ExperimentConfig, env, model_name: str, steps: int) -> DQN:
    """
    Attempts to load a model. Returns the loaded model or None if not found.
    """
    filename = f"{model_name}_{steps}.zip"
    path = os.path.join(config.checkpoint_dir, filename)
    
    if os.path.exists(path):
        print(f"Loading {model_name} from {path}...")
        try:
            return DQN.load(path, env=env)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    else:
        # print(f"Checkpoint {path} not found.")
        return None

def train(config: ExperimentConfig):
    env, n_agents = get_env(config)
    print(f"--- Starting Training: {config.mode} ---")
    
    # Setup models
    models = []
    model_names = []
    
    # Define model names and create stubs/envs
    if config.mode == "single":
        model_names = ["single_agent"]
        envs = [env]
    elif config.mode == "multi-independent":
        obs_space = env.observation_space.spaces[0]
        act_space = env.action_space.spaces[0]
        model_names = [f"multi_indep_agent_{i}" for i in range(n_agents)]
        envs = [SingleAgentStubEnv(obs_space, act_space) for _ in range(n_agents)]
    elif config.mode == "multi-shared":
        obs_space = env.observation_space.spaces[0]
        act_space = env.action_space.spaces[0]
        model_names = ["multi_shared_agent"]
        envs = [SingleAgentStubEnv(obs_space, act_space)] # Only one model for shared
    
    # Load or Create Models
    start_step = 0
    loaded_count = 0
    
    for i, name in enumerate(model_names):
        model_env = envs[i] if i < len(envs) else envs[0] # Handle shared case where len(envs)=1
        
        # Try loading if loading_steps > 0
        loaded_model = None
        if config.loading_steps > 0:
            loaded_model = load_model(config, model_env, name, config.loading_steps)
            
        if loaded_model:
            model = loaded_model
            loaded_count += 1
        else:
            if config.loading_steps > 0:
                print(f"Warning: Could not load {name} at step {config.loading_steps}. Initializing from scratch.")
            model = DQN(env=model_env, **config.dqn_params)
            
        models.append(model)

    if config.mode == "multi-shared":
        # Duplicate the reference for shared model
        models = [models[0]] * n_agents
        if loaded_count > 0:
             start_step = config.loading_steps # If shared model loaded, we start from there
    else:
        # For independent, if all loaded, set start_step
        if loaded_count == n_agents:
            start_step = config.loading_steps
        elif loaded_count > 0:
            print(f"Warning: Partial loading ({loaded_count}/{n_agents}). Starting from 0 to be safe/synced.")
            start_step = 0 # Or handle partial? Let's assume 0 for safety.
            
    if start_step > 0:
         print(f"Resuming training from step {start_step}.")

    # Training Loop
    obs, info = env.reset(seed=config.seed)
    
    current_step = 0 
    agents_dead = set()
    
    # Custom Logger
    sb3_logger = configure(folder=None, format_strings=[])
    unique_models = list(set(models))
    for m in unique_models:
        m.set_logger(sb3_logger)

    steps_to_train = config.total_steps - start_step
    if steps_to_train <= 0:
        print(f"Total steps {config.total_steps} already reached (start_step={start_step}). Skipping training.")
    else:
        print(f"Training for {steps_to_train} steps (from {start_step} to {config.total_steps})...")

        while current_step < steps_to_train:
            global_step = start_step + current_step
            
            # Get actions
            actions = []
            eps = _epsilon(global_step) if config.mode != "single" else max(0.1, 1.0 - global_step / (config.total_steps*2))

            if config.mode == "single":
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    action, _ = models[0].predict(obs, deterministic=True)
                action = int(action)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                models[0].replay_buffer.add(obs, next_obs, np.array([action]), reward, done, [info])
                
                if global_step > models[0].learning_starts:
                    models[0].train(gradient_steps=1, batch_size=32)
                
                obs = next_obs
                if done:
                    obs, info = env.reset()
                    
            else:
                # Multi-agent loop
                for i in range(n_agents):
                    if i in agents_dead:
                        actions.append(1) # Idle
                        continue
                        
                    obs_agent = obs[i].astype(np.float32)
                    if np.random.rand() < eps:
                        act = int(env.action_space.spaces[i].sample())
                    else:
                        act, _ = models[i].predict(obs_agent, deterministic=True)
                        act = int(act)
                    actions.append(act)
                
                action_tuple = tuple(actions)
                next_obs, rewards, terminated, truncated, info = env.step(action_tuple)
                done = terminated or truncated
                
                for i in range(n_agents):
                    if i in agents_dead:
                        continue
                    
                    agent_done = done or info.get("agents_dones", [False]*n_agents)[i]
                    
                    models[i].replay_buffer.add(
                        obs[i].astype(np.float32), 
                        next_obs[i].astype(np.float32), 
                        np.array([actions[i]]), 
                        rewards[i], 
                        agent_done, 
                        [info]
                    )
                    
                    if agent_done:
                        agents_dead.add(i)
                    
                    if global_step > models[i].learning_starts:
                         models[i].train(gradient_steps=1, batch_size=32)
                
                obs = next_obs
                if done:
                    obs, info = env.reset()
                    agents_dead = set()

            current_step += 1
            if current_step % 1000 == 0:
                print(f"Session Step {current_step}/{steps_to_train} (Global: {global_step})")
            
    # Save models
    if config.mode == "multi-shared":
        path = os.path.join(config.checkpoint_dir, f"{model_names[0]}_{config.total_steps}.zip")
        models[0].save(path)
        print(f"Saved shared model to {path} at step {config.total_steps}")
    else:
        for i, m in enumerate(models):
            name = model_names[i]
            path = os.path.join(config.checkpoint_dir, f"{name}_{config.total_steps}.zip")
            m.save(path)
            print(f"Saved model {name} to {path} at step {config.total_steps}")
    
    env.close()


def test(config: ExperimentConfig):
    env, n_agents = get_env(config, render_mode="human" if config.render_test else "rgb_array")
    print(f"--- Starting Testing: {config.mode} (Target Model: {config.total_steps} steps) ---")
    
    models = []
    
    # Define model names and create stubs/envs for testing
    if config.mode == "single":
        model_names = ["single_agent"]
        envs = [env]
    elif config.mode == "multi-independent":
        obs_space = env.observation_space.spaces[0]
        act_space = env.action_space.spaces[0]
        model_names = [f"multi_indep_agent_{i}" for i in range(n_agents)]
        envs = [SingleAgentStubEnv(obs_space, act_space) for _ in range(n_agents)]
    elif config.mode == "multi-shared":
        obs_space = env.observation_space.spaces[0]
        act_space = env.action_space.spaces[0]
        model_names = ["multi_shared_agent"]
        envs = [SingleAgentStubEnv(obs_space, act_space)]

    # Load Models
    for i, name in enumerate(model_names):
        model_env = envs[i] if i < len(envs) else envs[0]
        
        # Load using helper with TOTAL_STEPS (since we are testing final model usually)
        loaded_model = load_model(config, model_env, name, config.total_steps)
        
        if loaded_model:
            model = loaded_model
        else:
            print(f"Warning: Could not load {name} at step {config.total_steps}. Using random agent for testing.")
            model = DQN(env=model_env, **config.dqn_params)
            
        models.append(model)

    if config.mode == "multi-shared":
        models = [models[0]] * n_agents

    results = []

    for ep in range(config.test_episodes):
        seed_val = config.seed + ep if config.seed is not None else None
        obs, info = env.reset(seed=seed_val)
        done = False
        steps = 0
        
        ep_rewards = [0.0] * n_agents
        ep_speeds = [[] for _ in range(n_agents)]
        ep_distances = [0.0] * n_agents
        start_positions = [None] * n_agents
        
        agents_dead = set()
        crashed = [False] * n_agents
        
        # Snapshot variables for recording metrics at step 50
        snapshot_rewards = None
        snapshot_distances = None
        snapshot_crashed = None
        snapshot_ep_speeds = None

        while not done and steps < 75:
            actions = []
            
            if config.mode == "single":
                action, _ = models[0].predict(obs, deterministic=True)
                action = int(action)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                ep_rewards[0] += reward
                
                veh = env.unwrapped.vehicle
                if start_positions[0] is None:
                    start_positions[0] = veh.position[0]
                
                ep_speeds[0].append(veh.speed)
                ep_distances[0] = veh.position[0] - start_positions[0]
                
                if terminated or truncated:
                    if info.get("crashed", False) or (env.unwrapped.config["offroad_terminal"] and not veh.on_road):
                        crashed[0] = True
                
                obs = next_obs
                done = terminated or truncated
                
            else:
                for i in range(n_agents):
                    obs_agent = obs[i].astype(np.float32)
                    act, _ = models[i].predict(obs_agent, deterministic=True)
                    actions.append(int(act))
                
                action_tuple = tuple(actions)
                next_obs, rewards, terminated, truncated, info = env.step(action_tuple)
                
                for i in range(n_agents):
                    ep_rewards[i] += rewards[i]
                    
                    if i not in agents_dead:
                        speed = info.get(f"agent_{i}_speed", 0)
                        pos = info.get(f"agent_{i}_x", 0)
                        
                        ep_speeds[i].append(speed)
                        
                        if start_positions[i] is None:
                            start_positions[i] = pos
                        
                        ep_distances[i] = pos - start_positions[i]
                        
                        if info.get(f"agent_{i}_crashed", False):
                            crashed[i] = True
                            agents_dead.add(i)
                
                obs = next_obs
                done = terminated or truncated

            if config.render_test:
                env.render()
                time.sleep(0.05)
            
            steps += 1
            
            # Snapshot logic at step 50
            if steps == 50:
                snapshot_rewards = list(ep_rewards)
                snapshot_distances = list(ep_distances)
                snapshot_crashed = list(crashed)
                snapshot_ep_speeds = [list(s) for s in ep_speeds]
            
        if snapshot_rewards is None:
            snapshot_rewards = ep_rewards
            snapshot_distances = ep_distances
            snapshot_crashed = crashed
            snapshot_ep_speeds = ep_speeds
            
        # Post-episode data collection
        for i in range(n_agents):
            avg_speed = np.mean(snapshot_ep_speeds[i]) if snapshot_ep_speeds and snapshot_ep_speeds[i] else 0
            
            results.append({
                "episode": ep,
                "mode": config.mode,
                "agent_id": i,
                "reward": snapshot_rewards[i],
                "length": min(steps, 50) if snapshot_rewards is not None else steps,
                "crashed": snapshot_crashed[i],
                "avg_speed": avg_speed,
                "distance": snapshot_distances[i],
                "training_steps": config.total_steps
            })
        
        print(f"Episode {ep} finished. Steps: {steps}. Crashed: {crashed}. (Recorded stats at step {min(steps, 50)})")

    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(config.results_dir, f"results_{config.mode}_{config.total_steps}_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    env.close()

def main():
    config = get_experiment_config()
    
    if config.train:
        train(config)
    
    if config.test:
        test(config)

if __name__ == "__main__":
    main()