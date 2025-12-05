import gymnasium as gym
import highway_env
import numpy as np

def main():
    print("Initializing highway-env...")
    # Create the environment
    env = gym.make("highway-v0", render_mode="rgb_array")
    
    # Reset the environment
    obs, info = env.reset()
    print("Environment initialized successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run a short simulation
    done = False
    truncated = False
    steps = 0
    max_steps = 10
    
    print(f"Running simulation for {max_steps} steps...")
    while not (done or truncated) and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    
    print("Simulation loop finished successfully.")
    env.close()

if __name__ == "__main__":
    main()
