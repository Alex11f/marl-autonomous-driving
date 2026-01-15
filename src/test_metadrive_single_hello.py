import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
import cv2

def main():
    print("Inizializzazione MetaDrive Single-Agent...")

    env_config = {
        "use_render": False,
        "traffic_density": 0.1,
        "map": "S", # Simple map
        "num_scenarios": 1,
        "start_seed": 5000,
        "vehicle_config": {
            "lidar": {"num_lasers": 72, "distance": 40, "num_others": 4},
            "show_lidar": False,
            "random_color": False
        },
        "random_lane_width": True,
        "random_agent_model": False,
    }

    try:
        env = MetaDriveEnv(env_config)
        obs, info = env.reset()
        print("Ambiente Single-Agent creato!")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")

        print("Running 100 steps test...")
        for i in range(100):
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Simple render check (offscreen)
            img = env.render(mode="top_down", film_size=(600, 600))
            
            if terminated or truncated:
                print(f"Episode done at step {i+1}")
                obs, info = env.reset()

        print("Test Single-Agent completato con successo.")

    except Exception as e:
        print(f"Errore durante il test Single-Agent: {e}")
        raise e
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()
