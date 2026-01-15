import os
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from metadrive.envs.metadrive_env import MetaDriveEnv

# Configurazione
TOTAL_TIMESTEPS = 500_000
SAVE_FREQ = 250_000
TRAIN = False
TEST = True
MODEL_PATH = "results/metadrive_single_ppo/ppo_single_final"
LOG_DIR = "results/metadrive_single_ppo/"
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    # Configurazione Ambiente Single Agent
    # map: int indica il numero di blocchi procedurali (es. 5 = mappa complessa con curve/incroci)
    # environment_num: quante mappe diverse generare (seed diversi)
    env_config = {
        "use_render": False,
        "traffic_density": 0.1, # Densità traffico NPC
        "map": 5, 
        "num_scenarios": 100,
        "start_seed": 5000,
        "vehicle_config": {
            "lidar": {"num_lasers": 72, "distance": 40, "num_others": 4},
            "show_lidar": True,
            "random_color": False
        },
        "random_lane_width": True,
        "random_agent_model": False,
    }

    # --- TRAINING ---
    if TRAIN:
        print(f"--- Inizio Training Single Agent ({TOTAL_TIMESTEPS} steps) ---")
        
        # Wrapper DummyVecEnv per SB3 (necessario per gestire correttamente reset e vettorizzazione base)
        def make_env():
            return MetaDriveEnv(env_config)
            
        env = DummyVecEnv([make_env])
        
        if os.path.exists(MODEL_PATH + ".zip"):
            print(f"Caricamento modello esistente da {MODEL_PATH}...")
            model = PPO.load(MODEL_PATH, env=env)
        else:
            print("Creazione nuovo modello PPO Single Agent...")
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=LOG_DIR,
                learning_rate=3e-4,
            )

        checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=LOG_DIR, name_prefix="ppo_single")
        
        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
            model.save(MODEL_PATH)
            print("Training completato.")
        except KeyboardInterrupt:
            print("Training interrotto. Salvataggio...")
            model.save(os.path.join(LOG_DIR, "ppo_single_interrupted"))
        
        env.close()

    # --- TEST ---
    if TEST:
        print("--- Inizio Test Single Agent ---")
        test_config = env_config.copy()
        test_config["use_render"] = False # Usiamo il render manuale cv2
        
        # Qui usiamo l'ambiente diretto senza DummyVecEnv per avere controllo sul loop di render
        env = MetaDriveEnv(test_config)
        
        if os.path.exists(MODEL_PATH + ".zip"):
            model = PPO.load(MODEL_PATH, env=env)
        else:
            print("Nessun modello trovato, uso pesi casuali.")
            model = PPO("MlpPolicy", env)

        obs, _ = env.reset()
              
        # Metriche
        episodes = 0
        successes = 0
        collisions = 0
        out_of_roads = 0
        rewards = []
        current_reward = 0

        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                
                # Gymnasium return: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                current_reward += reward
                
                # Render Top Down
                img = env.render(mode="top_down", film_size=(800, 800))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Info Stats
                speed_kmh = env.vehicle.speed_km_h
                
                # Calcolo metriche real-time
                sr = (successes / episodes) * 100 if episodes > 0 else 0.0
                cr = (collisions / episodes) * 100 if episodes > 0 else 0.0
                oor = (out_of_roads / episodes) * 100 if episodes > 0 else 0.0
                
                # Overlay Info
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(img, (10, 10), (350, 160), (0, 0, 0), -1)
                cv2.putText(img, f"Speed: {speed_kmh:.1f} km/h", (20, 40), font, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Episodes: {episodes}", (20, 70), font, 0.6, (255, 255, 255), 1)
                cv2.putText(img, f"Success Rate: {sr:.1f}%", (20, 95), font, 0.6, (255, 255, 0), 1)
                cv2.putText(img, f"Collision Rate: {cr:.1f}%", (20, 120), font, 0.6, (0, 0, 255), 1)
                cv2.putText(img, f"Out of Road: {oor:.1f}%", (20, 145), font, 0.6, (0, 165, 255), 1)
                
                cv2.imshow("Single Agent Test", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if done:
                    episodes += 1
                    rewards.append(current_reward)
                    
                    if info.get("arrive_dest", False):
                        successes += 1
                        print(f"Episodio {episodes}: Successo! Reward: {current_reward:.2f}")
                    elif info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False):
                        collisions += 1
                        print(f"Episodio {episodes}: Collisione. Reward: {current_reward:.2f}")
                    elif info.get("out_of_road", False):
                        out_of_roads += 1
                        print(f"Episodio {episodes}: Fuori strada. Reward: {current_reward:.2f}")
                    else:
                        print(f"Episodio {episodes}: Terminato (Time limit?). Reward: {current_reward:.2f}")
                        
                    current_reward = 0
                    obs, _ = env.reset()
                    
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            env.close()
            
            # Print Final Stats
            if episodes > 0:
                print("\n--- Risultati Finali Test ---")
                print(f"Episodi Totali: {episodes}")
                print(f"Success Rate: {(successes/episodes)*100:.2f}%")
                print(f"Collision Rate: {(collisions/episodes)*100:.2f}%")
                print(f"Out of Road Rate: {(out_of_roads/episodes)*100:.2f}%")
                print(f"Average Reward: {sum(rewards)/episodes:.2f}")

if __name__ == "__main__":
    main()
