import gymnasium
import highway_env
from stable_baselines3 import DQN
from custom_env import HybridDrivingEnv

# ---- Config ----
custom_env = False
train = False
# ---- -----

if custom_env:
    env = HybridDrivingEnv(render_mode='rgb_array')
else:
  env = gymnasium.make("highway-fast-v0")

model_path = "agents/test_model"
try:
    print("Caricamento del modello esistente...")
    model = DQN.load(model_path, env=env)
    model.learning_rate = 1e-3
except FileNotFoundError:
    print("Modello non trovato. Creo un nuovo modello da zero...")
    model = DQN('MlpPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  learning_rate=1e-3,
                  buffer_size=15000,
                  learning_starts=200,
                  batch_size=32,
                  gamma=0.8,
                  train_freq=1,
                  gradient_steps=1,
                  target_update_interval=50,
                  verbose=1,
                  tensorboard_log="highway_dqn/")
    
if train:
    model.learn(total_timesteps=10000)

model.save("agents/test_model")
env.close()

total_steps = 0
max_steps = 500  # Step totali prima di terminare
reset_interval = 100  # Resetta l'ambiente ogni 100 step


if not custom_env:
  env = gymnasium.make("highway-fast-v0", 
                     render_mode='human',
                     config={"duration": 500})
else:
  env = HybridDrivingEnv(render_mode='human', steps_per_scenario=max_steps)

# Load and test saved model
model = DQN.load("agents/test_model", env=env)

# Loop principale
obs, info = env.reset()
episode_steps = 0

while total_steps < max_steps:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    
    total_steps += 1
    episode_steps += 1
    
    # Reset ogni reset_interval step oppure se l'episodio termina
    if episode_steps >= reset_interval or done or truncated:
        obs, info = env.reset()
        episode_steps = 0

print(f"\nTest completato! ({total_steps} step totali)")
env.close()