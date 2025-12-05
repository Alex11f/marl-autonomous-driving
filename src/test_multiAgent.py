import gymnasium
import highway_env
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

class MultiAgentWrapper(gymnasium.Wrapper):
    """
    Wrapper per gestire più veicoli controllati indipendentemente.
    Accetta tuple di azioni e le applica ai veicoli controllati.
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = len(env.unwrapped.controlled_vehicles)
        
    def step(self, actions):
        """
        actions: tuple/list di azioni, una per ogni veicolo controllato
        """
        if not isinstance(actions, (tuple, list)):
            actions = [actions] * self.n_agents
        
        # Applica le azioni ai veicoli controllati manualmente
        for i, (vehicle, action) in enumerate(zip(self.env.unwrapped.controlled_vehicles, actions)):
            vehicle.act(self.env.unwrapped.action_type.actions[int(action)])
        
        # Simula l'ambiente
        self.env.unwrapped.road.act()
        self.env.unwrapped.road.step(1 / self.env.unwrapped.config["simulation_frequency"])
        
        # Ottieni osservazioni, reward, ecc.
        obs = self.env.unwrapped.observation_type.observe()
        info = self.env.unwrapped._info(obs, action=actions)
        reward = self.env.unwrapped._reward(actions[0])  # Usa il reward del primo agente
        terminated = self.env.unwrapped._is_terminated()
        truncated = self.env.unwrapped._is_truncated()
        
        if self.render_mode == 'human':
            self.render()
            
        return obs, reward, terminated, truncated, info

print("=== Setup Multi-Agente (Stesso Ambiente) ===\n")

# Note: highway-env with controlled_vehicles > 1 doesn't create a true multi-agent environment
# The observation is still a single array (vehicles_count x features)
# For proper multi-agent RL, you need separate models per agent or use a multi-agent framework

# --- Crea ambiente multi-agente ---
base_env = gymnasium.make(
    "highway-v0",
    render_mode="human",
    config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "flatten": False,  # Keep 2D: (5 vehicles, 5 features)
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "controlled_vehicles": 2,  # DUE veicoli controllati
        "vehicles_count": 10,
        "duration": 200,
        "lanes_count": 4,
        "initial_lane_id": None,  # Random lane
        "screen_width": 1200,  # Schermo più largo per vedere meglio
        "screen_height": 300,
        "centering_position": [0.3, 0.5],  # Centra la visuale
        "scaling": 5.5,
        "render_agent": True,  # Evidenzia i veicoli controllati
    }
)

# Avvolgi l'ambiente con il wrapper multi-agente
env = MultiAgentWrapper(base_env)

# --- Reset to check actual observation shape ---
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Numero di veicoli controllati: {len(env.unwrapped.controlled_vehicles)}\n")

# --- Crea DUE modelli DQN separati (uno per ogni agente) ---
print("Creando due modelli DQN separati...\n")
model_1 = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=1e-3,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=0)

model_2 = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=1e-3,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=0)

print("Training for one episode...")
# A training episode
obs, info = env.reset()
done = truncated = False
step_count = 0
while not (done or truncated) and step_count < 1000:
  # Ogni agente ottiene la sua osservazione e decide l'azione
  # obs ha shape (2, 5, 5) per 2 agenti, ciascuno vede 5 veicoli con 5 features
  obs_agent_1 = obs[0] if len(obs.shape) == 3 else obs  # Prima osservazione
  obs_agent_2 = obs[1] if len(obs.shape) == 3 else obs  # Seconda osservazione
  
  # Ottieni azioni dai due modelli separati
  action_1, _ = model_1.predict(obs_agent_1, deterministic=False)
  action_2, _ = model_2.predict(obs_agent_2, deterministic=False)
  
  # Combina le azioni in una tupla per l'ambiente
  action = (int(action_1), int(action_2))
  
  # Execute the action
  next_obs, reward, done, truncated, info = env.step(action)
  
  # Estrai le osservazioni successive per ogni agente
  next_obs_agent_1 = next_obs[0] if len(next_obs.shape) == 3 else next_obs
  next_obs_agent_2 = next_obs[1] if len(next_obs.shape) == 3 else next_obs
  
  # Store transitions in replay buffers
  model_1.replay_buffer.add(obs_agent_1, next_obs_agent_1, action_1, reward, done, [info])
  model_2.replay_buffer.add(obs_agent_2, next_obs_agent_2, action_2, reward, done, [info])
  
  obs = next_obs
  step_count += 1
  
  # Train both models
  if model_1.num_timesteps > model_1.learning_starts:
    model_1.train(gradient_steps=model_1.gradient_steps, batch_size=model_1.batch_size)
  if model_2.num_timesteps > model_2.learning_starts:
    model_2.train(gradient_steps=model_2.gradient_steps, batch_size=model_2.batch_size)

print(f"Episode finished after {step_count} steps")




