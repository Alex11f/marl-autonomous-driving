import gymnasium
import numpy as np
import time

class HybridDrivingEnv(gymnasium.Env):
    """
    Ambiente ibrido che alterna Highway -> Merge -> Roundabout CONTINUAMENTE.
    Senza interruzioni tra i scenari.
    Ora supporta una configurazione personalizzata per il multi-agente.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    
    def __init__(self, render_mode=None, steps_per_scenario=None, config=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.user_config = config if config is not None else {}
        self.scenario_configs = [
            ("highway-fast-v0", {}),
            ("merge-v0", {"duration": 100}),
            ("roundabout-v0", {"duration": 100})
        ]
        
        self.current_scenario_idx = 0
        self.step_counter = 0
        
        self.env = None
        self.steps_per_scenario = steps_per_scenario if steps_per_scenario is not None else 100
        self._create_current_env()
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def _create_current_env(self):
        """Chiude il vecchio ambiente e ne crea uno nuovo pulito, applicando la configurazione utente."""
        if self.env is not None:
            self.env.close()
            time.sleep(0.1)  # Dai tempo a Pygame di chiudere
            
        env_id, base_config = self.scenario_configs[self.current_scenario_idx]
        
        full_config = base_config.copy()
        full_config.update({
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
        })
        # Applica la configurazione passata dall'utente (es. per multi-agente)
        full_config.update(self.user_config)
        
        self.env = gymnasium.make(env_id, render_mode=self.render_mode, config=full_config)
        
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.step_counter += 1
        
        # Se l'auto è fuori strada in roundabout, forza il cambio scenario
        if self.current_scenario_idx == 2:  # Se siamo in roundabout
            # In multi-agent, obs è una tupla, prendiamo l'osservazione del primo agente
            agent_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
            position = agent_obs[0, 0] if len(agent_obs.shape) > 1 else agent_obs[0]
            if position > 5:
                print("Auto fuori dalla rotonda! Cambio scenario.")
                self.step_counter = self.steps_per_scenario
    
        # Cambia scenario SENZA interrompere l'episodio
        if self.step_counter >= self.steps_per_scenario:
            self.step_counter = 0
            self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenario_configs)
            
            print(f"\n--- Cambio scenario a: {self.scenario_configs[self.current_scenario_idx][0]} ---")
            obs, info = self._create_current_env()
            # L'episodio CONTINUA, non finisce
            done = False
            truncated = False
            
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        # Reset totale (solo all'inizio dell'episodio principale)
        self.current_scenario_idx = np.random.randint(len(self.scenario_configs))
        self.step_counter = 0
        print(f"--- Inizio con scenario: {self.scenario_configs[self.current_scenario_idx][0]} ---")
        obs, info = self._create_current_env()
        return obs, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        if self.env:
            self.env.close()