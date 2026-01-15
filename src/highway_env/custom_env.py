import gymnasium
import numpy as np
import time

from highway_env.envs.highway_env import HighwayEnv
from highway_env.utils import near_split
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle

class RLVehicle(ControlledVehicle):
    def act(self, action: dict | str = None) -> None:
        # Override per forzare il cambio corsia basandosi sugli indici del grafo,
        # bypassando i controlli geometrici di next_lane che a volte falliscono.
        if action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target = _id + 1
            if _from in self.road.network.graph and _to in self.road.network.graph[_from]:
                lanes_count = len(self.road.network.graph[_from][_to])
                if target < lanes_count:
                    self.target_lane_index = (_from, _to, target)
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target = _id - 1
            if target >= 0:
                self.target_lane_index = (_from, _to, target)
        else:
            super().act(action)
            # Fix: Impediamo che l'auto vada in retromarcia (target_speed < 0)
            # Se l'agente spamma "SLOWER", la velocità target non deve scendere sotto 0
            # DA NON UTILIZZARE IN ENVIROMENT COME PARKING
            if self.target_speed < 0:
                self.target_speed = 0



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
    # Custom spawn parameters (ego vehicles)
    # "ego_spawn_spacing": 1.25,
    # "ego_spawn_min_gap": 5.0,
    # "ego_spawn_max_tries": 100,
    # "other_spawn_max_tries": 30,
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
    # "lane_change_reward": 0.01,
    # "high_speed_reward": 0.10,
    "collision_reward": -1.50,
    "right_lane_reward": 0.00,
}

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
            ("roundabout-v0", {"duration": 100}) # Al momento non funziona bene, non si interrompe l'episodio se si esce dalla rotonda
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


#--------------------------- MultiAgent Wrapper e CustomHighwayEnv ---------------------------#


class MultiAgentWrapper(gymnasium.Wrapper):
    """
    Wrapper per gestire più veicoli controllati indipendentemente.
    Accetta tuple di azioni e le applica ai veicoli controllati.
    """
    def __init__(self, env):
        super().__init__(env)
        self.agents = []
        self.n_agents = 0
        self.dones = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Salviamo i riferimenti ai veicoli all'inizio dell'episodio
        self.agents = self.env.unwrapped.controlled_vehicles[:]
        self.n_agents = len(self.agents)
        self.dones = set()
        return obs, info
        
    def step(self, actions):
        if not isinstance(actions, (tuple, list)):
            actions = [actions] * self.n_agents
        
        # --- FORCE RESTORE AGENTS ---
        if hasattr(self.env.unwrapped, "controlled_vehicles"):
             active_agents = [a for a in self.agents if a in self.env.unwrapped.road.vehicles]
             for a in active_agents:
                 a.is_controlled = True
             self.env.unwrapped.controlled_vehicles = active_agents

        current_vehicles = self.env.unwrapped.controlled_vehicles
        
        mapped_actions = []
        for v in current_vehicles:
            try:
                idx = self.agents.index(v)
                mapped_actions.append(actions[idx])
            except ValueError:
                print(f"WARNING: Agent {v} (ID: {id(v)}) not found in self.agents! Fallback to IDLE.")
                mapped_actions.append(1) # Fallback IDLE

        # 1. & 2. Delega la simulazione (azioni + fisica) all'ambiente interno
        # Questo gestisce correttamente MultiAgentAction e l'integrazione fisica
        self.env.unwrapped._simulate(tuple(mapped_actions))
        
        # 3. Ottieni e ALLINEA le osservazioni
        # highway-env restituisce obs solo per i veicoli vivi
        raw_obs = self.env.unwrapped.observation_type.observe()
        current_vehicles = self.env.unwrapped.controlled_vehicles
        aligned_obs = []
        
        for vehicle in self.agents:
            if vehicle in current_vehicles:
                # Il veicolo è vivo, prendiamo la sua osservazione
                # Usiamo l'indice REALE in controlled_vehicles per pescare l'osservazione corretta
                real_idx = current_vehicles.index(vehicle)
                if isinstance(raw_obs, (list, tuple)):
                    aligned_obs.append(raw_obs[real_idx])
                else:
                    aligned_obs.append(raw_obs[real_idx])
            else:
                # Il veicolo è morto/rimosso. Inseriamo osservazione vuota (zeri) per mantenere la shape.
                # Usiamo raw_obs[0] come template per la shape se disponibile, altrimenti fallback (5,5)
                if len(raw_obs) > 0:
                    aligned_obs.append(np.zeros_like(raw_obs[0]))
                else:
                    aligned_obs.append(np.zeros((5, 5), dtype=np.float32))

        if isinstance(raw_obs, tuple):
            obs = tuple(aligned_obs)
        else:
            obs = np.array(aligned_obs)

        # 4. Info, Reward e Termination
        info = self.env.unwrapped._info(obs, action=actions)
        truncated = self.env.unwrapped._is_truncated()
        rewards = []
        original_vehicle = self.env.unwrapped.vehicle
        agents_dones = []
        for i, vehicle in enumerate(self.agents):
            is_crashed = vehicle.crashed
            is_offroad = (self.env.unwrapped.config["offroad_terminal"] and not vehicle.on_road)
            
            if is_crashed or is_offroad:
                # Change color to red to indicate crash/death
                vehicle.color = (200, 0, 50)
                
                if i in self.dones:
                    rewards.append(0.0)
                else:
                    rewards.append(self.env.unwrapped.config["collision_reward"])
                    self.dones.add(i)
                agents_dones.append(True)
                
                if is_crashed:
                    info["crashed"] = True
                    info[f"agent_{i}_crashed"] = True
                
                if is_offroad:
                    info[f"agent_{i}_offroad"] = True
            else:
                self.env.unwrapped.vehicle = vehicle
                reward = self.env.unwrapped._reward(action=actions[i])
                rewards.append(reward)
                agents_dones.append(False)
                
                # Debug info per problemi nel cambio corsia
                info[f"agent_{i}_lane"] = vehicle.lane_index[2]
                # Se il veicolo ha un target lane (es. sta cambiando corsia), lo logghiamo
                if hasattr(vehicle, "target_lane_index"):
                    info[f"agent_{i}_target_lane"] = vehicle.target_lane_index[2]
                info[f"agent_{i}_speed"] = vehicle.speed
                info[f"agent_{i}_x"] = vehicle.position[0]

        self.env.unwrapped.vehicle = original_vehicle
        info["agents_dones"] = agents_dones

        terminated = all(agents_dones)
        
        if self.render_mode == 'human':
            self.render()
            
        return obs, rewards, terminated, truncated, info


class CustomHighwayEnv(HighwayEnv):
    """
    HighwayEnv con spawn controllato dei veicoli ego (multi-agent).
    """

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        ego_vehicle_type = RLVehicle
        
        # Configurazione posizionamento
        ego_pos = self.config.get("ego_initial_pos", 100.0)
        
        # 1) Spawn EGO vehicles
        self.controlled_vehicles = []
        lanes = self.road.network.lanes_list()
        for i in range(self.config["controlled_vehicles"]):
            lane_idx = i % len(lanes)
            lane = lanes[lane_idx]
            longitudinal_pos = ego_pos + i * self.config.get("ego_spacing", 10.0)
            position = lane.position(longitudinal_pos, 0)
            heading = lane.heading_at(longitudinal_pos)

            vehicle = ego_vehicle_type(
                self.road, position=position, heading=heading,
                speed=25.0 + self.np_random.uniform(-5, 5),
            )
            vehicle.color = (50, 200, 0)
            vehicle.is_controlled = True
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        # 2) VEICOLI VELOCI (prima degli ego < ego_pos)
        n_fast = 10
        for _ in range(n_fast):
            lane = self.np_random.choice(lanes)
            # Posizione casuale dietro
            long_pos = self.np_random.uniform(0, max(10, ego_pos - 30))
            
            position = lane.position(long_pos, 0)
            # Check collisioni
            if any(np.linalg.norm(v.position - position) < 10.0 for v in self.road.vehicles):
                continue
                
            heading = lane.heading_at(long_pos)
            speed = self.np_random.uniform(25, 30)
            
            v = other_vehicles_type(self.road, position=position, heading=heading, speed=speed)
            v.randomize_behavior()
            self.road.vehicles.append(v)

        # 3) ALTRI VEICOLI (dopo gli ego)
        remaining_vehicles = self.config["vehicles_count"] - len(self.controlled_vehicles) - n_fast
        
        for _ in range(remaining_vehicles):
            lane = self.np_random.choice(lanes)
            long_pos = self.np_random.uniform(ego_pos + 50, ego_pos + 400)
            
            position = lane.position(long_pos, 0)
            if any(np.linalg.norm(v.position - position) < 10.0 for v in self.road.vehicles):
                continue
                
            heading = lane.heading_at(long_pos)
            speed = self.np_random.uniform(15, 25)
            
            v = other_vehicles_type(self.road, position=position, heading=heading, speed=speed)
            v.randomize_behavior()
            self.road.vehicles.append(v)



class SingleAgentStubEnv(gymnasium.Env):
    """Env fittizio a singolo agente, usato solo per inizializzare DQN."""
    metadata = {"render_modes": []}

    def __init__(self, obs_space, act_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}
     