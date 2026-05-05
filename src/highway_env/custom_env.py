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
        # Override to force the lane change using graph indices,
        # bypassing the geometric checks of next_lane that sometimes fail.
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
            # Fix: Prevent the car from going in reverse (target_speed < 0)
            # If the agent spam "SLOWER", the target speed should not go below 0
            # NOT TO BE USED IN ENVIROMENTS LIKE PARKING
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
    Hybrid environment that alternates Highway -> Merge -> Roundabout CONTINUOUSLY.
    No interruption between scenarios.
    Now supports custom configuration for multi-agent.  
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    
    def __init__(self, render_mode=None, steps_per_scenario=None, config=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.user_config = config if config is not None else {}
        self.scenario_configs = [
            ("highway-fast-v0", {}),
            ("merge-v0", {"duration": 100}),
            ("roundabout-v0", {"duration": 100}) # For now it doesn't work well, the episode doesn't end if you exit the roundabout
        ]
        
        self.current_scenario_idx = 0
        self.step_counter = 0
        
        self.env = None
        self.steps_per_scenario = steps_per_scenario if steps_per_scenario is not None else 100
        self._create_current_env()
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def _create_current_env(self):
        """Close the old environment and create a new clean one, applying the user configuration."""
        if self.env is not None:
            self.env.close()
            time.sleep(0.1)  # Give Pygame time to close
            
        env_id, base_config = self.scenario_configs[self.current_scenario_idx]
        
        full_config = base_config.copy()
        full_config.update({
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
        })
        # Apply the configuration passed by the user (e.g. for multi-agent)
        full_config.update(self.user_config)
        
        self.env = gymnasium.make(env_id, render_mode=self.render_mode, config=full_config)
        
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.step_counter += 1
        
        # If the car is out of the road in roundabout, force the scenario change
        if self.current_scenario_idx == 2:  # If we are in roundabout
            # In multi-agent, obs is a tuple, take the observation of the first agent
            agent_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
            position = agent_obs[0, 0] if len(agent_obs.shape) > 1 else agent_obs[0]
            if position > 5:
                print("Car out of the roundabout! Scenario change.")
                self.step_counter = self.steps_per_scenario
    
        # Scenario change WITHOUT interrupting the episode
        if self.step_counter >= self.steps_per_scenario:
            self.step_counter = 0
            self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenario_configs)
            
            print(f"\n--- Scenario change to: {self.scenario_configs[self.current_scenario_idx][0]} ---")
            obs, info = self._create_current_env()
            # The episode CONTINUES, it doesn't end
            done = False
            truncated = False
            
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        # Total reset (only at the beginning of the main episode)
        self.current_scenario_idx = np.random.randint(len(self.scenario_configs))
        self.step_counter = 0
        print(f"--- Starting with scenario: {self.scenario_configs[self.current_scenario_idx][0]} ---")
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
    Wrapper to manage multiple independently controlled vehicles.
    Accepts action tuples and applies them to controlled vehicles.
    """
    def __init__(self, env):
        super().__init__(env)
        self.agents = []
        self.n_agents = 0
        self.dones = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Save references to vehicles at the beginning of the episode
        self.agents = self.env.unwrapped.controlled_vehicles[:]
        self.n_agents = len(self.agents)
        self.dones = set()
        
        # Provide initial positions in info so test code can record spawn locations
        for i, vehicle in enumerate(self.agents):
            info[f"agent_{i}_x"] = vehicle.position[0]
            info[f"agent_{i}_speed"] = vehicle.speed
        
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

        # 1. & 2. Delegate simulation (actions + physics) to the internal environment
        # This correctly handles MultiAgentAction and physical integration
        self.env.unwrapped._simulate(tuple(mapped_actions))
        
        # 3. Get and ALIGN observations
        # highway-env returns obs only for live vehicles
        raw_obs = self.env.unwrapped.observation_type.observe()
        current_vehicles = self.env.unwrapped.controlled_vehicles
        aligned_obs = []
        
        for vehicle in self.agents:
            if vehicle in current_vehicles:
                # The vehicle is alive, let's take its observation
                # Use the REAL index in controlled_vehicles to pick the correct observation
                real_idx = current_vehicles.index(vehicle)
                if isinstance(raw_obs, (list, tuple)):
                    aligned_obs.append(raw_obs[real_idx])
                else:
                    aligned_obs.append(raw_obs[real_idx])
            else:
                # The vehicle is dead/removed. Insert empty observation (zeros) to maintain the shape.
                # Use raw_obs[0] as template for the shape if available, otherwise fallback (5,5)
                if len(raw_obs) > 0:
                    aligned_obs.append(np.zeros_like(raw_obs[0]))
                else:
                    aligned_obs.append(np.zeros((5, 5), dtype=np.float32))

        if isinstance(raw_obs, tuple):
            obs = tuple(aligned_obs)
        else:
            obs = np.array(aligned_obs)

        # 4. Info, Reward and Termination
        info = self.env.unwrapped._info(obs, action=actions)
        truncated = self.env.unwrapped._is_truncated()
        rewards = []
        original_vehicle = self.env.unwrapped.vehicle
        agents_dones = []
        for i, vehicle in enumerate(self.agents):
            is_crashed = vehicle.crashed
            is_offroad = (self.env.unwrapped.config["offroad_terminal"] and not vehicle.on_road)
            
            # ALWAYS log position and speed (even for crashed agents)
            info[f"agent_{i}_speed"] = vehicle.speed
            info[f"agent_{i}_x"] = vehicle.position[0]
            info[f"agent_{i}_lane"] = vehicle.lane_index[2]
            if hasattr(vehicle, "target_lane_index"):
                info[f"agent_{i}_target_lane"] = vehicle.target_lane_index[2]
            
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

        self.env.unwrapped.vehicle = original_vehicle
        info["agents_dones"] = agents_dones

        terminated = all(agents_dones)
        
        if self.render_mode == 'human':
            self.render()
            
        return obs, rewards, terminated, truncated, info


class CustomHighwayEnv(HighwayEnv):
    """
    HighwayEnv with controlled ego vehicle spawning (multi-agent).
    """

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        ego_vehicle_type = RLVehicle
        
        # Configuration of positioning
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

        # 2) FAST VEHICLES (before ego < ego_pos)
        n_fast = 10
        for _ in range(n_fast):
            lane = self.np_random.choice(lanes)
            # Random position behind
            long_pos = self.np_random.uniform(0, max(10, ego_pos - 30))
            
            position = lane.position(long_pos, 0)
            # Check collisions
            if any(np.linalg.norm(v.position - position) < 10.0 for v in self.road.vehicles):
                continue
                
            heading = lane.heading_at(long_pos)
            speed = self.np_random.uniform(25, 30)
            
            v = other_vehicles_type(self.road, position=position, heading=heading, speed=speed)
            v.randomize_behavior()
            self.road.vehicles.append(v)

        # 3) OTHER VEHICLES (after ego)
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
    """Fake single-agent environment, used only to initialize DQN."""
    metadata = {"render_modes": []}

    def __init__(self, obs_space, act_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}
     