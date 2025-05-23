from typing import Optional
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
import dill as pickle
from IPython.display import display, clear_output
import copy
from scipy.ndimage import zoom
import pandas as pd
from collections import defaultdict
import math

input = int(input("Enter 0 to train a new Q-table or 1 to evaluate an existing Q-table or 2 to train a model using a predefined Q-table: "))

N_EPISODES = 10000
N_AGENTS = 4
UPDATE_STEP = 1     # Update q_values after each step
BETA = 0.6
ALPHA = 0.1
GAMMA = 0.9
SIZE = 30
STEPS = SIZE * SIZE
EPSILON = 0.8
EVALUATION_STEPS = SIZE * SIZE
EVALUATION_EPISODES = 4
SEED = 643
NUMBER_OF_POIS = 4
BATTERY_THRESHOLD = 10

#----------------------------------- World--------------------------------------------------- #

class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 10): # The size of the square grid
        self.size = size
        self.trajectory = []
        self.world = np.zeros((size, size))
        self.visited_states = np.zeros((size, size))
        self.global_location = np.zeros((size, size))
        self.POI_world = np.zeros((size, size))
        self._poi_list = []
        self._agent_poi = []
        self.agents_location_list = [np.zeros(2, dtype=int) for _ in range(N_AGENTS)]


        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([1, 1], dtype=np.int32)
        self._reward_near = np.zeros(8)
        self._nearby_agents = np.zeros(8)
        self._POI_direction = np.zeros(8)
        self._world_border = (size, size)
        


        # Observations are dictionaries with the agent's and the target's location.
        self.observation_space = gym.spaces.Dict(
            {
            "reward_near": gym.spaces.Box(0, 10, shape=(8,), dtype=np.int32),
            "nearby_agents": gym.spaces.Box(0, 1, shape=(9,), dtype=int),
            "POI_direction": gym.spaces.Box(0, 1, shape=(8,), dtype=int),
            }
        )

        # We have 8 actions, corresponding to "right", "up", "left", "down" and diagonal
        self.action_space = gym.spaces.Discrete(8)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
            4: np.array([1, 1]),  # up-right
            5: np.array([-1, 1]),  # up-left
            6: np.array([-1, -1]),  # down-left
            7: np.array([1, -1]),  # down-right
        }
    
    def normalize_env_reward(self):
        # Normalize the world to be between 0 and 1
        min_val = 0
        max_val = 10
        self.world = (self.world - min_val) / (max_val - min_val)

    def scale_env(self):
        # Scale all rewards to be between 0 and 10
        min_val = np.min(self.world)
        max_val = np.max(self.world)
        self.world = (self.world - min_val) / (max_val - min_val) * 10
        self.world = np.round(self.world, 0)

    def fill_white_space(self):
        self.world[self.world == 0] = 0.3

    def state_penalize(self):
        x, y = self._agent_location
        if(self.world[x][y] > 0):
            self.world[x][y] -= self.world[x][y] + 2
        else:
            self.world[x][y] -= 2

    def random_env(self):
        self.world = np.random.randint(1, 11, size=(self.size, self.size))

    def heatmap_env(self):
        with open("../heatmap/data/plots/heatmap_file.pkl", "rb") as f:
            heatmap = np.rot90(pickle.load(f), k=1)
        
        heatmap_scaled = zoom(heatmap, (self.size / heatmap.shape[0], self.size / heatmap.shape[1]), order=1)  # Bilinear interpolation

        self.world = heatmap_scaled
        # self.normalize_env_reward()
        self.scale_env()
        threshold = self.make_threshold()
        print(threshold)
        self.auto_poi(threshold)
        # self.fill_white_space()

    def reset_visited_states(self):
        self.visited_states = np.zeros((self.size, self.size))
    
    def world_from_matrix(self, matrix):
        self.world = matrix

    def setReward(self, x, y , reward):
        self.world[x][y] = reward

    def _get_info(self):
        return {
            "agent_location": self._agent_location,
            "world_border": self._world_border,
        }

    def _get_obs(self):
        return {
            "reward_near": self._reward_near,
            "nearby_agent": self._nearby_agents,
            "POI_vector": self._POI_direction,
        }

    def split_poi_list(self, num_agents):
        split_list = [self._poi_list[i::num_agents] for i in range(num_agents)]
        self._agent_poi = split_list

    def reset_global_location(self):
        self.global_location = np.zeros((self.size, self.size))

    def reset(self, agents, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random

        # reset visited states near and reward_near
        self.reset_visited_states()
        self.reset_global_location()
        self.split_poi_list(num_agents=len(agents))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def spawn_agents_random(self, agents, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        for i, agent in enumerate(agents):
            # Generate two different random locations for agents
            if i % 2 == 0:
                agent.location = self.np_random.integers(0, self.size - 1, size=2, dtype=int)
                agent.spawn_location = agent.location
            else:
                agent.location = self.np_random.integers(0, self.size - 1, size=2, dtype=int)
                agent.spawn_location = agent.location

    def spawn_agents_uniform(self, agents, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        location = self.np_random.integers(0, self.size - 1, size=2, dtype=int)
        for agent in agents:
            agent.location = location
            agent.spawn_location = location
        

    def getReward(self):
        return self.world[self._agent_location[0], self._agent_location[1]]


    def check_rewards_near(self):
        x, y = self._agent_location
        self._reward_near = np.zeros(8)  # Reset to zeros
        max_reward = float('-inf')
        max_idx = -1

        # First pass: apply base logic and find max positive reward
        for idx, (dx, dy) in self._action_to_direction.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                reward = self.world[nx][ny]
                if reward > 0:
                    self._reward_near[idx] = 2
                    if reward > max_reward:
                        max_reward = reward
                        max_idx = idx
                elif reward == 0:
                    self._reward_near[idx] = 1
                else:
                    self._reward_near[idx] = 0
            else:
                self._reward_near[idx] = 0

        # Second pass: set highest reward direction to 3
        if max_idx != -1:
            self._reward_near[max_idx] = 3


    def update_global_location(self):
        self.global_location[self._agent_location[0], self._agent_location[1]] = 1

    def check_nearby_agents(self):
        agent_x, agent_y = self._agent_location
        locations = self.agents_location_list

        self._nearby_agents = np.zeros(8)  # Reset

        for location in locations:
            if np.array_equal(location, [agent_x, agent_y]):
                continue  # Skip self

            distance = np.linalg.norm(np.array(location) - np.array([agent_x, agent_y]))
            if distance <= 2:
                dx = location[0] - agent_x
                dy = location[1] - agent_y

                # Normalize direction to one of 8 compass directions
                if dx != 0:
                    dx = dx // abs(dx)
                if dy != 0:
                    dy = dy // abs(dy)

                direction_vector = np.array([dx, dy])
                for index, (ddx, ddy) in self._action_to_direction.items():
                    if np.array_equal(direction_vector, np.array([ddx, ddy])):
                        self._nearby_agents[index] = 1
                        break
        # print(np.sum(self._nearby_agents))



    def set_POI(self, x, y):
        self._poi_list.append((x, y))


    def make_threshold(self):
        total = 0
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                reward = self.world[i][j]
                if reward > 0:
                    total += reward
                    count += 1
        if count == 0:
            return 0.0  # avoid division by zero
        threshold = total/count
        return threshold

    def auto_poi(self, threshold, min_distance=6):
        # Assuming self.world is a 2D numpy array of shape (size, size)
        poi_candidates = []
        for i in range(1, self.size - 1):  # avoid edges
            for j in range(1, self.size - 1):
                square = self.world[i-1:i+2, j-1:j+2]  # 3x3 window
                reward_average = np.mean(square)
                
                if reward_average >= threshold:
                    poi_candidates.append(((i, j), reward_average))


        # Sort by average reward in descending order
        poi_candidates.sort(key=lambda x: x[1], reverse=True)

        selected_pois = []

        for (i, j), avg in poi_candidates:
            too_close = False
            for (pi, pj) in selected_pois:
                if abs(pi - i) + abs(pj - j) < min_distance:  # Manhattan distance
                    too_close = True
                    break
            if not too_close:
                selected_pois.append((i, j))
                self._poi_list.append((i, j))
            if len(selected_pois) == NUMBER_OF_POIS:
                break


    def get_POI_direction(self, agent):
        agent_x, agent_y = self._agent_location

        # Find the coordinates of the POI (assuming one POI with value 1 in self.POI_world)
        # poi_coords = self._agent_poi[agent.agent_id]
        poi_coords = self._poi_list
            
        if len(poi_coords) == 0:
            # No POI set
            self._POI_direction = np.zeros(8)
            return

        # Take the one closest to the agent
        distances = [np.linalg.norm(np.array(poi) - np.array([agent_x, agent_y])) for poi in poi_coords]
        closest_poi_index = np.argmin(distances)
        poi_x, poi_y = poi_coords[closest_poi_index]

        dx = poi_x - agent_x
        dy = poi_y - agent_y


        # Normalize direction to one of 8 compass directions
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)

        # Find the matching index in _action_to_direction
        direction_vector = np.array([dx, dy])
        direction_index = None
        for index, (ddx, ddy) in self._action_to_direction.items():
            if np.array_equal(direction_vector, np.array([ddx, ddy])):
                direction_index = index
                break

        # Create direction vector
        self._POI_direction = np.zeros(8)
        if direction_index is not None:
            self._POI_direction[direction_index] = 1
        

    def remove_POI(self, agent):
        self._agent_poi[agent.agent_id] = [
            poi for poi in self._agent_poi[agent.agent_id] if not np.array_equal(poi, agent.location)
        ]


    def step(self, action, agent):
        # Map the action (element of {0,1,2,3,4,5,6,7}) to the direction we walk in
        direction = self._action_to_direction[action]

        agent.location = np.clip(
            agent.location + direction, 0, self.size - 1
        )

        # Set agent location so other agents can see each other
        self.agents_location_list[agent.agent_id] = agent.location

        self._agent_location = agent.location

        self.visited_states[agent.location[0], agent.location[1]] += 1

    
        # Reset agent location for nearby agent observation
        self.global_location[agent.location[0], agent.location[1]] = 0

        self.global_location[agent.location[0], agent.location[1]] = 1

        self.check_rewards_near()
        self.get_POI_direction(agent)
        self.check_nearby_agents()

        # An environment is completed if and only if the agent has searched all states
        terminated = False
        truncated = False
        reward = self.getReward()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    
#----------------------------------- Agent --------------------------------------------------- #


class SAR_agent:
    def __init__(
        self,
        agent_id,
        env: gym.Env,
        alpha: float,
        beta,
        q_values,
        epsilon,
        gamma: float = 0.0
    ):
        self.agent_id = agent_id
        self.env = env
        self.location = np.array([-1, -1], dtype=np.int32)
        self.obs = tuple()
        self.action = 1

        self.q_values = q_values

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta

        self.battery_percent = 100
        self.spawn_point= np.array([-1, -1], dtype=np.int32)

        self.trajectory = []
        self.training_error = []

    def reset_trajectory(self):
        self.trajectory = []

    def agent_spawn(self, info):
        location = info['agent_location']
        self.location = location

    def add_trajectory(self, info):
        location = self.location
        self.trajectory.append(location)

    def write_q_table(self):
            np.save(f"q_tables/q_table.npy", dict(self.q_values))
        
    def battery_handler(self):
        self.battery_percent -= np.random.uniform(0, 1) # Random float from 0 to 3
        if self.battery_percent < BATTERY_THRESHOLD:
            self.location = self.spawn_point
            self.battery_percent = 100

        
    def get_state(self, obs: dict):
        reward_near = obs["reward_near"]
        nearby_agent = obs["nearby_agent"]
        poi = obs["POI_vector"]

        return (
            tuple(reward_near.flatten()),
            tuple(poi.flatten()),
            tuple(nearby_agent.flatten())
        )

    def mega_greedy_swarm_action(self, obs: dict, info: dict) -> int:
        agent_state = self.get_state(obs)
        q_values = self.q_values[agent_state]
      

        agent_loc = self.location
        world_border = info["world_border"]  # (size_x, size_y)
        size_x, size_y = world_border

        # Get list of valid actions
        valid_actions = []
        for action, (dx, dy) in self.env._action_to_direction.items():
            nx, ny = agent_loc[0] + dx, agent_loc[1] + dy
            if 0 <= nx < size_x and 0 <= ny < size_y:
                valid_actions.append(action)

        # Mask out invalid actions by setting them to a very low value
        masked_q_values = np.full_like(q_values, -np.inf)
        for a in valid_actions:
            masked_q_values[a] = q_values[a]



        return int(np.argmax(masked_q_values))

    def get_action_boltz(self, obs: dict, info: dict) -> int:
        agent_state = self.get_state(obs)

        q_values = self.q_values[agent_state]
       

        # --- Action Masking based on world borders ---
        agent_loc = self.location
        world_border = info["world_border"]  # assuming tuple (size, size)
        size_x, size_y = world_border
        valid_actions = []

        for action, (dx, dy) in self.env._action_to_direction.items():
            nx, ny = agent_loc[0] + dx, agent_loc[1] + dy
            if 0 <= nx < size_x and 0 <= ny < size_y:
                valid_actions.append(action)

        # Build a mask
        mask = np.zeros_like(q_values)
        mask[valid_actions] = 1

        exp_q = np.exp(self.beta * q_values - np.max(self.beta * q_values))
        exp_q *= mask 

    
        boltzmann_probs = exp_q / np.sum(exp_q)

        return np.random.choice(len(q_values), p=boltzmann_probs)

    def get_action_epsilon(self, obs: dict) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            agent_state = self.get_state(obs)
            return int(np.argmax(self.q_values[agent_state]))

    def update(
        self,
        obs: dict,  
        action: int,
        reward: float,
        terminated: bool,
        next_obs: dict  
    ):
        agent_state = self.get_state(obs)
        next_agent_state = self.get_state(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_agent_state])

        temporal_difference = (
            reward + self.gamma * future_q_value - self.q_values[agent_state][action]
        )

        self.q_values[agent_state][action] += self.alpha * temporal_difference
        self.training_error.append(temporal_difference)


    def plot_trajectory(self, time_step):
        plt.figure()
        for i, state in enumerate(self.trajectory):
            clear_output(wait=True)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Agent's Trajectory Simulation")
            plt.grid(True)
            
            past_x, past_y = zip(*self.trajectory[: i + 1])
            plt.plot(past_x, past_y, "bo-", label="Past Path")

            plt.plot(state[0], state[1], "ro", markersize=10, label="Current Position")
            
            plt.legend()
            plt.xlim(-1, self.env.size + 1)
            plt.ylim(-1, self.env.size + 1)
            plt.pause(time_step)
        
        plt.show()


#----------------------------------- World--------------------------------------------------- #


class swarm:
    def __init__(self, env, agents, n_episodes, update_step=1):
        self.env = env
        self.agents = agents
        self.visited_states = [[0 for _ in range(env.size)] for _ in range(env.size)]
        self.cum_reward = []    # :TODO overvej nyt navn
        self.episode_cum_reward = [0 for _ in range(N_EPISODES)]
        self.acum_info_pr_step = []
        self.acum_step_info_pr_episode = []
        self.n_episodes = n_episodes
        self.revisits = []
        self.info_pr_episode = []
        self.update_step = update_step
        self.episode_trajectory = []
        self.info_pr_step = []
        self.training_info = []
        self.info_pr_step_training = []
        self.step_pr_episode = []

    def state_visited(self, obs):
        x, y = obs['agent']
        self.visited_states[y][x] = 1

    def state_has_been_visited(self, obs):
        x, y = obs['agent']
        return self.visited_states[y][x] == 1

    def reset_trajectory(self):
        for agent in self.agents:
            agent.reset_trajectory()

    def calc_revisits(self, train_env):
        revisited = 0
        for i in range(train_env.size):
            for j in range(train_env.size):
                if train_env.visited_states[i][j] >= 1:
                    revisited += train_env.visited_states[i][j] - 1
        self.revisits.append(revisited)
    
    def accum_info(self,train_env):
        visited_rewards = 0
        for i in range(train_env.size):
            for j in range(train_env.size):
                if train_env.visited_states[i][j] >= 1:
                    visited_rewards += self.env.world[i][j] * 1
        total_rewards = np.sum(self.env.world)
        acum = visited_rewards / total_rewards
        self.acum_info_pr_step.append(acum)
        return acum

    def calc_info_pr_episode_training(self, train_env, steps):
        visited_rewards = 0
        for i in range(train_env.size):
            for j in range(train_env.size):
                if train_env.visited_states[i][j] >= 1:
                    visited_rewards += self.env.world[i][j] * 1
        total_rewards = np.sum(self.env.world)
        
        info_pr_episode = visited_rewards / total_rewards

        self.info_pr_step_training.append(info_pr_episode/steps)
        self.training_info.append(info_pr_episode)

    def calc_info_pr_episode(self, train_env, steps):
        visited_rewards = 0
        for i in range(train_env.size):
            for j in range(train_env.size):
                if train_env.visited_states[i][j] >= 1:
                    visited_rewards += self.env.world[i][j] * 1
        total_rewards = np.sum(self.env.world)
        
        info_pr_episode = visited_rewards / total_rewards

        self.info_pr_step.append(info_pr_episode/steps)
        self.info_pr_episode.append(info_pr_episode)


    def update_episode_trajectory(self):
        swarm_trajectory = []
        for agent in self.agents:
            swarm_trajectory.append(agent.trajectory)

        self.episode_trajectory.append(swarm_trajectory)

    def train_swarm(self, max_steps):
        progress_bar = tqdm(total=self.n_episodes, desc="Training Progress", unit="episode", leave=True, dynamic_ncols=True)
        seed = SEED

        for episode in range(self.n_episodes):
            train_env = copy.deepcopy(self.env)

            progress_bar.set_postfix_str(f"Ep {episode + 1}/{self.n_episodes}")

            obs, info = train_env.reset(self.agents)

            for agent in self.agents:
                agent.obs = copy.deepcopy(obs)

            seed += 45

            self.env.spawn_agents_uniform(self.agents, seed=seed)
            # for agent in agents:
            #     print("agent: ", agent.agent_id, " Location: ", agent.location)

            self.cum_reward = 0
            steps = 0
            done = False

            while not done:
                for agent in self.agents:
                    agent.action = agent.get_action_boltz(agent.obs, info)

                for agent in self.agents:
                    train_env.state_penalize()

                    next_obs, reward, terminated, truncated, info = train_env.step(agent.action, agent)
                    train_env.remove_POI(agent)
                    

                    agent.update(agent.obs, agent.action, reward, terminated, next_obs)

                    # if episode % 1000 == 0:
                    #    self.env.write_q_table()

                    self.episode_cum_reward[episode] += reward

                    if steps >= max_steps:
                        terminated = True
                    # if steps >= max_steps or self.accum_info(train_env) > 1:
                    #     terminated = True

                    done = terminated or truncated
                    agent.obs = copy.deepcopy(next_obs)
                    steps += 1
                    # agent.battery_handler()


            
            # self.calc_info_pr_episode(train_env, steps)
            progress_bar.update(1)
            self.calc_info_pr_episode_training(train_env, steps)

        # self.write_q_table()
        progress_bar.close()




    def evaluate_swarm(self, max_steps, number_of_episode):
        self.revisits = []
        progress_bar = tqdm(total=number_of_episode, desc="Evaluation Progress", unit="episode", leave=True, dynamic_ncols=True)
        seed = SEED

        for episode in range(number_of_episode):
            train_env = copy.deepcopy(self.env)
            progress_bar.set_postfix_str(f"Ep {episode + 1}/{number_of_episode}")
            # Reset the environment for the start of the episode
            obs, info = train_env.reset(self.agents)

            for agent in self.agents:
                agent.obs = copy.deepcopy(obs)

            seed += 45
            
            self.env.spawn_agents_uniform(self.agents, seed=seed)
            self.reset_trajectory()

            agent.add_trajectory(info)

            terminated = False
            steps = 0
            done = False

            # Run the episode until termination
            while not done:

                for agent in self.agents:
                    agent.action = agent.mega_greedy_swarm_action(agent.obs,info)

                for agent in self.agents:
                    train_env.state_penalize()

                    next_obs, reward, terminated, truncated, info = train_env.step(agent.action, agent)
                    train_env.remove_POI(agent)

                    agent.add_trajectory(info)

                    self.episode_cum_reward[episode] += reward

                    if steps >= max_steps or self.accum_info(train_env) > 0.8:
                        terminated = True

                    done = terminated or truncated
                    agent.obs = copy.deepcopy(next_obs)
                    steps += 1
                    # agent.battery_handler()

            self.calc_revisits(train_env)
            self.calc_info_pr_episode(train_env, steps)
            self.update_episode_trajectory()
            self.step_pr_episode.append(steps)
            self.acum_step_info_pr_episode.append(self.acum_info_pr_step)
            self.acum_info_pr_step = []

            progress_bar.update(1)

        progress_bar.close()

    def plot_reward_episode(self, number_of_episodes, window_size=50):
        """Plots reward per episode with a rolling mean and shaded variance"""
        plt.figure()
        
        x = range(number_of_episodes)
        y = np.array(self.episode_cum_reward[:number_of_episodes])
        
        # Compute rolling mean and standard deviation
        y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
        y_std = pd.Series(y).rolling(window=window_size, min_periods=1).std()

        # Plot raw data faintly
        plt.plot(x, y, alpha=0.2, label="Raw Rewards", color='gray')

        # Plot rolling mean
        plt.plot(x, y_smooth, label="Smoothed Rewards", color='red')

        # Shaded area: Mean ± 1 standard deviation
        plt.fill_between(x, y_smooth - y_std, y_smooth + y_std, color='red', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')
        plt.legend()
        plt.savefig("plots/reward_pr_episode.png", dpi=500)

    def plot_acum_info_pr_step_per_episode(self):
        """Plots all episodes' step-wise data with unique colors and episode labels."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        plt.figure(figsize=(12, 6))

        num_episodes = len(self.acum_step_info_pr_episode)
        cmap = plt.get_cmap('tab20')  # Good for up to 20 distinct colors

        for idx, episode_data in enumerate(self.acum_step_info_pr_episode):
            steps = np.arange(len(episode_data))
            color = cmap(idx % 20)  # Reuse colors cyclically if > 20 episodes
            plt.plot(steps, episode_data, label=f"Episode {idx + 1}", color=color, linewidth=1.5)

        plt.xlabel('Steps')
        plt.ylabel('Accumulated info')
        plt.title('Accumulated info per step')
        plt.grid(True)

        # If too many episodes, shrink legend or skip it
        if num_episodes <= 20:
            plt.legend(loc='best', fontsize='small')
        else:
            plt.legend(loc='best', fontsize='x-small', ncol=2)

        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/acum_info_pr_step_colored_legend.png", dpi=500)



    def plot_revisited(self, number_of_episodes, window_size=50,):
        """Plots revisits per episode with rolling mean and variance shading"""
        plt.figure()

        x = range(number_of_episodes)
        y = np.array(self.revisits[:number_of_episodes])

        # Compute rolling mean and standard deviation
        y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
        y_std = pd.Series(y).rolling(window=window_size, min_periods=1).std()

        # Plot raw data faintly
        plt.plot(x, y, alpha=0.2, label="Raw Revisits", color='gray')

        # Plot rolling mean
        plt.plot(x, y_smooth, label="Smoothed Revisits", color='red')

        # Shaded area: Mean ± 1 standard deviation
        plt.fill_between(x, y_smooth - y_std, y_smooth + y_std, color='red', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Revisits')
        plt.title('Revisits per Episode')
        plt.legend()
        plt.savefig("plots/revisites.png", dpi=500)

    def plot_info(self, number_of_episodes, window_size=50):
        """Plots info gain per episode with rolling mean and variance shading"""
        plt.figure()

        x = range(number_of_episodes)
        y = np.array(self.info_pr_episode[:number_of_episodes])

        # Compute rolling mean and standard deviation
        y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
        y_std = pd.Series(y).rolling(window=window_size, min_periods=1).std()

        # Plot raw data faintly
        plt.plot(x, y, alpha=0.2, label="Raw Info Gain", color='gray')

        # Plot rolling mean
        plt.plot(x, y_smooth, label="Smoothed Info Gain", color='red')

        # Shaded area: Mean ± 1 standard deviation
        plt.fill_between(x, y_smooth - y_std, y_smooth + y_std, color='red', alpha=0.2)
        # :TODO promt 95% standard deviation

        plt.xlabel('Episodes')
        plt.ylabel('Info Gain')
        plt.title('Info Gain per Episode')
        plt.legend()
        plt.savefig("plots/info_pr_episode.png", dpi=500)

    def plot_steps(self, number_of_episodes, window_size=50):
        """Plots info gain per episode with rolling mean and variance shading"""
        plt.figure()

        x = range(number_of_episodes)
        y = np.array(self.step_pr_episode[:number_of_episodes])

        # Compute rolling mean and standard deviation
        y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
        y_std = pd.Series(y).rolling(window=window_size, min_periods=1).std()

        # Plot raw data faintly
        plt.plot(x, y, alpha=0.2, label="Steps pr Episode", color='gray')

        # Plot rolling mean
        plt.plot(x, y_smooth, label="Smoothed Info Gain", color='red')

        # Shaded area: Mean ± 1 standard deviation
        plt.fill_between(x, y_smooth - y_std, y_smooth + y_std, color='red', alpha=0.2)
        # :TODO promt 95% standard deviation

        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.title('Steps per Episode')
        plt.legend()
        plt.savefig("plots/steps_pr_episode.png", dpi=500)

    def plot_training_info(self, number_of_episodes, window_size=50):
        """Plots info gain per episode with rolling mean and variance shading"""
        plt.figure()

        x = range(number_of_episodes)
        y = np.array(self.training_info[:number_of_episodes])

        # Compute rolling mean and standard deviation
        y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
        y_std = pd.Series(y).rolling(window=window_size, min_periods=1).std()

        # Plot raw data faintly
        plt.plot(x, y, alpha=0.2, label="Raw Info Gain", color='gray')

        # Plot rolling mean
        plt.plot(x, y_smooth, label="Smoothed Info Gain", color='red')

        # Shaded area: Mean ± 1 standard deviation
        plt.fill_between(x, y_smooth - y_std, y_smooth + y_std, color='red', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Info Gain')
        plt.title('Info Gain per Episode')
        plt.legend()
        plt.savefig("plots/training_info_pr_episode.png", dpi=500)
    
    def plot_info_pr_step(self, number_of_episodes, window_size=50):
        """Plots info gain per episode with rolling mean and variance shading"""
        plt.figure()

        x = range(number_of_episodes)
        y = np.array(self.info_pr_step[:number_of_episodes])

        # Compute rolling mean and standard deviation
        y_smooth = pd.Series(y).rolling(window=window_size, min_periods=1).mean()
        y_std = pd.Series(y).rolling(window=window_size, min_periods=1).std()

        # Plot raw data faintly
        plt.plot(x, y, alpha=0.2, label="Raw step info", color='gray')

        # Plot rolling mean
        plt.plot(x, y_smooth, label="Smoothed step info", color='red')

        # Shaded area: Mean ± 1 standard deviation
        plt.fill_between(x, y_smooth - y_std, y_smooth + y_std, color='red', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Info pr Step')
        plt.title('Info pr step')
        plt.legend()
        plt.savefig("plots/info_pr_step.png", dpi=500)
    
    def plot_trajectories(self, train_env, num_episodes, episodes_to_plot=None, max_agents=5):
        world_map = train_env.world

        if episodes_to_plot is None:
            episodes_to_plot = list(range(num_episodes))

        num_plots = len(episodes_to_plot)
        cols = 2
        rows = int(np.ceil(num_plots / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
        axs = axs.flatten()

        for i, ep_idx in enumerate(episodes_to_plot):
            ax = axs[i]
            ax.imshow(
                world_map.T,
                origin="lower",
                cmap="Greys",
                interpolation="nearest",
                extent=[0, world_map.shape[0], 0, world_map.shape[1]]
            )

            swarm_traj = self.episode_trajectory[ep_idx]
            num_agents = len(self.agents)

            for agent_id in range(num_agents):
                traj = np.array(swarm_traj[agent_id])
                # Plot trajectory line
                ax.plot(traj[:, 0], traj[:, 1], label=f"A{agent_id}", alpha=0.7)
                # Plot spawn point (first position)
                ax.scatter(traj[0, 0], traj[0, 1], s=40, marker='o', edgecolors='k', facecolors='none')

            ax.set_title(f"Episode {ep_idx + 1}")
            ax.set_xlim(0, train_env.size)
            ax.set_ylim(0, train_env.size)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.label_outer()

        # Hide unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        # Shared legend
        handles, labels = axs[0].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.05, 0.5))
        # fig.suptitle("Agent Trajectories per Episode", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        # Save the entire plot as an image file
        plt.savefig("plots/agent_trajectories.png", dpi=500)

    def plot_single_episode(self, train_env):
        world_map = train_env.world
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            world_map.T,
            origin="lower",
            cmap="Greys",
            interpolation="nearest",
            extent=[0, world_map.shape[0], 0, world_map.shape[1]]
        )

        swarm_traj = self.episode_trajectory[-1]
        num_agents = len(self.agents)

        for agent_id in range(num_agents):
            traj = np.array(swarm_traj[agent_id])
            # Plot trajectory line
            ax.plot(traj[:, 0], traj[:, 1], label=f"A{agent_id}", alpha=0.7)
            # Plot spawn point (first position)
            ax.scatter(traj[0, 0], traj[0, 1], s=40, marker='o', edgecolors='k', facecolors='none')

        ax.set_title(f"Episode {len(self.episode_trajectory)}")
        ax.set_xlim(0, train_env.size)
        ax.set_ylim(0, train_env.size)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.label_outer()

        plt.savefig("plots/agent_trajectories.png", dpi=500)


#----------------------------------- Hyper parameters --------------------------------------------------- #


env = GridWorldEnv(size=SIZE)

# env.random_env()
# env.setReward(4, 4, 9)
env.heatmap_env()
print(env._poi_list)


# plt.gca().invert_yaxis()
# plt.imshow(env.world, cmap='viridis', origin='upper')
# plt.colorbar(label='Reward')
# plt.title('Environment World')
# plt.show()

env_timelimit = gym.wrappers.TimeLimit(env, max_episode_steps=1000000)
q_values = defaultdict(lambda: np.zeros(env.action_space.n))

if(input == 1 or input == 2):
    with open('q_values.pkl', 'rb') as f:
        q_values = pickle.load(f)
elif(input == 0):
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))

agents = [
    SAR_agent(
        agent_id=i,
        env=env,
        alpha=ALPHA,
        beta=BETA,
        q_values=q_values,
        epsilon=EPSILON,
        gamma=GAMMA
    )
    for i in range(N_AGENTS)
]


#----------------------------------- Hyper parameters --------------------------------------------------- #


swarm1 = swarm(env, agents, N_EPISODES, UPDATE_STEP)

if(input == 0 or input == 2):
    swarm1.train_swarm(STEPS)
    with open('q_values.pkl', 'wb') as f:
        pickle.dump(q_values, f)
    swarm1.plot_training_info(N_EPISODES)

if(input == 1):
    swarm1.evaluate_swarm(SIZE*SIZE,EVALUATION_EPISODES)
    swarm1.plot_reward_episode(EVALUATION_EPISODES)
    swarm1.plot_revisited(EVALUATION_EPISODES)
    swarm1.plot_info(EVALUATION_EPISODES)
    swarm1.plot_info_pr_step(EVALUATION_EPISODES)
    swarm1.plot_steps(EVALUATION_EPISODES)
    swarm1.plot_acum_info_pr_step_per_episode()
    if(EVALUATION_EPISODES == 1):
        swarm1.plot_single_episode(env)
    else:
        swarm1.plot_trajectories(env, EVALUATION_EPISODES)
