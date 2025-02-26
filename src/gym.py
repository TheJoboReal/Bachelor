import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiAgentScanEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), num_agents=3, discount_factor=0.9):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.state = np.zeros(grid_size)  # 0 means unscanned, 1 means scanned
        self.discount_factor = discount_factor
        
        # Each agent has an (x, y) position
        self.agent_positions = [
            (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])) 
            for _ in range(num_agents)
        ]
        
        # Define action space: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_space = spaces.MultiDiscrete([5] * num_agents)
        
        # Observation space: each agent sees the entire grid and all agents' positions
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.uint8
        )
        
    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        
        # Move each agent
        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            
            if action == 0 and x > 0:
                x -= 1  # Up
            elif action == 1 and x < self.grid_size[0] - 1:
                x += 1  # Down
            elif action == 2 and y > 0:
                y -= 1  # Left
            elif action == 3 and y < self.grid_size[1] - 1:
                y += 1  # Right
            
            self.agent_positions[i] = (x, y)
            
            # Reward for scanning new area
            if self.state[x, y] == 0:
                rewards[i] = 1  # Positive reward for scanning
                self.state[x, y] = 1  # Mark as scanned
            else:
                rewards[i] = -0.1  # Small penalty for redundant scanning
        
        # Compute discounted rewards
        rewards = rewards * self.discount_factor
        
        # Check if all grid is scanned (termination condition)
        done = np.all(self.state == 1)
        
        return self.state.copy(), rewards, done, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.grid_size)
        self.agent_positions = [
            (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])) 
            for _ in range(self.num_agents)
        ]
        return self.state.copy(), {}
    
    def render(self):
        grid = self.state.copy()
        for x, y in self.agent_positions:
            grid[x, y] = 2  # Represent agents with '2'
        print(grid)
