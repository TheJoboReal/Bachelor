import numpy as np
import matplotlib.pyplot as plt
import time

# Simulation parameters
NUM_DRONES = 10  # Number of drones in the simulation
GRID_SIZE = 20   # Size of the search grid
BETA = 0.15  # Softmax temperature parameter
ALPHA = 0.7  # Weight between individual and cooperative rewards

# Priority map is now input, no need for discovery
PRIORITY_MAP = np.random.randint(1, 6, (GRID_SIZE, GRID_SIZE))  # Predefined heatmap
SEARCHED_MAP = np.zeros((GRID_SIZE, GRID_SIZE))  # Track searched areas

class Drone:
    def __init__(self, drone_id):
        self.id = drone_id  # Unique identifier for the drone
        self.position = np.random.randint(0, GRID_SIZE, size=2)  # Initial random position
        self.visited_positions = [tuple(self.position)]  # Track visited positions
        self.cumulative_reward = 0  # Keep track of total search priority collected
        self.active = True  # Drone status
    
    def compute_reward(self, new_position):
        """ Compute reward using the weighted individual and cooperative components """
        R_ind = PRIORITY_MAP[new_position[0], new_position[1]]
        R_coop = 1 / (1 + SEARCHED_MAP[new_position[0], new_position[1]])  # Incentivize unexplored areas
        return ALPHA * R_ind + (1 - ALPHA) * R_coop
    
    def softmax_policy(self, available_actions):
        """ Select action based on softmax over Q-values """
        q_values = np.array([self.compute_reward(pos) for pos in available_actions])
        exp_q = np.exp(BETA * q_values)
        probabilities = exp_q / np.sum(exp_q)
        return available_actions[np.random.choice(len(available_actions), p=probabilities)]
    
    def update_position(self):
        """ Move towards an area based on softmax action selection """
        if not self.active:
            return
        
        search_directions = [
            np.array([1, 0]), np.array([-1, 0]),
            np.array([0, 1]), np.array([0, -1])
        ]
        
        available_actions = []
        for direction in search_directions:
            new_position = self.position + direction
            if 0 <= new_position[0] < GRID_SIZE and 0 <= new_position[1] < GRID_SIZE:
                available_actions.append(new_position)
        
        if available_actions:
            best_move = self.softmax_policy(available_actions)
            self.position = best_move
            self.cumulative_reward += self.compute_reward(best_move)
            SEARCHED_MAP[self.position[0], self.position[1]] += 1  # Mark as searched
        
        self.visited_positions.append(tuple(self.position))

def run_simulation():
    """ Run the search and visualization of drone movement in real time """
    drones = [Drone(i) for i in range(NUM_DRONES)]  # Initialize drones
    
    plt.ion()  # Enable interactive mode for real-time plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    
    iteration = 0
    while True:
        ax.clear()
        ax.imshow(PRIORITY_MAP, cmap='hot', origin='lower', alpha=0.5)  # Display priority map
        ax.imshow(SEARCHED_MAP, cmap='cool', origin='lower', alpha=0.5)  # Overlay search progress
        
        for drone in drones:
            drone.update_position()
            path = np.array(drone.visited_positions)
            ax.plot(path[:, 1], path[:, 0], linestyle='--', marker='o', markersize=3, label=f'Drone {drone.id}')
        
        ax.set_xlim(0, GRID_SIZE - 1)
        ax.set_ylim(0, GRID_SIZE - 1)
        ax.set_title(f"Iteration {iteration + 1}")
        ax.legend()
        ax.grid()
        plt.draw()
        plt.pause(0.1)  # Pause for a short duration to simulate real-time updates
        iteration += 1
    
run_simulation()
