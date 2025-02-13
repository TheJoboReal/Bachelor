import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Simulation parameters
NUM_DRONES = 10  # Number of drones in the simulation
GRID_SIZE = 100   # Size of the search grid
MAX_ITERATIONS = 1000  # Maximum number of iterations before stopping

# Define priority map (higher values indicate higher priority search areas)
# Create a random priority map
priority_map = np.random.randint(1, 5, (GRID_SIZE, GRID_SIZE))

# Apply Gaussian filter to create clusters
PRIORITY_MAP = gaussian_filter(priority_map, sigma=5)
PRIORITY_MAP = np.clip(PRIORITY_MAP, 1, 5)  # Ensure values are within the original range

class Drone:
    def __init__(self, drone_id):
        self.id = drone_id  # Unique identifier for the drone
        self.position = np.random.randint(0, GRID_SIZE, size=2)  # Initial random position
        self.velocity = np.random.randint(-1, 2, size=2)  # Initial velocity (not used extensively)
        self.visited_positions = [tuple(self.position)]  # Track visited positions
        self.cumulative_reward = 0  # Keep track of total search priority collected
    
    def update_position(self):
        """ Move towards the highest priority unexplored area """
        search_directions = [
            np.array([1, 0]), np.array([-1, 0]),
            np.array([0, 1]), np.array([0, -1])
        ]
        best_move = None
        best_priority = -1

        for direction in search_directions:
            new_position = self.position + direction
            if 0 <= new_position[0] < GRID_SIZE and 0 <= new_position[1] < GRID_SIZE:
                if tuple(new_position) not in self.visited_positions:
                    priority = PRIORITY_MAP[new_position[0], new_position[1]]
                    if priority > best_priority:
                        best_priority = priority
                        best_move = new_position

        if best_move is not None:
            self.position = best_move
            self.cumulative_reward += PRIORITY_MAP[self.position[0], self.position[1]]
        
        self.visited_positions.append(tuple(self.position))

def run_simulation():
    """ Run the search and visualization of drone movement """
    drones = [Drone(i) for i in range(NUM_DRONES)]  # Initialize drones
    searched_positions = set()
    
    for iteration in range(MAX_ITERATIONS):
        for drone in drones:
            drone.update_position()
            searched_positions.add(tuple(drone.position))
        
        # Stop if the entire map has been searched
        if len(searched_positions) >= GRID_SIZE * GRID_SIZE:
            print("Entire map searched.")
            break
    else:
        print("Simulation ended without fully searching the map.")
    
    # Display cumulative rewards collected by each drone
    for drone in drones:
        print(f"Drone {drone.id} cumulative reward: {drone.cumulative_reward}")
    
    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(PRIORITY_MAP, cmap='coolwarm', origin='lower')
    for drone in drones:
        path = np.array(drone.visited_positions)
        plt.plot(path[:, 1], path[:, 0], linestyle='--', marker='o', markersize=3, label=f'Drone {drone.id}')
    plt.colorbar(label='Search Priority')
    plt.xlim(0, GRID_SIZE - 1)
    plt.ylim(0, GRID_SIZE - 1)
    plt.legend()
    plt.grid()
    plt.show()

run_simulation()
