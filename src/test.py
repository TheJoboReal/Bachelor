import numpy as np
import matplotlib.pyplot as plt
import time

# Simulation parameters
NUM_DRONES = 10  # Number of drones in the simulation
GRID_SIZE = 20   # Size of the search grid
MAX_ITERATIONS = 100  # Maximum number of iterations before stopping

# Define priority map (higher values indicate higher priority search areas)
PRIORITY_MAP = np.random.randint(1, 5, (GRID_SIZE, GRID_SIZE))
SEARCHED_MAP = np.zeros((GRID_SIZE, GRID_SIZE))  # Track searched areas

class Drone:
    def __init__(self, drone_id):
        self.id = drone_id  # Unique identifier for the drone
        self.position = np.random.randint(0, GRID_SIZE, size=2)  # Initial random position
        self.visited_positions = [tuple(self.position)]  # Track visited positions
        self.cumulative_reward = 0  # Keep track of total search priority collected
    
    def update_position(self):
        """ Move towards the highest priority unexplored area while avoiding previously searched zones """
        search_directions = [
            np.array([1, 0]), np.array([-1, 0]),
            np.array([0, 1]), np.array([0, -1])
        ]
        best_move = None
        best_priority = -1

        for direction in search_directions:
            new_position = self.position + direction
            if 0 <= new_position[0] < GRID_SIZE and 0 <= new_position[1] < GRID_SIZE:
                if SEARCHED_MAP[new_position[0], new_position[1]] == 0:  # Ensure the area is not already searched
                    priority = PRIORITY_MAP[new_position[0], new_position[1]]
                    if priority > best_priority:
                        best_priority = priority
                        best_move = new_position

        if best_move is not None:
            self.position = best_move
            self.cumulative_reward += PRIORITY_MAP[self.position[0], self.position[1]]
            SEARCHED_MAP[self.position[0], self.position[1]] = 1  # Mark as searched
        
        self.visited_positions.append(tuple(self.position))

def run_simulation():
    """ Run the search and visualization of drone movement in real time """
    drones = [Drone(i) for i in range(NUM_DRONES)]  # Initialize drones
    
    plt.ion()  # Enable interactive mode for real-time plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for iteration in range(MAX_ITERATIONS):
        ax.clear()
        ax.imshow(PRIORITY_MAP, cmap='coolwarm', origin='lower')
        
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
        
        if np.all(SEARCHED_MAP == 1):  # Stop if the entire map has been searched
            print("Entire map searched.")
            break
    else:
        print("Simulation ended without fully searching the map.")
    
    # Display cumulative rewards collected by each drone
    for drone in drones:
        print(f"Drone {drone.id} cumulative reward: {drone.cumulative_reward}")
    
    plt.ioff()  # Disable interactive mode after simulation
    plt.show()

run_simulation()
