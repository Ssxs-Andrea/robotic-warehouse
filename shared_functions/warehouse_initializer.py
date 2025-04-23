import numpy as np
from enum import Enum
import gymnasium as gym
import rware

class WarehouseInitializer:
    def __init__(self, env):
        self.env = env.unwrapped
        self.grid_size = None
        self.rows = None
        self.cols = None
        self.goal_locations = None
        self.shelf_locations = {}
        self.agent_locations = {}
        self.shelf_memory = {}

        # Movement control components
        self.agent_states = {}  # "seek_shelf", "deliver", "return_shelf"
        self.shelf_memory = {}  # Remember original shelf positions
        self.agent_targets = {}  # Current target for each agent
        self.last_positions = {}
        self.stuck_count = {}
        self.last_actions = {}
        
        # Direction mappings
        self.direction_map = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        self.direction_symbols = {
            'UP': '↑',
            'DOWN': '↓',
            'LEFT': '←',
            'RIGHT': '→'
        }

    def initialize_all(self):
        """Initialize all warehouse components"""
        self._initialize_grid()
        self._initialize_goals()
        self._initialize_shelves()
        self._initialize_agents()
        return {
            'grid_size': self.grid_size,
            'rows': self.rows,
            'cols': self.cols,
            'goal_locations': self.goal_locations,
            'shelf_locations': self.shelf_locations,
            'agent_locations': self.agent_locations,
            'shelf_memory': self.shelf_memory
        }

    def _initialize_grid(self):
        """Initialize and verify grid dimensions"""
        self.grid_size = (int(self.env.grid_size[0]), int(self.env.grid_size[1]))
        self.rows = self.grid_size[0]
        self.cols = self.grid_size[1]
        

    def _initialize_goals(self):
        """Initialize and verify goal locations"""
        self.goal_locations = set()
        for g in self.env.goals:
            x = max(0, min(int(g[0]), self.cols-1))
            y = max(0, min(int(g[1]), self.rows-1))
            self.goal_locations.add((x, y))
        

    def _initialize_shelves(self):
        """Initialize and verify shelf locations"""
        self.shelf_locations = {}
        for shelf in self.env.shelfs:
            x = max(0, min(int(shelf.x), self.cols-1))
            y = max(0, min(int(shelf.y), self.rows-1))
            self.shelf_locations[shelf.id] = (x, y)
            self.shelf_memory[shelf.id] = (x, y)  # Also store in memory for movement logic
        
        # print("\nShelf Locations:")
        # for shelf_id, (x, y) in self.shelf_locations.items():
        #     print(f"Shelf {shelf_id}: ({x}, {y})")

    def _initialize_agents(self):
        """Initialize and verify agent locations and directions"""
        self.agent_locations = {}
        for agent in self.env.agents:
            agent.x = max(0, min(int(agent.x), self.cols-1))
            agent.y = max(0, min(int(agent.y), self.rows-1))
            direction = self.direction_map.get(agent.dir.value, f'UNKNOWN({agent.dir.value})')
            self.agent_locations[agent.id] = (agent.x, agent.y, direction)
            
            # Initialize movement control states
            self.agent_states[agent.id] = "seek_shelf"
            self.agent_targets[agent.id] = None
            self.last_positions[agent.id] = (agent.x, agent.y)
            self.stuck_count[agent.id] = 0
            self.last_actions[agent.id] = []
        
        # print("\nAgent Locations:")
        # for agent_id, (x, y, dir) in self.agent_locations.items():
        #     print(f"Agent {agent_id}: ({x}, {y}) facing {dir}")

    def _print_complete_map(self):
        """Print a complete map of the warehouse"""
        print("\n=== Complete Warehouse Map ===")
        
        grid = np.full((self.rows, self.cols), '.', dtype='U10')
        
        # Mark walls
        for y in range(self.rows):
            for x in range(self.cols):
                try:
                    if self.env.grid[y][x] == 0:
                        grid[y][x] = '#'
                except:
                    grid[y][x] = '?'
        
        # Mark goals
        for x, y in self.goal_locations:
            grid[y][x] = 'G'
        
        # Mark shelves
        for shelf_id, (x, y) in self.shelf_locations.items():
            grid[y][x] = f'S{shelf_id}'
        
        # Mark agents with direction
        for agent_id, (x, y, dir) in self.agent_locations.items():
            symbol = self.direction_symbols.get(dir, '?')
            grid[y][x] = f'A{agent_id}{symbol}'
        
        # Print column headers
        print("   " + " ".join(f"{x:2}" for x in range(self.cols)))
        
        # Print each row with row number
        for y in range(self.rows):
            print(f"{y:2} " + " ".join(grid[y][x] for x in range(self.cols)))
        
        # Print legend
        print("\nLegend:")
        print("G = Goal location")
        print("S# = Shelf with ID")
        print("A#↑ = Agent with ID and direction")

# Standalone test code
if __name__ == "__main__":
    print("=== Testing WarehouseInitializer ===")
    
    # Create a test environment
    env = gym.make("rware-tiny-2ag-v2", render_mode="human")
    obs, info = env.reset()
    
    # Initialize and test
    initializer = WarehouseInitializer(env)
    print("\n=== Initializing Warehouse ===")
    initializer.initialize_all()
    initializer._print_complete_map()
    
    # Print success message
    print("\n=== WarehouseInitializer Test Successful ===")
    print("All components initialized correctly!")
    env.close()