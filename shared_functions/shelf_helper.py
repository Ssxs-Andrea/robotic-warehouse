import numpy as np

class ShelfHelper:
    def __init__(self, env, agent_states_ref=None):
        self.env = env.unwrapped
        self.grid_size = (int(self.env.grid_size[0]), int(self.env.grid_size[1]))
        self.rows = self.grid_size[0]
        self.cols = self.grid_size[1]
        self.agent_states = agent_states_ref
        self.shelf_memory = {}  # Track original shelf positions
        self._initialize_shelf_memory()
        self.shelf_locations = self._extract_shelf_locations()

    def _initialize_shelf_memory(self):
        """Store original shelf positions"""
        for shelf in self.env.shelfs:
            self.shelf_memory[shelf.id] = (int(shelf.x), int(shelf.y))

    def _extract_shelf_locations(self):
        """Extract shelf positions directly from environment"""
        shelf_locations = set()
        for shelf in self.env.shelfs:
            x = max(0, min(int(shelf.x), self.cols-1))
            y = max(0, min(int(shelf.y), self.rows-1))
            shelf_locations.add((x, y))
        return shelf_locations

    def get_blocked_positions(self, agent, shelf_width=1, shelf_height=1):
        """Get blocked positions based on agent state"""
        blocked = set()

        # Add obstacle positions
        for y in range(self.env.obstacles.shape[0]):
            for x in range(self.env.obstacles.shape[1]):
                if self.env.obstacles[y, x]:
                    blocked.add((x, y))
        
        # Always block other agents (regardless of state)
        for other_agent in self.env.agents:
            if other_agent != agent:
                blocked.add((int(other_agent.x), int(other_agent.y)))

        # Get current state (with fallback for agents without state)
        current_state = getattr(agent, 'state', None) or \
                    (self.agent_states.get(agent.id) if self.agent_states else None)

        # Block shelves only in deliver or return_shelf states
        if current_state in ["deliver", "return_shelf", "charging"]:
            if agent.carrying_shelf:
                # Don't block the shelf we're carrying (for return_shelf)
                carried_shelf_pos = self.shelf_memory.get(agent.carrying_shelf.id, (-1, -1))
                for pos in self.shelf_locations:
                    if pos != carried_shelf_pos:
                        blocked.add(pos)
            else:
                # Block all shelves if not carrying (shouldn't happen in these states)
                blocked.update(self.shelf_locations)
        # In seek_shelf state, shelves are NOT blocked (agent can pass through them)

        return blocked

    def get_shelf_memory(self):
        """Get the shelf memory dictionary"""
        return self.shelf_memory

    def get_shelf_locations(self):
        """Get the current shelf locations"""
        return self.shelf_locations
    
    def print_warehouse_map(self, agent):
        """Print an ASCII map showing shelves, agents, and open spaces"""
        # Create empty grid
        grid = np.full((self.rows, self.cols), '.', dtype='U1')
        
        # Mark shelves
        for shelf_pos in self.shelf_locations:
            x, y = shelf_pos
            grid[y, x] = 'S'  # Note: y comes first in numpy arrays
        
        # Mark blocked positions (including shelves when applicable)
        blocked = self.get_blocked_positions(agent)
        for (x, y) in blocked:
            if grid[y, x] == '.':
                grid[y, x] = 'X'  # Other blocked positions
        
        # Mark agents
        for other_agent in self.env.agents:
            x, y = int(other_agent.x), int(other_agent.y)
            if other_agent == agent:
                grid[y, x] = 'A'  # Current agent
            else:
                grid[y, x] = 'O'  # Other agents

        # Mark goal position if provided
        for goal_pos in self.env.goals:
            gx, gy = int(goal_pos[0]), int(goal_pos[1])
            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                grid[gy, gx] = 'G'  # Goal position
        
        # Print the map with borders
        print("+" + "-" * self.cols + "+")
        for row in reversed(range(self.rows)):  # Reverse to match coordinate system
            print("|" + "".join(grid[row, :]) + "|")
        print("+" + "-" * self.cols + "+")
        
        # Print legend
        print("Legend:")
        print("A - Current agent")
        print("O - Other agents")
        print("S - Shelves")
        print("X - Blocked positions")
        print(". - Free space")