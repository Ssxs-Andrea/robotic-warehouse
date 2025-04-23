import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from collections import deque
from shared_functions.enums import Action
from shared_functions.shelf_helper import ShelfHelper

class ShelfCarryingMovement:
    def __init__(self, env, agent_states_ref=None):
        self.env = env.unwrapped
        self.grid_size = (int(self.env.grid_size[0]), int(self.env.grid_size[1]))
        self.rows = self.grid_size[0]
        self.cols = self.grid_size[1]
        self.movement_queue = []
        self.debug = False
        self.agent_states = agent_states_ref 
        self.shelf_memory = {}
        
        # Initialize shelf helper
        self.shelf_helper = ShelfHelper(env, agent_states_ref)
        
        self.direction_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.action_names = {
            0: 'NOOP', 1: 'FORWARD', 2: 'LEFT', 3: 'RIGHT', 4: 'TOGGLE_LOAD'
        }

    def calculate_movement(self, agent, goal_pos, shelf_width=1, shelf_height=1):
        current_pos = (int(agent.x), int(agent.y))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        
        if self.debug:
            print(f"\n=== Starting Movement Calculation ===")
            print(f"Agent Start: {current_pos}, Facing: {self.direction_map[agent.dir.value]}")
            print(f"Goal Position: {goal_pos}")
            print(f"Shelf Size: {shelf_width}x{shelf_height}")

        # Get blocked positions
        blocked = self.shelf_helper.get_blocked_positions(agent, shelf_width, shelf_height)
        
        if self.debug:
            print(f"Blocked positions count: {len(blocked)}")
            if len(blocked) < 20:
                print(f"Blocked positions: {sorted(blocked)}")

        # Goal validation
        if not self._is_position_valid(goal_pos):
            if self.debug:
                print(f"❌ Invalid goal position: {goal_pos} (Grid size: {self.rows}x{self.cols})")
            return []
        elif goal_pos in blocked:
            if self.debug:
                print(f"❌ Goal position blocked: {goal_pos}")
            return []
        
        if self.debug:
            print("Attempting BFS pathfinding...")
        
        actions = self._pathfind_movement(agent, goal_pos, blocked)
        
        if actions:
            if self._validate_action_sequence(agent, actions, blocked):
                if self.debug:
                    print(f"✅ BFS found valid path with {len(actions)} actions")
                return actions
            else:
                if self.debug:
                    print("❌ BFS path invalid - would collide")
        
        if self.debug:
            print("❌ BFS failed to find valid path")
        return []

    def _validate_action_sequence(self, agent, actions, blocked):
        """Proper validation that tracks position AND direction after each action"""
        x, y = int(agent.x), int(agent.y)
        direction = agent.dir.value
        
        for action in actions:
            if action == Action.FORWARD.value:
                # Calculate new position based on CURRENT direction
                new_x, new_y = x, y
                if direction == 0: new_y -= 1    # UP
                elif direction == 1: new_y += 1  # DOWN
                elif direction == 2: new_x -= 1  # LEFT
                elif direction == 3: new_x += 1  # RIGHT
                
                # Check new position
                if not self._is_position_valid((new_x, new_y)) or (new_x, new_y) in blocked:
                    return False
                    
                # Update position after successful move
                x, y = new_x, new_y
                
            elif action in (Action.LEFT.value, Action.RIGHT.value):
                # Update direction
                if action == Action.LEFT.value:
                    direction = {0:2, 1:3, 2:1, 3:0}[direction]
                else:
                    direction = {0:3, 1:2, 2:0, 3:1}[direction]
        
        return True

    def _is_position_valid(self, pos):
        """Check if position is within bounds"""
        x, y = pos
        return 0 <= x < self.cols and 0 <= y < self.rows

    def _pathfind_movement(self, agent, goal_pos, blocked):
        """BFS pathfinding implementation"""
        start_pos = (int(agent.x), int(agent.y))
        
        if start_pos == goal_pos:
            return []
        
        # BFS setup
        queue = deque()
        queue.append(start_pos)
        visited = {start_pos: None}  # Tracks where we came from
        
        found = False
        
        while queue and not found:
            current = queue.popleft()
            
            for neighbor in self._get_neighbors(current):
                if neighbor == goal_pos:
                    visited[neighbor] = current
                    found = True
                    break
                
                if neighbor not in visited and neighbor not in blocked:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        if not found:
            return []  # No path found
        
        # Reconstruct path
        path = []
        current = goal_pos
        while current != start_pos:
            path.append(current)
            current = visited[current]
        path.reverse()
        
        # Convert path to actions
        return self._convert_path_to_actions(agent, path)

    def _get_neighbors(self, pos):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
                neighbors.append((new_x, new_y))
                
        return neighbors

    def _convert_path_to_actions(self, agent, path):
        """Convert a path of positions to a sequence of actions"""
        if not path:
            return []
            
        actions = []
        current_dir = agent.dir.value
        current_pos = (int(agent.x), int(agent.y))
        
        for next_pos in path:
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            
            # Determine desired direction
            if dx == 1:
                desired_dir = 3  # RIGHT
            elif dx == -1:
                desired_dir = 2  # LEFT
            elif dy == 1:
                desired_dir = 1  # DOWN
            elif dy == -1:
                desired_dir = 0  # UP
            else:
                continue  # Shouldn't happen for valid path
                
            # Turn to face direction if needed
            if current_dir != desired_dir:
                actions += self._turn_to_face(current_dir, desired_dir)
                current_dir = desired_dir
                
            # Move forward
            actions.append(Action.FORWARD.value)
            current_pos = next_pos
            
        return actions

    def _turn_to_face(self, current, desired):
        """Returns turn actions based on actual direction mapping"""
        turn_map = {
            # Current: {Desired: [actions]}
            0: {1: [Action.LEFT.value, Action.LEFT.value],  # UP -> DOWN
                2: [Action.LEFT.value],                     # UP -> LEFT
                3: [Action.RIGHT.value]},                   # UP -> RIGHT
                
            1: {0: [Action.LEFT.value, Action.LEFT.value],  # DOWN -> UP
                2: [Action.RIGHT.value],                    # DOWN -> LEFT
                3: [Action.LEFT.value]},                    # DOWN -> RIGHT
                
            2: {0: [Action.RIGHT.value],                    # LEFT -> UP
                1: [Action.LEFT.value],                     # LEFT -> DOWN
                3: [Action.LEFT.value, Action.LEFT.value]}, # LEFT -> RIGHT
                
            3: {0: [Action.LEFT.value],                    # RIGHT -> UP
                1: [Action.RIGHT.value],                   # RIGHT -> DOWN
                2: [Action.LEFT.value, Action.LEFT.value]}  # RIGHT -> LEFT
        }
        
        if current == desired:
            return []
        return turn_map[current][desired]