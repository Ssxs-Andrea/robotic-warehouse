import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
        # Visualization and initial setup (keep your existing code)
        # self.shelf_helper.print_warehouse_map(agent)
        current_pos = (int(agent.x), int(agent.y))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        
        if self.debug:
            print(f"\n=== Starting Movement Calculation ===")
            print(f"Agent Start: {current_pos}, Facing: {self.direction_map[agent.dir.value]}")
            print(f"Goal Position: {goal_pos}")
            print(f"Shelf Size: {shelf_width}x{shelf_height}")

        # Get blocked positions (keep your existing code)
        blocked = self.shelf_helper.get_blocked_positions(agent, shelf_width, shelf_height)
        # self.shelf_helper.print_warehouse_map(agent)
        if self.debug:
            print(f"Blocked positions count: {len(blocked)}")
            if len(blocked) < 20:
                print(f"Blocked positions: {sorted(blocked)}")

        # Goal validation (keep your existing code)
        if not self._is_position_valid(goal_pos):
            if self.debug:
                print(f"❌ Invalid goal position: {goal_pos} (Grid size: {self.rows}x{self.cols})")
            return []
        elif goal_pos in blocked:
            if self.debug:
                print(f"❌ Goal position blocked: {goal_pos}")
            return []
        else:
            if self.debug:
                print(f"✅ Goal position valid and not blocked")

        # Enhanced direct path checking with full validation
        direct_clear = self._is_direct_path_clear(current_pos, goal_pos, blocked, shelf_width, shelf_height)
        if self.debug:
            print(f"Direct path clear: {direct_clear}")
        
        if direct_clear:
            actions = self._direct_movement(agent, goal_pos)
            if self.debug:
                print(f"Direct movement actions: {[self.action_names[a] for a in actions]}")
            
            # NEW: Validate the entire action sequence
            if self._validate_action_sequence(agent, actions, blocked):
                if self.debug:
                    print("✅ Direct path validated")
                return actions
            else:
                if self.debug:
                    print("❌ Direct path invalid - would collide")
                direct_clear = False  # Force fallback to A*

        # Fall back to pathfinding (keep your existing code but add validation)
        if not direct_clear:
            if self.debug:
                print("Attempting A* pathfinding...")
            actions = self._pathfind_movement(agent, goal_pos, blocked)
            # print(actions)
            if actions:
                # NEW: Validate A* path
                if self._validate_action_sequence(agent, actions, blocked):
                    if self.debug:
                        print(f"✅ A* found valid path with {len(actions)} actions")
                    return actions
                else:
                    if self.debug:
                        print("❌ A* path invalid - would collide")
            
            if self.debug:
                print("❌ A* failed to find valid path")
            
        
        return []


    def _validate_action_sequence(self, agent, actions, blocked):
        """Proper validation that tracks position AND direction after each action"""
        x, y = int(agent.x), int(agent.y)
        direction = agent.dir.value
        
        for action in actions:
            # print(f"Current: ({x},{y}) facing {direction} | Action: {action}")
            
            if action == Action.FORWARD.value:
                # Calculate new position based on CURRENT direction
                new_x, new_y = x, y
                if direction == 0: new_y -= 1    # UP
                elif direction == 1: new_y += 1  # DOWN
                elif direction == 2: new_x -= 1  # LEFT
                elif direction == 3: new_x += 1  # RIGHT
                
                # Check new position
                if not self._is_position_valid((new_x, new_y)) or (new_x, new_y) in blocked:
                    # print(f"Collision at ({new_x},{new_y}) - Blocked: {(new_x,new_y) in blocked}")
                    return False
                    
                # Update position after successful move
                x, y = new_x, new_y
                
            elif action in (Action.LEFT.value, Action.RIGHT.value):
                # First verify we're not blocked at current position
                # if (x, y) in blocked:
                #     print(f"Turn blocked at current position ({x},{y})")
                #     return False
                    
                # Update direction
                if action == Action.LEFT.value:
                    direction = {0:2, 1:3, 2:1, 3:0}[direction]
                else:
                    direction = {0:3, 1:2, 2:0, 3:1}[direction]
                
                # print(f"New direction: {direction}")
        
        return True


    def _is_position_valid(self, pos):
        """Check if position is within bounds"""
        x, y = pos
        return 0 <= x < self.cols and 0 <= y < self.rows

    def _is_direct_path_clear(self, start, end, blocked, shelf_width=1, shelf_height=1):
        """More thorough path clearance checking"""
        x1, y1 = start
        x2, y2 = end
        
        # Check each step along the path
        if x1 != x2:  # Horizontal movement
            step = 1 if x2 > x1 else -1
            for x in range(x1, x2, step):
                if (x, y1) in blocked:
                    return False
                    
        if y1 != y2:  # Vertical movement
            step = 1 if y2 > y1 else -1
            for y in range(y1, y2, step):
                if (x2, y) in blocked:
                    return False
                    
        return True

    def _direct_movement(self, agent, goal_pos):
        """Calculate direct Manhattan movement sequence"""
        current_x, current_y = int(agent.x), int(agent.y)
        target_x, target_y = goal_pos
        actions = []
        
        current_dir = agent.dir.value
        current_dir_name = self.direction_map[current_dir]
        
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Determine primary movement axis
        primary_axis = 'x' if abs(dx) > abs(dy) else 'y'
        
        if primary_axis == 'x':
            if dx != 0:
                desired_dir = 3 if dx > 0 else 2  # RIGHT or LEFT
                if current_dir != desired_dir:
                    actions += self._turn_to_face(current_dir, desired_dir)
                    current_dir = desired_dir
                
                for _ in range(abs(dx)):
                    actions.append(Action.FORWARD.value)
            
            if dy != 0:
                desired_dir = 1 if dy > 0 else 0  # DOWN or UP
                if current_dir != desired_dir:
                    actions += self._turn_to_face(current_dir, desired_dir)
                    current_dir = desired_dir
                
                for _ in range(abs(dy)):
                    actions.append(Action.FORWARD.value)
        else:
            if dy != 0:
                desired_dir = 1 if dy > 0 else 0  # DOWN or UP
                if current_dir != desired_dir:
                    actions += self._turn_to_face(current_dir, desired_dir)
                    current_dir = desired_dir
                
                for _ in range(abs(dy)):
                    actions.append(Action.FORWARD.value)
            
            if dx != 0:
                desired_dir = 3 if dx > 0 else 2  # RIGHT or LEFT
                if current_dir != desired_dir:
                    actions += self._turn_to_face(current_dir, desired_dir)
                    current_dir = desired_dir
                
                for _ in range(abs(dx)):
                    actions.append(Action.FORWARD.value)
        
        return actions

    def _pathfind_movement(self, agent, goal_pos, blocked):
        """A* pathfinding implementation"""

        
        start_pos = (int(agent.x), int(agent.y))
        start_dir = agent.dir.value
        
        # A* algorithm setup
        open_set = []
        closed_set = set()
        came_from = {}
        
        # g_score[node] = cost from start to node
        g_score = {start_pos: 0}
        
        # f_score[node] = g_score[node] + h(node)
        f_score = {start_pos: self._heuristic(start_pos, goal_pos)}
        
        open_set.append(start_pos)
        
        while open_set:
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            
            if current == goal_pos:
                # Reconstruct path
                path = []
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                if not path:
                    return []
                
                # Convert path to actions
                return self._convert_path_to_actions(agent, path)
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set or neighbor in blocked:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal_pos)
        
        return []  # No path found

    def _heuristic(self, pos, goal):
        """Manhattan distance heuristic for A*"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

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
    