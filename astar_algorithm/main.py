import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import gymnasium as gym
import rware
from rware.warehouse import RewardType, Warehouse
from shelf_movement import ShelfCarryingMovement
from shared_functions.warehouse_initializer import WarehouseInitializer
from shared_functions.metrics_tracker import MetricsTracker
from shared_functions.enums import Action

class WarehouseController:
    def __init__(self, env):
        self.env = env.unwrapped
        self.metrics = MetricsTracker() 
        
        # Initialize agent states
        self.agent_states = {}  # "seek_shelf", "deliver", "return_shelf"
        self.agent_targets = {}  # Current target for each agent
        self.last_positions = {}
        self.stuck_count = {}
        self.last_actions = {}
        self.agent_capacities = {}  # Track agent capacities
        self.shelf_weights = {}  # Track shelf weights
        self.reserved_shelves = {}  # Track which shelves are reserved by which agents
        self.shelf_reservation_time = {}  # Track how long a shelf has been reserved
        self.current_step = 0  
        self.low_battery_threshold = 20  # Percentage at which agent seeks charging
        self.critical_battery_threshold = 10  # Percentage where charging is mandatory
        self.charging_stations = self._identify_charging_stations()
        self.pre_charging_states = {}  # To remember what agents were doing before charging
        self.pre_charging_targets = {}  # To remember targets before charging
        self.pre_charging_shelf = {}    # To remember which shelf was being carried before charging
        self.pre_charging_data = {} # Stores dict of {state, targets, shelf_info}
        
        self.assigned_waiting_areas = {}  # New: track assigned waiting area
        self.in_recovery = {} 

        # Initialize agent-specific data
        for agent in self.env.agents:
            self.agent_states[agent.id] = "seek_shelf"
            self.agent_targets[agent.id] = None
            self.last_positions[agent.id] = (agent.x, agent.y)
            self.stuck_count[agent.id] = 0
            self.last_actions[agent.id] = []
            self.agent_capacities[agent.id] = agent.max_carry_weight  # Store capacity

        # Initialize shelf weights
        for shelf in self.env.request_queue:
            self.shelf_weights[shelf.id] = shelf.weight


    def _identify_charging_stations(self):
        """Identify charging station locations (assuming they're in the corners)"""
        grid_size = self.env.grid_size
        return [
            (0, 0),  # Top-left
            (grid_size[1]-1, 0),  # Top-right
            (0, grid_size[0]-1),  # Bottom-left
            (grid_size[1]-1, grid_size[0]-1)  # Bottom-right
        ]

    def _needs_charging(self, agent):
        """Check if agent needs to charge based on battery level"""
        battery_pct = (agent.battery_level / self.env.battery_capacity) * 100
        return battery_pct < self.low_battery_threshold


    def _get_closest_charging_station(self, agent):
        """Find the nearest available charging station with queuing logic"""
        current_pos = (int(agent.x), int(agent.y))
        
        # Check if agent is already at a charging station
        if current_pos in self.charging_stations:
            # print(f"Agent {agent.id} is already at charging station {current_pos}")
            return None  # Already at charging station
        
        # Get all stations sorted by distance (closest first)
        stations_sorted = sorted(
            self.charging_stations,
            key=lambda station: abs(station[0] - agent.x) + abs(station[1] - agent.y)
        )
        
        # Check each station in order of distance
        for station in stations_sorted:
            # Skip if this is the agent's current position
            if (int(agent.x), int(agent.y)) == station:
                continue
                
            dist = abs(station[0] - agent.x) + abs(station[1] - agent.y)
            charging_agents = []
            
            # Check which agents are at this station
            for other_agent in self.env.agents:
                if (int(other_agent.x), int(other_agent.y)) == station:
                    charging_agents.append(other_agent)
            
            # If station is empty, it's available
            if not charging_agents:
                # print(f"Agent {agent.id} found available charging station: {station} (Distance: {dist})")
                return station
            
            # If station has charging agents, check if any are nearly full
            for charging_agent in charging_agents:
                battery_pct = (charging_agent.battery_level / self.env.battery_capacity) * 100
                if battery_pct > 90:  # Agent is nearly full, will vacate soon
                    # print(f"Agent {agent.id} queuing at station {station} (Agent {charging_agent.id} is at {battery_pct:.1f}%)")
                    return station
        
        # If all stations are occupied, go to the closest one and wait in queue
        if stations_sorted:
            closest_station = stations_sorted[0]
            # print(f"Agent {agent.id} all stations occupied - going to closest ({closest_station}) to queue")
            return closest_station
        
        # print(f"Agent {agent.id} - No charging stations found!")
        return None

    def initialize_and_verify(self, env):
        """Initialize and verify all environment components"""
        initialization_data = WarehouseInitializer(env)
        initialization_data.initialize_all()
        self.grid_size = initialization_data.grid_size
        self.rows = initialization_data.rows
        self.cols = initialization_data.cols
        self.goal_locations = initialization_data.goal_locations
        self.shelf_locations = initialization_data.shelf_locations
        self.agent_locations = initialization_data.agent_locations
        self.shelf_memory = initialization_data.shelf_memory
        self.agent_states = initialization_data.agent_states
        self.shelf_memory = initialization_data.shelf_memory
        self.agent_targets = initialization_data.agent_targets
        self.last_positions = initialization_data.last_positions
        self.stuck_count = initialization_data.stuck_count
        self.last_actions = initialization_data.last_actions

        self.shelf_mover = ShelfCarryingMovement(env, self.agent_states)
        # self.movement_controller = MovementController(self)
        self.waiting_areas = self._identify_waiting_areas()  # New: identify waiting areas

    def _get_aligned_position(self, pos):
        """Ensure position is within grid bounds"""
        x = max(0, min(int(pos[0]), self.cols-1))
        y = max(0, min(int(pos[1]), self.rows-1))
        return (x, y)
    
    def _get_recovery_actions(self, agent):
        """More intelligent recovery considering carrying state"""
        recovery = []
        agent_id = agent.id
        is_carrying = agent.carrying_shelf is not None
        
        # Different patterns for carrying vs not carrying
        if is_carrying:
            # When carrying, prioritize backing up and turning
            recovery = [
                Action.LEFT.value,
                Action.LEFT.value,  # Turn 180
                Action.FORWARD.value,  # Back up
                Action.RIGHT.value,  # Turn right
                Action.FORWARD.value  # Move forward
            ]
        else:
            # When not carrying, can be more flexible
            pattern = (self.env.current_step + agent_id) % 3
            if pattern == 0:
                recovery = [Action.LEFT.value, Action.FORWARD.value]
            elif pattern == 1:
                recovery = [Action.RIGHT.value, Action.FORWARD.value]
            else:
                recovery = [
                    Action.LEFT.value, 
                    Action.LEFT.value,
                    Action.FORWARD.value
                ]
        
        # Mark current position as temporarily blocked
        self.shelf_helper.add_temporary_block((int(agent.x), int(agent.y)))
        
        return recovery
    
    def _can_carry_any_shelf(self, agent):
        """Check if there are any shelves in request queue that agent can carry"""
        for shelf in self.env.request_queue:
            # Skip if shelf is already being carried
            if any(a.carrying_shelf == shelf for a in self.env.agents):
                continue
                
            # Skip if shelf is reserved by another agent (unless reservation expired)
            if shelf.id in self.reserved_shelves:
                reserving_agent = self.reserved_shelves[shelf.id]
                reservation_time = self.shelf_reservation_time.get(shelf.id, 0)
                if reserving_agent != agent.id and (self.current_step - reservation_time) < 20:
                    continue
            
            # If we find at least one shelf the agent can carry, return True
            if shelf.weight <= agent.max_carry_weight:
                return True
        return False
    
    def _identify_waiting_areas(self):
        """Identify suitable waiting areas for idle agents.
        These are spaces that aren't shelf locations, goal locations, charging stations,
        and aren't adjacent to shelves in any of the 8 possible directions.
        """
        grid_size = self.env.grid_size
        rows, cols = grid_size
        
        # Start with all possible positions
        all_positions = [(x, y) for x in range(cols) for y in range(rows)]
        
        # Get all shelf positions
        shelf_positions = set((int(shelf.x), int(shelf.y)) for shelf in self.env.shelfs)
        
        # Remove occupied locations
        occupied = set()
        
        # Remove shelf locations
        occupied.update(shelf_positions)
        
        # Remove goal locations
        for goal in self.goal_locations:
            occupied.add(goal)
        
        # Remove charging stations
        for station in self.charging_stations:
            occupied.add(station)
        
        # Function to check if a position is adjacent to any shelf in 8 directions
        def is_adjacent_to_shelf(pos):
            x, y = pos
            # All 8 possible adjacent positions
            adjacent_positions = [
                (x-1, y),    # left
                (x+1, y),    # right
                (x, y-1),    # up
                (x, y+1),    # down
                (x-1, y-1),  # left-up
                (x+1, y-1),  # right-up
                (x-1, y+1),  # left-down
                (x+1, y+1)   # right-down
            ]
            return any(adj_pos in shelf_positions for adj_pos in adjacent_positions)
        
        # Available positions are those not occupied and not adjacent to shelves
        available = [
            pos for pos in all_positions 
            if pos not in occupied and not is_adjacent_to_shelf(pos)
        ]
        
        # If no positions are available, use corners as a fallback
        if not available:
            corners = [
                (0, 0),            # Top-left
                (cols-1, 0),       # Top-right
                (0, rows-1),       # Bottom-left
                (cols-1, rows-1)   # Bottom-right
            ]
            # Filter out corners that are adjacent to shelves
            available = [pos for pos in corners if not is_adjacent_to_shelf(pos)]
            
            # If still no positions available, just return all corners
            if not available:
                return corners
        
        return available

    # Add this method to assign a waiting area to an idle agent:
    def _get_waiting_area(self, agent):
        """Get suitable waiting area for an idle agent"""
        if not hasattr(self, 'waiting_areas'):
            self.waiting_areas = self._identify_waiting_areas()
        
        if not hasattr(self, 'assigned_waiting_areas'):
            self.assigned_waiting_areas = {}
        
        # If agent already has an assigned area, keep using it
        if agent.id in self.assigned_waiting_areas:
            return self.assigned_waiting_areas[agent.id]
        
        # Find which waiting areas are already assigned
        occupied_areas = set(self.assigned_waiting_areas.values())
        
        # Find an unoccupied waiting area
        available_areas = [area for area in self.waiting_areas if area not in occupied_areas]
        
        if available_areas:
            # Assign the closest available area
            current_pos = (int(agent.x), int(agent.y))
            closest_area = min(available_areas, 
                            key=lambda pos: abs(pos[0] - current_pos[0]) + abs(pos[1] - current_pos[1]))
            self.assigned_waiting_areas[agent.id] = closest_area
            return closest_area
        else:
            # If all areas are occupied, just pick one based on agent ID to ensure consistent assignment
            area_idx = agent.id % len(self.waiting_areas)
            area = self.waiting_areas[area_idx]
            self.assigned_waiting_areas[agent.id] = area
            return area
        
    def get_actions(self):
        

        current_step = self.current_step  # Use our controller's step counter
        expired = [shelf_id for shelf_id, step in self.shelf_reservation_time.items() 
                if (current_step - step) >= 20]  # 20 step timeout
        for shelf_id in expired:
            del self.reserved_shelves[shelf_id]
            del self.shelf_reservation_time[shelf_id]
        actions = []

        self.metrics.record_total_steps()

        for agent in self.env.agents:
            self.metrics.record_movement(agent.id)
            # Check if agent has no shelf to carry (only if not already in another state)
            if (self.agent_states[agent.id] not in ["deliver", "return_shelf", "charging"] and 
                not self._can_carry_any_shelf(agent)):
                self.agent_states[agent.id] = "no_shelf_to_carry"
                self.agent_targets[agent.id] = None
                self.last_actions[agent.id] = []
                # print(f"Agent {agent.id} has no shelf it can carry (max capacity: {agent.max_carry_weight})")
            
            # Replace it with:
            if self.agent_states[agent.id] == "no_shelf_to_carry":
                if self._can_carry_any_shelf(agent):
                    # Found a shelf we can carry - switch back to seek_shelf
                    self.agent_states[agent.id] = "seek_shelf"
                else:
                    # Move to waiting area instead of just stopping
                    waiting_area = self._get_waiting_area(agent)
                    current_pos = (int(agent.x), int(agent.y))
                    
                    # If already at waiting area, stay there
                    if current_pos == waiting_area:
                        actions.append(Action.NOOP.value)
                    else:
                        # Move toward waiting area
                        if not self.agent_targets.get(agent.id):
                            self.agent_targets[agent.id] = {'position': waiting_area}
                        
                        movement_sequence = self.shelf_mover.calculate_movement(agent, waiting_area)
                        if movement_sequence:
                            self.last_actions[agent.id] = movement_sequence[1:]
                            actions.append(movement_sequence[0])
                        else:
                            actions.append(Action.NOOP.value)
                    continue
            battery_pct = (agent.battery_level / self.env.battery_capacity) * 100

            # Add this check for battery failure at the beginning of your per-agent loop in get_actions()
            if agent.battery_level <= 0:  # Battery completely depleted
                self.metrics.record_battery_failure(agent.id)
                actions.append(Action.NOOP.value)  # Agent can't do anything when battery is dead
                continue

            # --- Battery Check Logic ---
            # Emergency charging if critical
            if battery_pct < self.critical_battery_threshold and not agent.is_charging:
                self.metrics.record_critical_battery(agent.id, battery_pct)  # Add this line
                if self.agent_states[agent.id] != "charging":  # Only store state if we're not already charging
                    if (agent.id in self.agent_targets and 
                        self.agent_targets[agent.id] is not None):
                        self._store_pre_charging_state(agent)
                closest_station = self._get_closest_charging_station(agent)
                if closest_station:
                    print(f"Agent {agent.id} in CRITICAL battery ({battery_pct:.1f}%) - going to charge")
                    self.metrics.record_charging_start(agent.id)
                    self.agent_states[agent.id] = "charging"
                    self.agent_targets[agent.id] = {'position': closest_station}
                    self.last_actions[agent.id] = []
            
            # Normal charging when low
            if (self._needs_charging(agent) and not agent.is_charging and self.agent_states[agent.id] != "charging"):
                self.metrics.record_low_battery(agent.id, battery_pct)  # Add this line
                if self.agent_states[agent.id] != "charging":  # Only store state if we're not already charging
                    self._store_pre_charging_state(agent)
                closest_station = self._get_closest_charging_station(agent)
                if closest_station:
                    # print(f"Agent {agent.id} low battery ({battery_pct:.1f}%) - going to charge")
                    self.metrics.record_charging_start(agent.id)
                    self.agent_states[agent.id] = "charging"
                    self.agent_targets[agent.id] = {'position': closest_station}
                    self.last_actions[agent.id] = []
            
            current_pos = (int(agent.x), int(agent.y))

            # Check if agent is stuck (same position for multiple steps)
            if self.last_actions.get(agent.id) and current_pos == self.last_positions.get(agent.id) and self.last_actions[agent.id][0] == Action.FORWARD.value:
                self.stuck_count[agent.id] += 1
                
            else:
                self.stuck_count[agent.id] = 0
                # self.last_positions[agent.id] = current_pos

            if (self.stuck_count.get(agent.id, 0) > 1 and 
                not agent.is_charging and 
                not self.agent_states[agent.id] == "no_shelf_to_carry") and not self.in_recovery.get(agent.id, False):
                print(f"Agent {agent.id} stuck at {current_pos}, initiating recovery...")
                # self.stuck_count[agent.id] = 0
                # Clear current action queue
                self.last_actions[agent.id] = []
                self.metrics.record_collision(agent.id)  # Record collision
                
                self.in_recovery[agent.id] = True
            
            if self.in_recovery.get(agent.id) is True:
                self.metrics.record_recovery_step(agent.id)
                # self.last_positions[agent.id] = current_pos
            
            if current_pos != self.last_positions.get(agent.id) and self.last_actions.get(agent.id) and self.last_actions[agent.id][0] == Action.FORWARD.value and self.in_recovery.get(agent.id) is True:
                # self.last_positions[agent.id] = current_pos
                print(f"Agent {agent.id} recovered from stuck state at {current_pos}")
                self.stuck_count[agent.id] = 0
                self.in_recovery[agent.id] = False
                self.metrics.record_recovery_complete(agent.id)
                
            self.last_positions[agent.id] = current_pos    
            # Rest of your existing action selection logic...
            queued_actions = self.last_actions.get(agent.id, [])
            
            if queued_actions:
                action = queued_actions.pop(0)
                actions.append(action)
                continue
            
            # Default to NOOP if no other action is determined
            action = Action.NOOP.value
            
            if self.agent_states[agent.id] == "charging":
                if agent.is_charging:
                    # Already charging
                    if agent.battery_level >= self.env.battery_capacity:  
                        self._restore_pre_charging_state(agent)
                        self.metrics.record_charging_end(agent.id)
                    else:
                        actions.append(Action.NOOP.value)
                        continue
                else:
                    # Moving to charging station
                    station_pos = self.agent_targets[agent.id]['position']
                    current_pos = (int(agent.x), int(agent.y))
                    
                    # If reached station, start charging if available
                    if current_pos == station_pos:
                        # Check if station is available (no one charging or someone nearly full)
                        station_available = True
                        for other_agent in self.env.agents:
                            if (other_agent != agent and 
                                (int(other_agent.x), int(other_agent.y)) == station_pos and
                                other_agent.is_charging):
                                battery_pct = (other_agent.battery_level / self.env.battery_capacity) * 100
                                if battery_pct < 90:  # Other agent still needs significant charging
                                    station_available = False
                                    break
                        
                        if station_available:
                            actions.append(Action.NOOP.value)  # Will start charging
                        else:
                            # Wait in queue (small movements to avoid blocking)
                            if len(self.last_actions.get(agent.id, [])) < 2:
                                # Create small back-and-forth pattern
                                self.last_actions[agent.id] = [
                                    Action.LEFT.value,
                                    Action.RIGHT.value
                                ]
                            action = self.last_actions[agent.id].pop(0)
                            actions.append(action)
                    else:
                        # Continue moving to station
                        movement_sequence = self.shelf_mover.calculate_movement(agent, station_pos)
                        if movement_sequence:
                            self.last_actions[agent.id] = movement_sequence[1:]
                            actions.append(movement_sequence[0])
                        else:
                            actions.append(Action.NOOP.value)
                    continue
            
            # State: Seeking a shelf to pick up
            if self.agent_states[agent.id] == "seek_shelf":
                if agent.id in self.assigned_waiting_areas:
                    del self.assigned_waiting_areas[agent.id]
                # Replan if no current path or target not reached
                if not self.last_actions.get(agent.id) or \
                (int(agent.x), int(agent.y)) != (int(self.agent_targets[agent.id]['position'][0]), 
                                                int(self.agent_targets[agent.id]['position'][1])):
                    
                    # Find closest available shelf
                    closest_shelf = None
                    min_dist = float('inf')
                    
                    for shelf in self.env.request_queue:
                        # Skip if shelf is already being carried
                        if any(a.carrying_shelf == shelf for a in self.env.agents):
                            carrying_agent = next(a for a in self.env.agents if a.carrying_shelf == shelf)
                            continue
                        
                        # Skip if shelf is reserved by another agent (unless reservation expired)
                        if shelf.id in self.reserved_shelves:
                            reserving_agent = self.reserved_shelves[shelf.id]
                            reservation_time = self.shelf_reservation_time.get(shelf.id, 0)
                            
                            # If reserved by another agent and not expired, skip
                            if reserving_agent != agent.id and (self.current_step - reservation_time) < 20:  # 20 step timeout
                                # print(f"  Shelf {shelf.id} reserved by agent {reserving_agent}")
                                continue
                        

                        # Check if agent can carry this shelf
                        if shelf.weight > agent.max_carry_weight:
                            # print(f"  Agent {agent.id} cannot carry shelf {shelf.id} (weight {shelf.weight} > capacity {agent.max_carry_weight})")
                            continue
                        
                        shelf_pos = self._get_aligned_position((shelf.x, shelf.y))
                        dist = abs(shelf_pos[0]-agent.x) + abs(shelf_pos[1]-agent.y)
                        if dist < min_dist:
                            min_dist = dist
                            closest_shelf = shelf
                    
                    if closest_shelf:
                        # Reserve this shelf for current agent
                        self.reserved_shelves[closest_shelf.id] = agent.id
                        self.shelf_reservation_time[closest_shelf.id] = self.current_step
                        
                        target_pos = self._get_aligned_position((closest_shelf.x, closest_shelf.y))
                        self.agent_targets[agent.id] = {
                            'id': closest_shelf.id,
                            'position': target_pos
                        }
                        

                        if closest_shelf and not (int(agent.x), int(agent.y)) == (int(closest_shelf.x), int(closest_shelf.y)):
                            self.metrics.record_task_start(agent.id, closest_shelf.id)
                        
                        
                        # Calculate movement sequence
                        movement_sequence = self.shelf_mover.calculate_movement(agent, target_pos)
                        
                        if movement_sequence:
                            self.last_actions[agent.id] = movement_sequence[1:]
                            action = movement_sequence[0]

                    
                
                # Check if reached shelf position
                if closest_shelf and (int(agent.x), int(agent.y)) == (int(closest_shelf.x), int(closest_shelf.y)):
                    # Verify weight constraint again before picking up
                    if closest_shelf.weight <= agent.max_carry_weight:
                        action = Action.TOGGLE_LOAD.value
                        self.agent_states[agent.id] = "deliver"
                        self.last_actions[agent.id] = []
                        
                        # Clear the reservation since we're now carrying the shelf
                        if closest_shelf.id in self.reserved_shelves:
                            del self.reserved_shelves[closest_shelf.id]
                            del self.shelf_reservation_time[closest_shelf.id]
                        
                        # print(f"Agent {agent.id} picked up shelf {closest_shelf.id} (weight: {closest_shelf.weight})")
                    else:
                        # print(f"ERROR: Agent {agent.id} tried to pick up shelf {closest_shelf.id} that's too heavy!")
                        # Release reservation if can't pick up
                        self.metrics.record_overcapacity_attempt(agent.id, closest_shelf.id, closest_shelf.weight, agent.max_carry_weight)
                        if closest_shelf.id in self.reserved_shelves:
                            del self.reserved_shelves[closest_shelf.id]
                            del self.shelf_reservation_time[closest_shelf.id]
            
            # State: Delivering shelf to goal location
            elif self.agent_states[agent.id] == "deliver" and agent.carrying_shelf:
                closest_goal = min(self.goal_locations, 
                                key=lambda g: abs(g[0]-agent.x) + abs(g[1]-agent.y))
                
                if agent.id in self.agent_targets:
                    self.agent_targets[agent.id]['position'] = closest_goal
                
                movement_sequence = self.shelf_mover.calculate_movement(agent, closest_goal)
                
                if movement_sequence:
                    self.last_actions[agent.id] = movement_sequence[1:]
                    action = movement_sequence[0]
                
                # Check if reached goal position
                if (int(agent.x), int(agent.y)) == closest_goal:
                    action = Action.TOGGLE_LOAD.value
                    self.agent_states[agent.id] = "return_shelf"
                    self.last_actions[agent.id] = []
                    # print(f"Agent {agent.id} delivered shelf {agent.carrying_shelf.id}")
                    
            
            # State: Returning shelf to original position
            elif self.agent_states[agent.id] == "return_shelf":
                if agent.carrying_shelf and agent.carrying_shelf.id in self.shelf_memory:
                    original_pos = self.shelf_memory[agent.carrying_shelf.id]
                    
                    if agent.id in self.agent_targets:
                        self.agent_targets[agent.id]['position'] = original_pos
                    
                    movement_sequence = self.shelf_mover.calculate_movement(agent, original_pos)
                    
                    if movement_sequence:
                        self.last_actions[agent.id] = movement_sequence[1:]
                        action = movement_sequence[0]
                    
                    # Check if reached original position
                    if (int(agent.x), int(agent.y)) == original_pos:
                        action = Action.TOGGLE_LOAD.value
                        self.agent_states[agent.id] = "seek_shelf"
                        self.agent_targets[agent.id] = None
                        self.last_actions[agent.id] = []
                        # print(f"Agent {agent.id} returned shelf {agent.carrying_shelf.id}")
                        self.metrics.record_task_completion(agent.id, agent.carrying_shelf.id)
                else:
                    self.agent_states[agent.id] = "seek_shelf"
                    self.agent_targets[agent.id] = None
                    self.last_actions[agent.id] = []
            

            
            actions.append(action)
        
            
            # print(f"Agent {agent.id} - Action: {action}, State: {self.agent_states[agent.id]}, Targets: {self.agent_targets[agent.id]}")
            self.metrics.record_step_completion()
        
        
        return actions
    
    def _store_pre_charging_state(self, agent):
        """Store complete task context before charging"""
        if agent.id not in self.agent_targets or self.agent_targets[agent.id] is None:
            self.pre_charging_data[agent.id] = {
                'state': self.agent_states[agent.id],
                'targets': None,
                'shelf_id': agent.carrying_shelf.id if agent.carrying_shelf else None,
                'shelf_origin': None
            }
            return

        storage = {
            'state': self.agent_states[agent.id],
            'targets': None,
            'shelf_id': agent.carrying_shelf.id if agent.carrying_shelf else None,
            'shelf_origin': None
        }

        # Store state-specific targets
        if self.agent_states[agent.id] == "seek_shelf":
            if self.agent_targets.get(agent.id):
                storage['targets'] = {
                    'shelf_location': self.agent_targets[agent.id]['position'],
                    'shelf_id': self.agent_targets[agent.id].get('id')  # Safe get with default
                }
                
        elif self.agent_states[agent.id] == "deliver":
            if agent.carrying_shelf:
                storage['targets'] = {
                    'goal_location': min(self.goal_locations, 
                                    key=lambda g: abs(g[0]-agent.x) + abs(g[1]-agent.y)),
                    'shelf_id': agent.carrying_shelf.id
                }
                
        elif self.agent_states[agent.id] == "return_shelf":
            if agent.carrying_shelf and agent.carrying_shelf.id in self.shelf_memory:
                storage['targets'] = {
                    'shelf_origin': self.shelf_memory[agent.carrying_shelf.id],
                    'shelf_id': agent.carrying_shelf.id
                }
                storage['shelf_origin'] = self.shelf_memory[agent.carrying_shelf.id]

        self.pre_charging_data[agent.id] = storage

    def _restore_pre_charging_state(self, agent):
        """Restore complete task context after charging"""
        if agent.id not in self.pre_charging_data:
            self.agent_states[agent.id] = "seek_shelf"
            return

        data = self.pre_charging_data[agent.id]
        self.agent_states[agent.id] = data['state']
        
        # Clear any existing actions
        self.last_actions[agent.id] = []
        
        # State-specific restoration
        if data['state'] == "seek_shelf" and data['targets']:
            shelf_id = data['targets']['shelf_id']
            # Verify shelf still exists and is available
            shelf = next((s for s in self.env.request_queue if s.id == shelf_id), None)
            if shelf and not any(a.carrying_shelf == shelf for a in self.env.agents):
                target_pos = data['targets']['shelf_location']
                self.agent_targets[agent.id] = {
                    'id': shelf_id,
                    'position': target_pos
                }
                # Calculate new movement sequence
                movement_sequence = self.shelf_mover.calculate_movement(agent, target_pos)
                if movement_sequence:
                    self.last_actions[agent.id] = movement_sequence[1:]
                    
        elif data['state'] == "deliver" and data['targets']:
            shelf_id = data['targets']['shelf_id']
            shelf = next((s for s in self.env.shelfs if s.id == shelf_id), None)
            if shelf and any(a.carrying_shelf == shelf for a in self.env.agents if a.id == agent.id):
                target_pos = data['targets']['goal_location']
                self.agent_targets[agent.id] = {
                    'position': target_pos
                }
                # Calculate new movement sequence
                movement_sequence = self.shelf_mover.calculate_movement(agent, target_pos)
                if movement_sequence:
                    self.last_actions[agent.id] = movement_sequence[1:]
                    
        elif data['state'] == "return_shelf" and data['targets']:
            shelf_id = data['targets']['shelf_id']
            shelf = next((s for s in self.env.shelfs if s.id == shelf_id), None)
            if shelf and any(a.carrying_shelf == shelf for a in self.env.agents if a.id == agent.id):
                target_pos = data['targets']['shelf_origin']
                self.agent_targets[agent.id] = {
                    'position': target_pos
                }
                # Calculate new movement sequence
                movement_sequence = self.shelf_mover.calculate_movement(agent, target_pos)
                if movement_sequence:
                    self.last_actions[agent.id] = movement_sequence[1:]

        # print(f"Restored Agent {agent.id} to {data['state']} with targets: {data['targets']}")
        del self.pre_charging_data[agent.id]
    
    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

def run_simulation():
    env = gym.make("rware-easy-1ag-v2")
    
    # Initialize the controller
    controller = WarehouseController(env)
    obs, info = env.reset()
    controller.initialize_and_verify(env)

    max_deliveries = 5
    step = 0  # Initialize step counter

    try:
        while True:  # Infinite loop until break condition
            
            # Check completion condition
            if controller.metrics.get_successful_tasks().count >= max_deliveries:
                print(f"\nAll {max_deliveries} deliveries completed! Simulation ending.")
                break
                
            # Get and execute actions
            actions = controller.get_actions()
            obs, rewards, done, truncated, info = env.step(actions)
            env.render()

            # Optional: Slow down the simulation
            time.sleep(0.1)

            controller.current_step = step
            step += 1  # Increment step counter
            
    except KeyboardInterrupt:
        delivered = controller.metrics.get_successful_tasks().count
        print(f"\nSimulation stopped. Delivered {delivered}/{max_deliveries} shelves")
    finally:
        controller.metrics.print_summary()
        env.close()

if __name__ == "__main__":
    run_simulation()