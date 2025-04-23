import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import deque
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
        self.current_step = 0  
        self.in_recovery = {} 
        
        self.no_movement_steps = 0
        self.unreachable_shelves = set()  # Track shelves that can't be carried
        self.all_stuck_counter = 0  # Initialize counter for all agents stuck

        # Initialize agent-specific data
        for agent in self.env.agents:
            self._init_agent_data(agent)

        # Initialize shelf weights
        for shelf in self.env.request_queue:
            self.shelf_weights[shelf.id] = shelf.weight

    def _all_agents_dead(self):
        """Check if all agents have depleted their batteries and aren't charging"""
        return all(agent.battery_level <= 0 and not agent.is_charging 
                for agent in self.env.agents)

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

    def _get_aligned_position(self, pos):
        """Ensure position is within grid bounds"""
        x = max(0, min(int(pos[0]), self.cols-1))
        y = max(0, min(int(pos[1]), self.rows-1))
        return (x, y)
    
    def _get_recovery_actions(self, agent):
        """Basic recovery actions when stuck"""
        recovery = []
        agent_id = agent.id
        is_carrying = agent.carrying_shelf is not None
        
        if is_carrying:
            recovery = [
                Action.LEFT.value,
                Action.LEFT.value,
                Action.FORWARD.value,
                Action.RIGHT.value,
                Action.FORWARD.value
            ]
        else:
            pattern = (self.env.current_step + agent_id) % 3
            if pattern == 0:
                recovery = [Action.LEFT.value, Action.FORWARD.value]
            elif pattern == 1:
                recovery = [Action.RIGHT.value, Action.FORWARD.value]
            else:
                recovery = [Action.LEFT.value, Action.LEFT.value, Action.FORWARD.value]
        
        return recovery

    def _init_agent_data(self, agent):
        """Initialize all data structures for a new agent"""
        self.agent_states[agent.id] = "seek_shelf"
        self.agent_targets[agent.id] = None
        self.last_positions[agent.id] = (agent.x, agent.y)
        self.stuck_count[agent.id] = 0
        self.last_actions[agent.id] = []
        self.agent_capacities[agent.id] = agent.max_carry_weight
        # Define the unreachable_shelves set for this agent
        agent.unreachable_shelves = set()

    def _can_carry_any_shelf(self, agent):
        """Check if there are any shelves this agent could potentially carry"""
        # First ensure the agent has the unreachable_shelves set
        if not hasattr(agent, 'unreachable_shelves'):
            agent.unreachable_shelves = set()
            
        return any(
            not any(a.carrying_shelf == shelf for a in self.env.agents) and
            shelf.id not in agent.unreachable_shelves
            for shelf in self.env.request_queue
        )
    
    def _all_agents_stuck(self):
        """Check if all agents are currently stuck"""
        return all(self.stuck_count.get(agent.id, 0) > 1 for agent in self.env.agents)
            
    def get_actions(self):
        any_movement = any(
            (int(agent.x), int(agent.y)) != self.last_positions.get(agent.id, (-1, -1))
            for agent in self.env.agents
            if agent.battery_level > 0 and not agent.is_charging and self.agent_states[agent.id] != "no_shelf_to_carry"
        )
        
        if any_movement:
            self.no_movement_steps = 0
        else:
            self.no_movement_steps += 1
        actions = []

        self.metrics.record_total_steps()

        if self._all_agents_stuck():
            self.all_stuck_counter += 1
        else:
            self.all_stuck_counter = 0  # Reset if at least one agent is moving

        for agent in self.env.agents:
            self.metrics.record_movement(agent.id)
            
            # Check if agent has no shelf to carry
            if (self.agent_states[agent.id] not in ["deliver", "return_shelf"] and 
                not self._can_carry_any_shelf(agent)):
                self.agent_states[agent.id] = "no_shelf_to_carry"
                self.agent_targets[agent.id] = None
                self.last_actions[agent.id] = []
            
            if self.agent_states[agent.id] == "no_shelf_to_carry":
                if self._can_carry_any_shelf(agent):
                    self.agent_states[agent.id] = "seek_shelf"
                else:
                    actions.append(Action.NOOP.value)
                    continue
            
            current_pos = (int(agent.x), int(agent.y))

            # Check if agent is stuck
            if (self.last_actions.get(agent.id) and 
                current_pos == self.last_positions.get(agent.id) and 
                self.last_actions[agent.id][0] == Action.FORWARD.value):
                self.stuck_count[agent.id] += 1
            else:
                self.stuck_count[agent.id] = 0

            if (self.stuck_count.get(agent.id, 0) > 1 and 
                not self.agent_states[agent.id] == "no_shelf_to_carry" and 
                not self.in_recovery.get(agent.id, False)):
                print(f"Agent {agent.id} stuck at {current_pos}, initiating recovery...")
                self.metrics.record_collision(agent.id)
                self.in_recovery[agent.id] = True
            
            if self.in_recovery.get(agent.id) is True:
                self.metrics.record_recovery_step(agent.id)
            
            if (current_pos != self.last_positions.get(agent.id) and 
                self.last_actions.get(agent.id) and 
                self.last_actions[agent.id][0] == Action.FORWARD.value and 
                self.in_recovery.get(agent.id) is True):
                print(f"Agent {agent.id} recovered from stuck state at {current_pos}")
                self.stuck_count[agent.id] = 0
                self.in_recovery[agent.id] = False
                self.metrics.record_recovery_complete(agent.id)
                
            self.last_positions[agent.id] = current_pos    
            
            queued_actions = self.last_actions.get(agent.id, [])
            
            if queued_actions:
                action = queued_actions.pop(0)
                actions.append(action)
                continue
            
            # Default to NOOP if no other action is determined
            action = Action.NOOP.value
        
            # State: Seeking a shelf to pick up
            if self.agent_states[agent.id] == "seek_shelf":
                if not self.last_actions.get(agent.id) or \
                   (int(agent.x), int(agent.y)) != (int(self.agent_targets[agent.id]['position'][0]), 
                                                  int(self.agent_targets[agent.id]['position'][1])):

                    closest_shelf = None
                    min_dist = float('inf')
                    
                    for shelf in self.env.request_queue:
                        # Skip if shelf is being carried or marked unreachable by this agent
                        if (any(a.carrying_shelf == shelf for a in self.env.agents) or
                            shelf.id in agent.unreachable_shelves):
                            continue
                            
                        shelf_pos = self._get_aligned_position((shelf.x, shelf.y))
                        dist = abs(shelf_pos[0]-agent.x) + abs(shelf_pos[1]-agent.y)
                        if dist < min_dist:
                            min_dist = dist
                            closest_shelf = shelf
                    
                    if closest_shelf:
                        target_pos = self._get_aligned_position((closest_shelf.x, closest_shelf.y))
                        self.agent_targets[agent.id] = {
                            'id': closest_shelf.id,
                            'position': target_pos
                        }

                        if closest_shelf and not (int(agent.x), int(agent.y)) == (int(closest_shelf.x), int(closest_shelf.y)):
                            self.metrics.record_task_start(agent.id, closest_shelf.id)
                        
                        movement_sequence = self.shelf_mover.calculate_movement(agent, target_pos)
                        
                        if movement_sequence:
                            self.last_actions[agent.id] = movement_sequence[1:]
                            action = movement_sequence[0]
                
                # When reached shelf position - check weight for THIS agent
                if closest_shelf and (int(agent.x), int(agent.y)) == (int(closest_shelf.x), int(closest_shelf.y)):
                    if closest_shelf.weight <= agent.max_carry_weight:
                        action = Action.TOGGLE_LOAD.value
                        self.agent_states[agent.id] = "deliver"
                        self.last_actions[agent.id] = []
                        # Clear from this agent's unreachable list
                        agent.unreachable_shelves.discard(closest_shelf.id)
                    else:
                        # Add to THIS AGENT'S unreachable list only
                        agent.unreachable_shelves.add(closest_shelf.id)
                        self.agent_targets[agent.id] = None
                        self.last_actions[agent.id] = []
                        action = Action.NOOP.value
                        self.metrics.record_overcapacity_attempt(
                            agent.id, closest_shelf.id, 
                            closest_shelf.weight, agent.max_carry_weight
                        )
            
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
                
                if (int(agent.x), int(agent.y)) == closest_goal:
                    action = Action.TOGGLE_LOAD.value
                    self.agent_states[agent.id] = "return_shelf"
                    self.last_actions[agent.id] = []
                    
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
                    
                    if (int(agent.x), int(agent.y)) == original_pos:
                        action = Action.TOGGLE_LOAD.value
                        self.agent_states[agent.id] = "seek_shelf"
                        self.agent_targets[agent.id] = None
                        self.last_actions[agent.id] = []
                        self.metrics.record_task_completion(agent.id, agent.carrying_shelf.id)
                else:
                    self.agent_states[agent.id] = "seek_shelf"
                    self.agent_targets[agent.id] = None
                    self.last_actions[agent.id] = []
            actions.append(action)
        
            self.metrics.record_step_completion()
        
        return actions
    
    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None


def run_simulation():
    env = gym.make("rware-easy-1ag-v2")
    env = env.unwrapped
    controller = WarehouseController(env)
    obs, info = env.reset()
    controller.initialize_and_verify(env)

    max_deliveries = 50
    step = 0

    try:
        while True:            
            # Check completion conditions
            if (controller.metrics.get_successful_tasks().count >= max_deliveries or 
                controller._all_agents_dead()):
                if controller._all_agents_dead():
                    print("\nEMERGENCY STOP: All agents have depleted their batteries!")
                else:
                    print(f"\nAll {max_deliveries} deliveries completed!")
                
                break
            if controller.no_movement_steps > 100:  # 100 steps of no movement
                    print("\nEMERGENCY STOP: No agent has moved for 100 steps!")
                    break   
            actions = controller.get_actions()
            obs, rewards, done, truncated, info = env.step(actions)
            controller.metrics.record_battery_levels(env.agents)
            env.render()

            time.sleep(0.01)
            controller.current_step = step
            step += 1
            
    except KeyboardInterrupt:
        delivered = controller.metrics.get_successful_tasks().count
        print(f"\nSimulation stopped. Delivered {delivered}/{max_deliveries} shelves")
    finally:
        controller.metrics.print_summary()
        env.close()

if __name__ == "__main__":
    run_simulation()