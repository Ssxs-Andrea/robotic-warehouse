import time
from collections import defaultdict
import numpy as np
from types import SimpleNamespace

class MetricsTracker:
    def __init__(self):
        # Delivery metrics
        self.total_deliveries = 0
        self.successful_deliveries = 0
        self.failed_deliveries = 0
        self.completed_shelves = set()
        
        # Task timing metrics
        self.task_start_times = {}
        self.task_durations = []
        self.task_step_counts = []
        self.current_task_steps = defaultdict(int)
        
        # Movement metrics
        self.total_steps = 0
        self.collision_count = 0
        self.recovery_steps = defaultdict(int)
        self.recovery_steps_count = []
        self.in_recovery = defaultdict(bool)
        
        # Capacity metrics
        self.overcapacity_attempts = 0
        self.overcapacity_agents = set()
        

        # Battery metrics
        self.low_battery_events = 0
        self.critical_battery_events = 0
        self.battery_failures = 0
        self.charging_time = defaultdict(float)
        self.charging_durations = [] 
        self.charging_step_counts = [] 
        self.charging_steps = defaultdict(int)  
        self.discharged_agents = set()
        
        # Efficiency metrics
        self.idle_time = defaultdict(float)
        self.last_active_time = defaultdict(float)
        self.total_distance = defaultdict(float)
        self.last_positions = {}
        
        # Path metrics
        self.optimal_path_lengths = {}
        self.actual_path_lengths = defaultdict(int)
        
        # Initialize timing
        self.start_time = time.time()
        self.last_step_time = self.start_time

    def record_battery_levels(self, agents):
        """Record current battery levels of all agents"""
        for agent in agents:
            if agent.battery_level <= 0:
                self.battery_failures += 1

    # In your MetricsTracker class, add:
    def record_all_stuck(self, steps):
        """Record when all agents are stuck"""
        self.all_stuck_events += 1
        self.max_stuck_duration = max(self.max_stuck_duration, steps)

    def record_task_start(self, agent_id, shelf_id):
        """Record when an agent starts a new task"""
        self.task_start_times[(agent_id, shelf_id)] = time.time()
        self.current_task_steps[(agent_id)] = 0
        self.last_active_time[agent_id] = time.time()
        
    def record_movement(self, agent_id):
        """Record movement-related metrics"""
        self.current_task_steps[(agent_id)] += 1
        self.last_active_time[agent_id] = time.time()

        # If agent is charging, count the step
        if agent_id in self.charging_time:
            self.charging_steps[agent_id] += 1

    def record_total_steps(self):
        """Record movement-related metrics"""
        self.total_steps += 1


    def record_task_completion(self, agent_id, shelf_id):
        """Record when a task is successfully completed
        
        Args:
            agent_id: ID of the agent completing the task
            shelf_id: ID of the shelf being delivered
        """
        if (agent_id, shelf_id) not in self.task_start_times:
            return
        duration = time.time() - self.task_start_times[(agent_id, shelf_id)]
        steps = self.current_task_steps[(agent_id)]
        
        self.task_durations.append(duration)
        self.task_step_counts.append(steps)
        self.successful_deliveries += 1
        self.total_deliveries += 1
        self.completed_shelves.add(shelf_id)
        
        # Record path efficiency
        optimal_length = self.optimal_path_lengths.get((agent_id, shelf_id), 0)
        actual_length = self.actual_path_lengths.get((agent_id, shelf_id), 0)
        if optimal_length > 0:
            self.path_efficiency = actual_length / optimal_length
        
        # Clean up
        del self.task_start_times[(agent_id, shelf_id)]
        del self.current_task_steps[(agent_id)]

    def record_collision(self, agent_id):
        """Record a collision event"""
        self.collision_count += 1
        self.in_recovery[agent_id] = True
        print(f"Agent {agent_id} stuck!")
        
    def record_recovery_step(self, agent_id):
        """Record steps taken to recover from collision"""
        if self.in_recovery.get(agent_id, False):
            self.recovery_steps[agent_id] += 1
            
    def record_recovery_complete(self, agent_id):
        """Record when recovery from collision is complete"""
        if agent_id in self.in_recovery:
            
            self.recovery_steps_count.append(self.recovery_steps[agent_id])
            self.recovery_steps[agent_id] = 0  # Reset recovery steps for the next collision
            print(f"Total recovery steps: {self.recovery_steps_count[-1]}")
            del self.in_recovery[agent_id]
            del self.recovery_steps[agent_id]

    def record_overcapacity_attempt(self, agent_id, shelf_id, shelf_weight, agent_capacity):
        """Record when an agent attempts to carry a shelf that's too heavy"""
        self.overcapacity_attempts += 1
        self.overcapacity_agents.add(agent_id)
        self.failed_deliveries += 1
        self.total_deliveries += 1



    def record_low_battery(self, agent_id, battery_level):
        """Record when an agent battery is low"""
        self.low_battery_events += 1
        
    def record_critical_battery(self, agent_id, battery_level):
        """Record when an agent battery is critically low"""
        self.critical_battery_events += 1
        
    def record_battery_failure(self, agent_id):
        """Record when an agent runs out of battery before reaching charger"""
        self.battery_failures += 1
        self.discharged_agents.add(agent_id)
        self.failed_deliveries += 1
        self.total_deliveries += 1

    def record_charging_start(self, agent_id):
        """Record when an agent starts charging"""
        self.charging_time[agent_id] = time.time()
        self.charging_steps[agent_id] = 0  # Reset step counter
        
    def record_charging_end(self, agent_id):
        """Record when an agent finishes charging"""
        if agent_id in self.charging_time:
            start_time = self.charging_time[agent_id]
            duration = time.time() - start_time
            self.charging_time[agent_id] = duration
            self.charging_durations.append(duration)  # Store completed duration
            self.charging_step_counts.append(self.charging_steps[agent_id]) 
            del self.charging_time[agent_id]  # Remove from active charging

    def record_step_completion(self):
        """Record completion of a simulation step"""
        current_time = time.time()
        self.last_step_time = current_time
        
        # Update idle time for all agents
        for agent_id, last_active in self.last_active_time.items():
            idle_time = current_time - last_active
            self.idle_time[agent_id] += idle_time

    def get_metrics_summary(self):
        """Generate a comprehensive metrics summary"""
        avg_task_duration = np.mean(self.task_durations) if self.task_durations else 0
        avg_task_steps = np.mean(self.task_step_counts) if self.task_step_counts else 0
        total_recovery_steps = sum(self.recovery_steps_count) if self.recovery_steps_count else 0
        
        return {
            # Delivery metrics
            "total_deliveries": self.total_deliveries,
            
            # Timing metrics
            "total_simulation_time": time.time() - self.start_time,
            "average_task_duration": avg_task_duration,
            "average_task_steps": avg_task_steps,
            "total_steps": self.total_steps,
            
            # Movement metrics
            "total_collisions": self.collision_count,
            "average_recovery_steps": total_recovery_steps / self.collision_count if self.collision_count > 0 else 0,
        }

    def get_successful_tasks(self):
        """Returns metrics about successful tasks as an object with attribute access
        
        Returns:
            SimpleNamespace: Object with count, avg_duration, avg_steps, shelf_ids attributes
        """
        return SimpleNamespace(
            count=self.successful_deliveries,
            avg_duration=np.mean(self.task_durations) if self.task_durations else 0,
            avg_steps=np.mean(self.task_step_counts) if self.task_step_counts else 0,
            shelf_ids=self.completed_shelves
        )

    def print_summary(self):
        """Print a formatted summary of all metrics"""
        metrics = self.get_metrics_summary()
        
        print("\n=== Performance Metrics Summary ===")
        print(f"\n--- Delivery Metrics ---")
        print(f"Total deliveries attempted: {metrics['total_deliveries']}")
        
        print(f"\n--- Timing Metrics ---")
        print(f"Total simulation time: {metrics['total_simulation_time']:.2f} seconds")
        print(f"Average task duration: {metrics['average_task_duration']:.2f} seconds")
        print(f"Average steps per task: {metrics['average_task_steps']:.1f}")
        print(f"Total steps taken: {metrics['total_steps']}")
        
        print(f"\n--- Movement Metrics ---")
        print(f"Total collisions: {metrics['total_collisions']}")
        print(f"Average recovery steps per collision: {metrics['average_recovery_steps']:.1f}")
        