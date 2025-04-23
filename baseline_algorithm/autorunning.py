from multiprocessing import Queue, Process, Event
import time
from main import WarehouseController
import gymnasium as gym
import io
import contextlib

# Sample predefined environments
settings = [
    ("rware-easy-1ag-v2"),
    ("rware-easy-2ag-v2"),
    ("rware-med-2ag-v2"),
    ("rware-med-3ag-v2"),
    ("rware-hard-5ag-v2"),
    ("rware-hard-7ag-v2"),
]

stop_event = Event()

def run_simulation(env_name, log_queue):
    
    # Initialize environment and controller
    env = gym.make(env_name)
    controller = WarehouseController(env)
    obs, info = env.reset()
    controller.initialize_and_verify(env)

    max_deliveries = 50
    step = 0
    try:
        while not stop_event.is_set():
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
            #env.render()
            #time.sleep(0.001)

            controller.current_step = step
            step += 1
    finally:
        

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            controller.metrics.print_summary()
            summary = buf.getvalue()
        log_queue.put(f"=== {env_name} ===\n{summary}\n")


        env.close()
        del env

def save_summary_to_file(log_queue):
    with open("simulation_baseline.txt", "w") as file:
        while not log_queue.empty():
            summary = log_queue.get_nowait()
            file.write(summary + "\n")

def automate_simulation():
    log_queue = Queue()

    for env_name in settings:
        print(f"Running simulation for: {env_name}")

        # Run the simulation process for the current environment
        sim_process = Process(target=run_simulation, args=(env_name, log_queue))
        sim_process.start()
        sim_process.join()  # Wait for the process to finish before proceeding to the next one
    
    # Save all collected summaries to a file
    save_summary_to_file(log_queue)

if __name__ == "__main__":
    automate_simulation()
