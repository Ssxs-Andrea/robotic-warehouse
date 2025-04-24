import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Event
import sys
from main import WarehouseController
import rware
from rware.warehouse import RewardType, Warehouse
import gymnasium as gym
import time


settings = [
    "rware-cond1-v2",
    "rware-cond2-v2",
    "rware-cond3-v2",
    "rware-cond4-v2",
    "rware-cond5-v2",
]

stop_event = Event()

# Define the custom environment setup
def run_simulation(env, log_queue, render_enabled=True, sleep_time=0.1, max_deliveries=50):
    controller = WarehouseController(env)
    obs, info = env.reset()
    controller.initialize_and_verify(env)

    step = 0
    try:
        while not stop_event.is_set():
            if controller.metrics.get_successful_tasks().count >= max_deliveries:
                print(f"\nAll {max_deliveries} deliveries completed! Simulation ending.")
                break

            actions = controller.get_actions()
            obs, rewards, done, truncated, info = env.step(actions)
            
            if render_enabled:
                env.render()
                time.sleep(sleep_time)

            controller.current_step = step
            step += 1
    finally:
        import io
        import contextlib

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            controller.metrics.print_summary()
            summary = buf.getvalue()
        log_queue.put(summary)

        env.close()  # This should now call the custom close method
        del env


class TextRedirector:
    def __init__(self, text_widget, log_queue, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag
        self.queue = log_queue

    def write(self, str):
        self.queue.put(str)

    def flush(self):
        pass

    def update_widget(self):
        while not self.queue.empty():
            msg = self.queue.get_nowait()
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg, (self.tag,))
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')


def gui():
    from multiprocessing import Queue
    log_queue = Queue()

    root = tk.Tk()
    root.title("Robotic Warehouse Simulator")
    root.geometry("900x800")  # Adjusted window size for better appearance
    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use("clam")

    # Set global font style for ttk widgets
    style.configure('TButton', font=('Helvetica', 12), padding=6)
    style.configure('TLabel', font=('Helvetica', 12))
    style.configure('TCombobox', font=('Helvetica', 12), padding=5)

    # Layout frames
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)

    control_frame = ttk.Frame(main_frame, padding=(10, 5))
    control_frame.pack(side="left", fill="y", padx=(0, 10), pady=10)

    output_frame = ttk.Frame(main_frame, padding=(10, 5))
    output_frame.pack(side="right", expand=True, fill="both", pady=10)

    # Title
    ttk.Label(control_frame, text="Select Environment", font=("Helvetica", 14, "bold")).pack(pady=(0, 15))

    # Radio buttons for selecting environment type
    env_type = tk.StringVar(value="predefined")  # Default option is predefined

    ttk.Radiobutton(control_frame, text="Pre-defined Environment", variable=env_type, value="predefined", command=lambda: toggle_inputs(env_type.get())).pack()
    ttk.Radiobutton(control_frame, text="Custom Environment", variable=env_type, value="custom", command=lambda: toggle_inputs(env_type.get())).pack()

    # Predefined environment dropdown

    env_var = tk.StringVar()
    env_menu = ttk.Combobox(control_frame, textvariable=env_var, values=settings, state="readonly", width=30)

    env_menu.pack(pady=10)
    env_menu.current(0)

    # Section for custom environment input
    custom_frame = ttk.Frame(control_frame)
    custom_frame.pack(pady=(10, 15))

    # ttk.Label(custom_frame, text="Custom Environment", font=("Helvetica", 12, "bold")).pack(pady=(10, 15))

    ttk.Label(custom_frame, text="Number of Agents:", font=("Helvetica", 10)).pack()
    agents_entry = ttk.Entry(custom_frame, font=("Helvetica", 10))
    agents_entry.pack(pady=5)

    ttk.Label(custom_frame, text="Number of Rows:", font=("Helvetica", 10)).pack()
    rows_entry = ttk.Entry(custom_frame, font=("Helvetica", 10))
    rows_entry.pack(pady=5)

    ttk.Label(custom_frame, text="Number of Columns:", font=("Helvetica", 10)).pack()
    cols_entry = ttk.Entry(custom_frame, font=("Helvetica", 10))
    cols_entry.pack(pady=5)

    ttk.Label(custom_frame, text="Column Height:", font=("Helvetica", 10)).pack()
    col_h_entry = ttk.Entry(custom_frame, font=("Helvetica", 10))
    col_h_entry.pack(pady=5)

    ttk.Label(custom_frame, text="Request Queue Size:", font=("Helvetica", 10)).pack()
    rq_size_entry = ttk.Entry(custom_frame, font=("Helvetica", 10))
    rq_size_entry.pack(pady=5)

    ttk.Label(custom_frame, text="Agent Capacities (comma separated):", font=("Helvetica", 10)).pack()
    agent_cap_entry = ttk.Entry(custom_frame, font=("Helvetica", 10))
    agent_cap_entry.pack(pady=5)

    ttk.Label(custom_frame, text="Shelf Weight Range (min, max):", font=("Helvetica", 10)).pack()
    weight_range_entry = ttk.Entry(custom_frame, font=("Helvetica", 10))
    weight_range_entry.pack(pady=5)

    # Simulation settings frame
    settings_frame = ttk.Frame(control_frame)
    settings_frame.pack(pady=(15, 10), fill="x")

    # Max deliveries entry
    ttk.Label(settings_frame, text="Request item count to be delivered:", font=("Helvetica", 9)).pack(anchor="w")
    max_deliveries_var = tk.IntVar(value=50)  # Default value
    max_deliveries_entry = ttk.Entry(settings_frame, textvariable=max_deliveries_var, font=("Helvetica", 10))
    max_deliveries_entry.pack(fill="x", pady=(0, 10))

    # Render controls
    render_frame = ttk.Frame(settings_frame)
    render_frame.pack(fill="x", pady=(0, 10))

    # Checkbox for enabling/disabling rendering
    render_var = tk.BooleanVar(value=True)  # Default to checked
    render_check = ttk.Checkbutton(render_frame, text="Enable Rendering", variable=render_var)
    render_check.pack(anchor="w", pady=(0, 10))

    # Slider for sleep time
    ttk.Label(render_frame, text="Simulation Speed:", font=("Helvetica", 9)).pack(anchor="w")
    sleep_time_var = tk.DoubleVar(value=0.1)  # Default value (0.1 seconds)
    
    # Frame for slider and value display
    slider_frame = ttk.Frame(render_frame)
    slider_frame.pack(fill="x", pady=(0, 10))
    
    sleep_slider = ttk.Scale(
        slider_frame, 
        from_=0.01, 
        to=1.0, 
        variable=sleep_time_var,
        command=lambda v: sleep_time_label.config(text=f"{float(v):.2f}s")
    )
    sleep_slider.pack(side="left", expand=True, fill="x")
    
    sleep_time_label = ttk.Label(slider_frame, text=f"{sleep_time_var.get():.2f}s", width=5, font=("Helvetica", 9))
    sleep_time_label.pack(side="left", padx=(1, 0))

    # Function to toggle visibility between predefined and custom environments
    def toggle_inputs(value):
        if value == "predefined":
            env_menu.pack(pady=10)
            custom_frame.pack_forget()
        else:
            env_menu.pack_forget()
            custom_frame.pack(pady=(10, 15))

    global sim_process
    sim_process = None
    stop_event = Event()

    def start():
        global sim_process
        if sim_process and sim_process.is_alive():
            return  # already running

        # Clear previous logs
        log_text.configure(state="normal")
        log_text.delete(1.0, tk.END)
        log_text.configure(state="disabled")

        stop_event.clear()

        try:
            max_deliveries = max_deliveries_var.get()
            if max_deliveries <= 0:
                raise ValueError("Max deliveries must be greater than 0")
        except tk.TclError:
            log_queue.put("Invalid max deliveries value. Using default (50).")
            max_deliveries = 50

        if env_type.get() == "predefined":
            selected_env = env_var.get()
            env = gym.make(selected_env)
            sim_process = Process(
                target=run_simulation, 
                args=(env, log_queue, render_var.get(), sleep_time_var.get(), max_deliveries)
            )
            sim_process.start()

        elif env_type.get() == "custom":
            try:
                n_agents = int(agents_entry.get())
                n_rows = int(rows_entry.get())
                n_cols = int(cols_entry.get())
                col_height = int(col_h_entry.get())
                rq_size = int(rq_size_entry.get())
                agent_capacities = list(map(int, agent_cap_entry.get().split(",")))
                weight_range = tuple(map(int, weight_range_entry.get().split(",")))

                env = Warehouse(
                    shelf_columns=n_cols,
                    shelf_rows=n_rows,
                    column_height=col_height,
                    n_agents=n_agents,
                    msg_bits=0,
                    sensor_range=1,
                    request_queue_size=rq_size,
                    agent_capacities=agent_capacities,
                    shelf_weight_range=weight_range,
                    max_inactivity_steps=None,
                    max_steps=None,
                    reward_type=RewardType.GLOBAL,
                )

                sim_process = Process(
                    target=run_simulation, 
                    args=(env, log_queue, render_var.get(), sleep_time_var.get(), max_deliveries)
                )
                sim_process.start()

            except ValueError as e:
                log_queue.put(f"Invalid input: {e}")
                return

    def stop():
        stop_event.set()
        if sim_process and sim_process.is_alive():
            sim_process.terminate()

    # Buttons
    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(pady=0)

    ttk.Button(btn_frame, text="▶ Run", command=start).grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    ttk.Button(btn_frame, text="⛔ Stop", command=stop).grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    # ttk.Label(control_frame, text="Use Stop button to force stop", font=("Helvetica", 9), foreground="gray").pack(pady=10)
    # Initialize with the predefined environment visible
    toggle_inputs(env_type.get())
    # Output area
    log_text = tk.Text(output_frame, wrap="word", height=20, state="disabled", bg="#f4f4f4", font=("Courier", 10))
    log_text.pack(expand=True, fill="both")
    log_scroll = ttk.Scrollbar(output_frame, command=log_text.yview)
    log_scroll.pack(side="right", fill="y")
    log_text.config(yscrollcommand=log_scroll.set)

    redirector = TextRedirector(log_text, log_queue)
    sys.stdout = redirector
    sys.stderr = redirector

    def poll_output():
        redirector.update_widget()
        root.after(100, poll_output)

    poll_output()

    root.mainloop()


if __name__ == "__main__":
    gui()