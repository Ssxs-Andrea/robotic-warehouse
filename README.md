# Enhanced Robotic Warehouse Simulation



**The warehouse environment (folder robotic-warehouse) is adapted from [semitable/robotic-warehouse](https://github.com/semitable/robotic-warehouse)** with major enhancements:
- 🏋️ Shelf weight capacity system
- 🔋 Agent battery & charging stations
- 🚧 Static obstacle support
- 🆕 Custom environments (e.g. `rware-cond1-v2`)

## 🚀 Installation

### Prerequisites
- [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
- Git (for cloning)

### Setup
To setup the environment, run the following command:
```bash
# 1. Clone repository
git clone https://github.com/Ssxs-Andrea/robotic-warehouse.git


# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# 3. Install core package and run
cd robotic-warehouse
cd robotic-warehouse
pip install -e .
cd ..
```
## Getting Started
### GUI for A* Algorithm
We design a GUI for A* Algorithm which can choose to run on pre-defined or custom environment. For both environment, you can set the number of items to be delivered, enable rendering or not and simulation speed.

To run the GUI for A* Algorithm, use the following command:
```bash
cd astar_algorithm
python gui.py
```
### Autorunning
Autorunning will run the pre-defined environments without rendering it. The result will then saved as simulation_astar.txt or simulation_baseline.txt.


To run the autorunning for A*:
```bash
cd astar_algorithm
python autorunning.py
```

To run the autorunning for BFS:
```bash
cd baseline_algorithm
python autorunning.py
```

