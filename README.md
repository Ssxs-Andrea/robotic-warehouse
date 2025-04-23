# Enhanced Robotic Warehouse Simulation



**The warehouse environment (folder robotic-warehouse) is adapted from [semitable/robotic-warehouse](https://github.com/semitable/robotic-warehouse)** with major enhancements:
- 🏋️ Shelf weight capacity system
- 🔋 Agent battery & charging stations
- 🚧 Static obstacle support
- 🆕 Custom environments (e.g. `rware-easy-1ag-2v`)

## 🚀 Installation

### Prerequisites
- [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
- Git (for cloning)

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/Ssxs-Andrea/robotic-warehouse.git
cd robotic-warehouse

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# 3. Install core package
cd robotic-warehouse
pip install -e .
