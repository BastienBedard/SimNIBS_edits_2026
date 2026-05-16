# SimNIBS Error Modelling — Maîtrise 2026
**Bastien Bedard** | Université — 2026

This repository contains scripts and tools for modelling simulation errors in transcranial Direct Current Stimulation (tDCS) using a selectively modified version of [SimNIBS](https://simnibs.github.io/simnibs/).

---

## Project Structure

```
SimNIBS_edits_2026/
├── simulations/                # Python scripts for tDCS simulations
├── data/ernie/                 # Subject mesh data (e.g. m2m_ernie)
├── resultats/resultats_tdcs/   # Simulation outputs (local only, not tracked by git)
├── .gitignore
└── README.md
```

---

## How it works

This project uses the official SimNIBS installation for all compiled 
dependencies, while selectively overriding specific Python modules 
with modified versions from a forked repository.

In each simulation script, the fork path is added to `sys.path` 
**after** the official SimNIBS import:

​```python
import sys
from pathlib import Path
import simnibs  # official installation

# Add fork to path for modified modules
FORK_PATH = Path(__file__).parent.parent.parent / "simnibs" / "simnibs"
sys.path.insert(0, str(FORK_PATH))

# Import modified modules from fork
from simulation import fem  # example
​```

---

## Dependencies

- [SimNIBS 4.6](https://simnibs.github.io/simnibs/) — Brain stimulation simulation software
- Python 3.11 (included with SimNIBS installer)
- A forked version of SimNIBS source code (see below)

---

## Installation

### Step 1 — Install SimNIBS
Download and run the official SimNIBS GUI installer for Windows:
👉 https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html

This installs SimNIBS and its Python environment at:
​```
C:\Users\<your_username>\SimNIBS-4.6\
​```

### Step 2 — Clone this repository
Both repositories must be in the **same parent folder**:
​```bash
git clone https://github.com/BastienBedard/SimNIBS_edits_2026.git
​```

### Step 3 — Clone the modified SimNIBS fork
​```bash
git clone https://github.com/BastienBedard/simnibs.git
​```

Your folder structure should look like this:
​```
ParentFolder/
├── SimNIBS_edits_2026/    ← this repo
└── simnibs/               ← forked SimNIBS source
​```

### Step 4 — Select Python Interpreter in VSCode
`Ctrl + Shift + P` → `Python: Select Interpreter` → enter path manually:
​```
C:\Users\<your_username>\SimNIBS-4.6\simnibs_env\python.exe
```

### Step 5 — Activate the environment
Use Command Prompt (not PowerShell) to activate the SimNIBS environment:
​```cmd
C:\Users\<your_username>\SimNIBS-4.6\simnibs_env\Scripts\activate.bat
​```

---

## Running a Simulation
With the environment active, run a script from the repo root:
​```cmd
python simulations\Test_TDCS.py
​```

Scripts use relative paths via `pathlib` — no manual path configuration needed!

---

## Data
Subject mesh data (`m2m_ernie`) will be added in a future commit.
It is the standard SimNIBS example dataset available at:
👉 https://simnibs.github.io/simnibs/build/html/dataset.html

---

## Notes for Collaborators
- ⚠️ Never run `pip uninstall simnibs` — reinstalling requires the full GUI installer
- Simulation results in `resultats/` are excluded from git (too large)
- Always use **Command Prompt** instead of PowerShell on Windows
- The `_version.py` file in the fork must be created manually (see below)

## Known Setup Steps for the Fork
After cloning the fork, create this file manually:

**`simnibs/simnibs/_version.py`**
​```python
__version__ = "4.6.0"
```