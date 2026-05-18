# SimNIBS Error Modelling 
**Bastien Bedard**

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

```
import sys
from pathlib import Path
import simnibs  # official installation
```

Add fork to path for modified modules

```
FORK_PATH = Path(__file__).parent.parent.parent / "simnibs" / "simnibs"
sys.path.insert(0, str(FORK_PATH))
```

Import modified modules from fork

```
from simulation import fem  # example
```

---

## Dependencies

- [SimNIBS 4.6](https://simnibs.github.io/simnibs/) — Brain stimulation simulation software
- Python 3.11 (included with SimNIBS installer)
- A forked version of SimNIBS source code (see below)
- SimNIBS example dataset (`m2m_ernie`) — see Data section for download instructions

---

## Installation

### Step 1 — Install SimNIBS
Download and run the official SimNIBS GUI installer for Windows:
 https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html

This installs SimNIBS and its Python environment at:
​```
C:\Users\<your_username>\SimNIBS-4.6\
​```

### Step 2 — Clone this repository
Both repositories must be in the **same parent folder**:
​```
git clone https://github.com/BastienBedard/SimNIBS_edits_2026.git
​```

### Step 3 — Clone the modified SimNIBS fork
​```
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

```​
C:\Users\<your_username>\SimNIBS-4.6\simnibs_env\python.exe
```

### Step 5 — Activate the SimNIBS environment
​```
C:\Users\<your_username>\SimNIBS-4.6\simnibs_env\Scripts\activate.bat
​```
If it does work in the PowerShell try in Command Prompt

---

## Running a Simulation
With the environment active, run a script from the repo root:
​```
python simulations\Test_TDCS.py
​```

Scripts use relative paths via `pathlib` — no manual path configuration needed!

---

## Data

Subject mesh data is not included in this repository due to file size.
Download the standard SimNIBS example dataset (`ernie`) here:
 https://simnibs.github.io/simnibs/build/html/dataset.html

Once downloaded, place the folder `m2m_ernie` in the following folder:
```
SimNIBS_edits_2026/
└── data/
    └── ernie/
```

---

## Notes for Collaborators
- Never run `pip uninstall simnibs` — reinstalling requires the full GUI installer
- Simulation results in `resultats/` are excluded from git (too large)
- Always use **Command Prompt** instead of PowerShell on Windows
