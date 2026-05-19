import sys
from pathlib import Path

import simnibs

# Chemin vers le fork
FORK_PATH = Path(__file__).parent.parent.parent / "simnibs" / "simnibs"
sys.path.insert(0, str(FORK_PATH))

# from simulation import analytical_solutions

# Racine du projet — remonte depuis simulations/ vers SimNIBS_edits_2026/
PROJECT_ROOT = Path(__file__).parent.parent


# Créer une session de simulation
s = simnibs.sim_struct.SESSION()

# Chemin vers le dataset m2m_ernie
s.subpath = str(PROJECT_ROOT / "data" / "ernie" / "m2m_ernie")

# Dossier de sortie pour les résultats
s.pathfem = str(PROJECT_ROOT / "results" / "results_tdcs")

# Créer une simulation tDCS
tdcs = s.add_tdcslist()
tdcs.currents = [0.001, -0.001]  # 1mA sur chaque électrode (total = 0)

# Électrode 1 — Anode (positive)
electrode1 = tdcs.add_electrode()
electrode1.channelnr = 1
electrode1.centre = 'C3'        # Position sur le crâne (système 10-20)
electrode1.shape = 'ellipse'
electrode1.dimensions = [50, 50]  # mm
electrode1.thickness = 4          # mm

# Électrode 2 — Cathode (negative)
electrode2 = tdcs.add_electrode()
electrode2.channelnr = 2
electrode2.centre = 'AF4'
electrode2.shape = 'ellipse'
electrode2.dimensions = [50, 50]
electrode2.thickness = 4

# Lancer la simulation
simnibs.run_simnibs(s)

print("Simulation terminée!")
