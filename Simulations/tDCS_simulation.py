"""
Run the same tDCS simulation on several head meshes (m2m folders) of the
ernie head model, so the resulting fields can be compared across mesh sizes.

Each simulation is written directly under MESHCOMPARE_FOLDER, in its own
sub-folder (sim_<m2m folder name>). No result file is copied elsewhere.
"""
from pathlib import Path

from simnibs import run_simnibs, sim_struct

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing the m2m_* folders (one per head mesh)
HEAD_MESHES = PROJECT_ROOT / "data" / "ernie"

# Simulation results are saved here
MESHCOMPARE_FOLDER = PROJECT_ROOT / "results" / "results_multisized_mesh"

# ------------------------------------------------------------
# Head meshes to simulate (sub-folders of HEAD_MESHES)
# ------------------------------------------------------------

M2M_FOLDER_NAMES = [
    "m2m_ernie_uni_075",
    "m2m_ernie_uni_085",
    "m2m_ernie_uni_1",  
    "m2m_ernie_uni_2",
    "m2m_ernie_uni_3",
]

# ------------------------------------------------------------
# tDCS settings
# ------------------------------------------------------------

# Fields saved in the output .msh:
# v = potential, e = |E|, E = E vector, j = |J|, J = J vector
FIELDS = "veEjJ"

CURRENTS = [0.001, -0.001]          # A, one value per electrode

# One entry per electrode, in the same order as CURRENTS
ELECTRODE_CENTRES = ["C3", "FC2"]   # EEG positions
ELECTRODE_SHAPE = "rect"
ELECTRODE_DIMENSIONS = [50, 50]     # mm
ELECTRODE_THICKNESS = 4             # mm


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_m2m_folders():
    """Return the listed m2m folders, failing early if any is missing."""
    folders = [HEAD_MESHES / name for name in M2M_FOLDER_NAMES]
    missing = [folder for folder in folders if not folder.is_dir()]

    if missing:
        missing_text = "\n".join(f"  {folder}" for folder in missing)
        raise FileNotFoundError(f"m2m folder(s) not found:\n{missing_text}")

    return folders


def build_session(m2m_folder, output_folder):
    """Create the SimNIBS session for one head mesh."""
    session = sim_struct.SESSION()
    session.open_in_gmsh = False
    session.fields = FIELDS
    session.subpath = str(m2m_folder)
    session.pathfem = str(output_folder)

    tdcs = session.add_tdcslist()
    tdcs.currents = CURRENTS

    print("\nConductivity table used by this simulation:")
    for i, cond in enumerate(tdcs.cond):
        name = getattr(cond, "name", "no_name")
        value = getattr(cond, "value", "no_value")
        print(f"cond[{i}] | name = {name} | value = {value}")

    for channel, centre in enumerate(ELECTRODE_CENTRES, start=1):
        electrode = tdcs.add_electrode()
        electrode.channelnr = channel
        electrode.centre = centre
        electrode.shape = ELECTRODE_SHAPE
        electrode.dimensions = ELECTRODE_DIMENSIONS
        electrode.thickness = ELECTRODE_THICKNESS

    return session


def report_result_meshes(output_folder):
    """Print the scalar result meshes found in the output folder."""
    msh_files = sorted(output_folder.rglob("*_scalar.msh"))

    if not msh_files:
        print(f"WARNING: no *_scalar.msh result found in {output_folder}")
        return

    for msh_file in msh_files:
        print(f"Result mesh: {msh_file}")


def run_tdcs_simulation(m2m_folder):
    output_folder = MESHCOMPARE_FOLDER / f"sim_{m2m_folder.name}"

    print("\n" + "=" * 70)
    print(f"Running simulation for: {m2m_folder}")
    print(f"Saving output to:       {output_folder}")
    print("=" * 70)

    session = build_session(m2m_folder, output_folder)
    run_simnibs(session)
    report_result_meshes(output_folder)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    m2m_folders = get_m2m_folders()
    MESHCOMPARE_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"{len(m2m_folders)} head meshes will be simulated:")
    for folder in m2m_folders:
        print(f"  {folder}")

    for folder in m2m_folders:
        run_tdcs_simulation(folder)

    print("\nAll simulations finished.")


if __name__ == "__main__":
    main()