"""
Run the same tDCS simulation on several meshes so the resulting fields can be
compared across mesh sizes.

Two kinds of model are supported (select with MODEL_TYPE):

  "head"   : SimNIBS head models stored as m2m_* folders (ernie). Electrodes
             are placed with EEG position names (C3, FC2).
  "sphere" : standalone spherical .msh files (no m2m folder). Electrodes are
             placed geometrically on the +x and -x sides of the sphere.

Each simulation is written directly under the results folder, in its own
sub-folder (sim_<model name>). No result file is copied elsewhere.
"""
from pathlib import Path

from simnibs import run_simnibs, sim_struct
from simnibs.mesh_tools import mesh_io

# ------------------------------------------------------------
# Model type
# ------------------------------------------------------------

MODEL_TYPE = "sphere"     # "head" or "sphere"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Head models: folder containing the m2m_* folders (one per head mesh)
HEAD_MESHES = PROJECT_ROOT / "data" / "ernie" / "m2m_ernie_uni"
HEAD_RESULTS_FOLDER = PROJECT_ROOT / "results" / "results_multisized_mesh"

# Sphere models: folder containing the standalone sphere .msh files
SPHERE_MESHES = PROJECT_ROOT / "data" / "m2m_spheres"
SPHERE_RESULTS_FOLDER = PROJECT_ROOT / "results" / "results_multisized_spheres"

# ------------------------------------------------------------
# Models to simulate
# ------------------------------------------------------------

# Sub-folders of HEAD_MESHES (used when MODEL_TYPE == "head")
M2M_FOLDER_NAMES = [
    # "m2m_ernie_uni_075",
    # "m2m_ernie_uni_085",
    # "m2m_ernie_uni_1",
    # "m2m_ernie_uni_2",
    # "m2m_ernie_uni_3",
    "m2m_ernie_uni_10",
]

# .msh files inside SPHERE_MESHES (used when MODEL_TYPE == "sphere")
SPHERE_MESH_NAMES = [
     "m2m_sphere",
]

# ------------------------------------------------------------
# tDCS settings
# ------------------------------------------------------------

# Fields saved in the output .msh:
# v = potential, e = |E|, E = E vector, j = |J|, J = J vector
FIELDS = "veEjJ"

CURRENTS = [0.001, -0.001]          # A, one value per electrode

ELECTRODE_SHAPE = "rect"
ELECTRODE_DIMENSIONS = [50, 50]     # mm
ELECTRODE_THICKNESS = 4             # mm

# Head models: one EEG position per electrode, same order as CURRENTS
ELECTRODE_CENTRES = ["C3", "FC2"]

# Sphere models: the pads are oriented towards a point this far (mm) from the
# electrode centre along +y, which fixes the rotation of the rectangular pad
SPHERE_PAD_YDIR_OFFSET = 20


# ------------------------------------------------------------
# Model discovery
# ------------------------------------------------------------

def get_model_paths():
    """Return the listed models (m2m folders or .msh files), failing early if any is missing."""
    if MODEL_TYPE == "head":
        paths = [HEAD_MESHES / name for name in M2M_FOLDER_NAMES]
        missing = [p for p in paths if not p.is_dir()]
    elif MODEL_TYPE == "sphere":
        paths = [SPHERE_MESHES / name for name in SPHERE_MESH_NAMES]
        missing = [p for p in paths if not p.is_file()]
    else:
        raise ValueError(f"MODEL_TYPE must be 'head' or 'sphere', got {MODEL_TYPE!r}")

    if not paths:
        raise ValueError(f"No models listed for MODEL_TYPE={MODEL_TYPE!r}.")

    if missing:
        missing_text = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(f"Model(s) not found:\n{missing_text}")

    return paths


def get_results_folder():
    return HEAD_RESULTS_FOLDER if MODEL_TYPE == "head" else SPHERE_RESULTS_FOLDER


def model_name(model_path):
    """Name used for the sim_<name> output folder."""
    return model_path.name if MODEL_TYPE == "head" else model_path.stem


# ------------------------------------------------------------
# Electrode placement
# ------------------------------------------------------------

def get_sphere_electrode_positions(mesh_file):
    """
    Return (centres, ydirs) for two electrodes on opposite sides of a sphere.

    Electrode 1 is at the +x extreme and electrode 2 at the -x extreme, at the
    middle of the y/z bounding box. The order matches CURRENTS.
    """
    mesh = mesh_io.read_msh(str(mesh_file))
    coords = mesh.nodes.node_coord

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    center_y = 0.5 * (y_min + y_max)
    center_z = 0.5 * (z_min + z_max)

    centres = [
        [x_max, center_y, center_z],
        [x_min, center_y, center_z],
    ]
    ydirs = [
        [x_max, center_y + SPHERE_PAD_YDIR_OFFSET, center_z],
        [x_min, center_y + SPHERE_PAD_YDIR_OFFSET, center_z],
    ]
    return centres, ydirs


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def build_session(model_path, output_folder):
    """Create the SimNIBS session for one model."""
    session = sim_struct.SESSION()
    session.open_in_gmsh = False
    session.fields = FIELDS
    session.pathfem = str(output_folder)

    if MODEL_TYPE == "head":
        session.subpath = str(model_path)
        centres = ELECTRODE_CENTRES
        ydirs = [None] * len(centres)
    else:
        session.fnamehead = str(model_path)
        centres, ydirs = get_sphere_electrode_positions(model_path)
        print("Electrode positions:")
        for channel, centre in enumerate(centres, start=1):
            print(f"  Electrode {channel} centre: {centre}")

    if len(centres) != len(CURRENTS):
        raise ValueError(
            f"{len(centres)} electrode positions but {len(CURRENTS)} currents."
        )

    tdcs = session.add_tdcslist()
    tdcs.currents = CURRENTS

    print("\nConductivity table used by this simulation:")
    for i, cond in enumerate(tdcs.cond):
        name = getattr(cond, "name", "no_name")
        value = getattr(cond, "value", "no_value")
        print(f"cond[{i}] | name = {name} | value = {value}")

    for channel, (centre, ydir) in enumerate(zip(centres, ydirs), start=1):
        electrode = tdcs.add_electrode()
        electrode.channelnr = channel
        electrode.centre = centre
        if ydir is not None:
            electrode.pos_ydir = ydir
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


def run_tdcs_simulation(model_path):
    output_folder = get_results_folder() / f"sim_{model_name(model_path)}"

    print("\n" + "=" * 70)
    print(f"Running simulation for: {model_path}")
    print(f"Saving output to:       {output_folder}")
    print("=" * 70)

    session = build_session(model_path, output_folder)
    run_simnibs(session)
    report_result_meshes(output_folder)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    model_paths = get_model_paths()
    get_results_folder().mkdir(parents=True, exist_ok=True)

    print(f"{len(model_paths)} {MODEL_TYPE} models will be simulated:")
    for path in model_paths:
        print(f"  {path}")

    for path in model_paths:
        run_tdcs_simulation(path)

    print("\nAll simulations finished.")


if __name__ == "__main__":
    main()