from simnibs import sim_struct, run_simnibs
from pathlib import Path
import shutil


#This script will run a tDCS simulation for every m2m folder found in the base_folder.
#You can customize the electrode positions, sizes, 
# and currents as well as the tissue conductivities
#You can also customize if the simulation output will contain E, J or V.

# ------------------------------------------------------------
# Folder containing all m2m folders
# ------------------------------------------------------------

base_folder = Path("/Users/noahholm/Desktop/m2m_folders")

# Automatically find every folder whose name starts with m2m_
m2m_folders = sorted([
    folder
    for folder in base_folder.iterdir()
    if folder.is_dir() and folder.name.startswith("m2m_")
])

# Parent folder where all simulation outputs will be saved
output_parent = Path("/Users/noahholm/Desktop/simsfolders")

# Folder where only the final simulation result .msh files will be copied
simulation_result_folder = Path("/Users/noahholm/Desktop/simnibs_compare")

simulation_result_folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Function to collect simulation result .msh files
# ------------------------------------------------------------

def collect_simulation_result_msh_files(output_folder, m2m_folder):

    msh_files = sorted(output_folder.rglob("*_EJV_*_scalar.msh"))

    if len(msh_files) == 0:
        print(f"WARNING: No simulation result .msh files found in {output_folder}")
        return

    for msh_file in msh_files:
        destination_name = f"{m2m_folder.name}_TDCS_1_scalar.msh"
        destination_path = simulation_result_folder / destination_name

        shutil.copy2(msh_file, destination_path)

        print(f"Copied simulation result file:")
        print(f"  From: {msh_file}")
        print(f"  To:   {destination_path}")

# ------------------------------------------------------------
# Simulation function
# ------------------------------------------------------------

def run_tdcs_simulation(m2m_folder):
    output_folder = output_parent / f"sim_{m2m_folder.name}"

    print("\n" + "=" * 70)
    print(f"Running simulation for: {m2m_folder}")
    print(f"Saving output to:      {output_folder}")
    print("=" * 70 + "\n")

    S = sim_struct.SESSION()

    # Do not automatically open the result in Gmsh
    S.open_in_gmsh = False

    # Save fields in the output .msh file:
    # v = electric potential / voltage
    # e = electric-field magnitude
    # E = electric-field vector
    # j = current-density magnitude
    # J = current-density vector
    S.fields = "veEjJ"

    # Head model folder
    S.subpath = str(m2m_folder)

    # Output folder
    S.pathfem = str(output_folder)

    # Create tDCS simulation
    tdcs = S.add_tdcslist()
    tdcs.currents = [0.001, -0.001]

    # metal rod is tag 51
    tdcs.cond[50].value = 0.465
    tdcs.cond[50].name = "metal_rod_tag51"

    # Tag 52 = near-rod refinement shell
    # This is not metal. It is artificial tissue used to force local mesh refinement.
    # Choose a tissue-like value.
    tdcs.cond[51].value = 0.465
    tdcs.cond[51].name = "near_rod_shell_tag52"

    # Tissue tags:
    # 1 = White-Matter
    # 2 = Gray-Matter
    # 3 = CSF
    # 4 = Bone
    # 5 = Scalp
    # 6 = Eye_balls
    # 7 = Compact_bone
    # 8 = Spongy_bone
    # 9 = Blood
    # 10 = Muscle
    # 100 = Electrode
    # 500 = Saline_or_gel

    print("\nConductivity table used by this simulation:")
    for i, cond in enumerate(tdcs.cond):
        name = getattr(cond, "name", "no_name")
        value = getattr(cond, "value", "no_value")
        print(f"cond[{i}] | name = {name} | value = {value}")

    # Electrode 1: C3
    e1 = tdcs.add_electrode()
    e1.channelnr = 1
    e1.centre = "C3"
    e1.shape = "rect"
    e1.dimensions = [50, 50]          # mm
    e1.thickness = 4                  # mm

    # Electrode 2: FC2
    e2 = tdcs.add_electrode()
    e2.channelnr = 2
    e2.centre = "FC2"
    e2.shape = "rect"
    e2.dimensions = [50, 50]          # mm
    e2.thickness = 4                  # mm

    # Electrode 3: Oz
    # e3 = tdcs.add_electrode()
    # e3.channelnr = 3
    # e3.centre = "Oz"
    # e3.shape = "rect"
    # e3.dimensions = [50, 50]          # mm
    # e3.thickness = 4                  # mm

    # Electrode 4: TP10
    # e4 = tdcs.add_electrode()
    # e4.channelnr = 4
    # e4.centre = "TP10"
    # e4.shape = "rect"
    # e4.dimensions = [50, 50]          # mm
    # e4.thickness = 4                  # mm

    run_simnibs(S)

    # After SimNIBS finishes, copy the simulation result .msh files
    collect_simulation_result_msh_files(output_folder, m2m_folder)


# ------------------------------------------------------------
# Check folders before running
# ------------------------------------------------------------

print(f"Found {len(m2m_folders)} m2m folders:")

for folder in m2m_folders:
    print(f"  {folder}")

if len(m2m_folders) == 0:
    raise RuntimeError("No m2m folders found. Check the folder path.")

# ------------------------------------------------------------
# Run all simulations consecutively
# ------------------------------------------------------------

for m2m_folder in m2m_folders:
    run_tdcs_simulation(m2m_folder)

print("\nAll simulations finished.")