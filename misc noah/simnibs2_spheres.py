#simnibs2_spheres
from simnibs import sim_struct, run_simnibs
from simnibs.mesh_tools import mesh_io
from pathlib import Path
import shutil


# ------------------------------------------------------------
# Folder containing standalone spherical .msh files
# ------------------------------------------------------------

mesh_folder = Path("/Users/noahholm/Desktop/m2m_spheres")

sphere_meshes = sorted(mesh_folder.glob("*.msh"))

# Parent folder where all simulation outputs will be saved
output_parent = Path("/Users/noahholm/Desktop/simsfolders")

# Folder where only the final simulation result .msh files will be copied
simulation_result_folder = Path("/Users/noahholm/Desktop/simnibs_compare_spheres")

simulation_result_folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Function to collect simulation result .msh files
# ------------------------------------------------------------

def collect_simulation_result_msh_files(output_folder, mesh_file):

    msh_files = sorted(output_folder.rglob("*_EJV_*_scalar.msh"))

    if len(msh_files) == 0:
        msh_files = sorted(output_folder.rglob("*_TDCS_*_scalar.msh"))

    if len(msh_files) == 0:
        print(f"WARNING: No simulation result .msh files found in {output_folder}")
        return

    for msh_file in msh_files:
        destination_name = f"{mesh_file.stem}_TDCS_1_scalar.msh"
        destination_path = simulation_result_folder / destination_name

        shutil.copy2(msh_file, destination_path)

        print(f"Copied simulation result file:")
        print(f"  From: {msh_file}")
        print(f"  To:   {destination_path}")


# ------------------------------------------------------------
# Get opposite electrode positions from mesh bounds
# ------------------------------------------------------------

def get_opposite_sphere_positions(mesh_file):

    mesh = mesh_io.read_msh(str(mesh_file))
    coords = mesh.nodes.node_coord

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    center_y = 0.5 * (y_min + y_max)
    center_z = 0.5 * (z_min + z_max)

    electrode_1_center = [x_max, center_y, center_z]
    electrode_2_center = [x_min, center_y, center_z]

    # Direction used to orient rectangular electrodes on the sphere surface
    electrode_1_ydir = [x_max, center_y + 20, center_z]
    electrode_2_ydir = [x_min, center_y + 20, center_z]

    return electrode_1_center, electrode_2_center, electrode_1_ydir, electrode_2_ydir


# ------------------------------------------------------------
# Simulation function
# ------------------------------------------------------------

def run_tdcs_simulation(mesh_file):

    output_folder = output_parent / f"sim_{mesh_file.stem}"

    print("\n" + "=" * 70)
    print(f"Running simulation for: {mesh_file}")
    print(f"Saving output to:      {output_folder}")
    print("=" * 70 + "\n")

    e1_center, e2_center, e1_ydir, e2_ydir = get_opposite_sphere_positions(mesh_file)

    print("Electrode positions:")
    print(f"  Electrode 1 center: {e1_center}")
    print(f"  Electrode 2 center: {e2_center}")

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

    # Standalone head mesh file
    S.fnamehead = str(mesh_file)

    # Output folder
    S.pathfem = str(output_folder)

    # Create tDCS simulation
    tdcs = S.add_tdcslist()
    tdcs.currents = [0.001, -0.001]

    print("\nConductivity table used by this simulation:")
    for i, cond in enumerate(tdcs.cond):
        name = getattr(cond, "name", "no_name")
        value = getattr(cond, "value", "no_value")
        print(f"cond[{i}] | name = {name} | value = {value}")

    # Electrode 1: positive electrode on +x side of sphere
    e1 = tdcs.add_electrode()
    e1.channelnr = 1
    e1.centre = e1_center
    e1.pos_ydir = e1_ydir
    e1.shape = "rect"
    e1.dimensions = [50, 50]      # mm
    e1.thickness = 4              # mm

    # Electrode 2: negative electrode on -x side of sphere
    e2 = tdcs.add_electrode()
    e2.channelnr = 2
    e2.centre = e2_center
    e2.pos_ydir = e2_ydir
    e2.shape = "rect"
    e2.dimensions = [50, 50]      # mm
    e2.thickness = 4              # mm

    run_simnibs(S)

    # After SimNIBS finishes, copy the simulation result .msh files
    collect_simulation_result_msh_files(output_folder, mesh_file)


# ------------------------------------------------------------
# Check files before running
# ------------------------------------------------------------

print(f"Found {len(sphere_meshes)} sphere mesh files:")

for mesh_file in sphere_meshes:
    print(f"  {mesh_file}")

if len(sphere_meshes) == 0:
    raise RuntimeError("No .msh files found. Check the folder path.")


# ------------------------------------------------------------
# Run all simulations consecutively
# ------------------------------------------------------------

for mesh_file in sphere_meshes:
    run_tdcs_simulation(mesh_file)

print("\nAll simulations finished.")