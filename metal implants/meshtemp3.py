import numpy as np
import pandas as pd
from pathlib import Path
from simnibs import mesh_io


# ============================================================
# USER SETTINGS
# ============================================================

MESH_FOLDER = Path("/Users/noahholm/Desktop/simnibs_compare")
OUT_MESH_PATH = Path("/Users/noahholm/Desktop/simnibs_temp/joule_heating.msh")
OUT_CSV_PATH = Path("/Users/noahholm/Desktop/simnibs_temp/joule_heating_stats.csv")
E_FIELD_NAME = "E"
J_FIELD_NAME = "J"

# Stimulation duration
DURATION_S = 20 * 60   # 20 minutes = 1200 seconds

# ------------------------------------------------------------
# Volumetric heat capacity settings
# ------------------------------------------------------------

# Default value used for all non-metal tissues/elements.
# Units: J/(m^3*C)
RHO_C = 3.5e6

# Volumetric heat capacity for medical-grade titanium / Ti implant region.
# Approximate value:
# rho_Ti ~ 4430 kg/m^3
# c_Ti   ~ 526 J/(kg*C)
# rho*c  ~ 2.33e6 J/(m^3*C)
RHO_C_TITANIUM = 2.33e6

# Tag used for the metal rod
METAL_ROD_TAG = 51

# Output field names
Q_FIELD_NAME = "joule_heating_W_per_m3"
DT_FIELD_NAME = "adiabatic_delta_T_C"
POWER_FIELD_NAME = "element_power_W"
RHO_C_FIELD_NAME = "volumetric_heat_capacity_J_per_m3_C"

# ------------------------------------------------------------
# Near-metal shell test settings
# ------------------------------------------------------------

DO_NEAR_METAL_SHELL_TEST = True

# Tissue tag used for the metal rod in your mesh.
# This should match METAL_ROD_TAG.
METAL_TAG = 51

# Radius around the metal rod in mm
NEAR_METAL_RADIUS_MM = 2.0

# If True, the shell contains only non-metal tetrahedra.
# This is usually what you want because the question is whether
# tissue near the rod heats up.
EXCLUDE_METAL_FROM_SHELL = True

# If True, excludes electrode and gel tags from the near-metal shell.
# Usually useful because you want patient tissue around the implant.
EXCLUDE_ELECTRODES_AND_GEL_FROM_SHELL = True

# If True, adds separate shell summaries for each tissue tag present
# inside the near-metal shell.
SHELL_BY_TISSUE_TAG = False

# If True, writes a binary element field showing which tetrahedra are
# inside the near-metal shell.
SHELL_FIELD_NAME = "near_metal_shell_mask"

# Optional: print stats separately for these tags if present
TAGS_TO_REPORT = {
    1: "WM",
    2: "GM",
    3: "CSF",
    4: "Bone",
    5: "Scalp",
    6: "Eyeballs",
    7: "Compact bone",
    8: "Spongy bone",
    9: "Blood",
    10: "Muscle",
    51: "Metal rod",
    100: "Electrode",
    500: "Gel",
}


msh_files = sorted(MESH_FOLDER.glob("*.msh"))

if len(msh_files) == 0:
    raise FileNotFoundError(f"No .msh files found in folder: {MESH_FOLDER}")

if len(msh_files) > 1:
    raise RuntimeError(
        f"More than one .msh file found in folder: {MESH_FOLDER}\n"
        "Please leave only one .msh file in this folder.\n\n"
        "Files found:\n" +
        "\n".join(str(p) for p in msh_files)
    )

MESH_PATH = msh_files[0]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_element_fields(msh):
    print("\nAvailable element fields:")
    for i, d in enumerate(msh.elmdata):
        value = np.asarray(d.value)
        print(f"  [{i}] {d.field_name}: shape {value.shape}")


def get_element_field(msh, field_name):
    for d in msh.elmdata:
        if d.field_name == field_name:
            return np.asarray(d.value)

    raise ValueError(
        f"Could not find element field '{field_name}'. "
        "Run the script and check the available field names printed above."
    )


def get_tetra_mask(msh):
    """
    Gmsh element type 4 = 4-node tetrahedron.
    """
    return np.asarray(msh.elm.elm_type) == 4


def get_element_tags(msh):
    """
    SimNIBS usually stores tissue labels in msh.elm.tag1.
    """
    return np.asarray(msh.elm.tag1)


def get_tetra_node_indices(msh, tetra_mask):
    """
    Returns tetrahedron node indices as 0-based Python indices.
    """

    tet_nodes = np.asarray(msh.elm.node_number_list)[tetra_mask, :4]

    # SimNIBS/Gmsh node numbers are generally 1-based.
    # Convert to 0-based Python indices.
    tet_nodes_0 = tet_nodes.astype(int) - 1

    return tet_nodes_0


def get_tetra_volumes_m3(msh, tetra_mask):
    """
    Computes tetrahedron volumes from node coordinates.

    SimNIBS/Gmsh coordinates are usually in millimeters, so the raw volume is mm^3.
    We convert to m^3 by multiplying by 1e-9.
    """

    node_coords_mm = np.asarray(msh.nodes.node_coord)
    tet_nodes_0 = get_tetra_node_indices(msh, tetra_mask)

    p0 = node_coords_mm[tet_nodes_0[:, 0]]
    p1 = node_coords_mm[tet_nodes_0[:, 1]]
    p2 = node_coords_mm[tet_nodes_0[:, 2]]
    p3 = node_coords_mm[tet_nodes_0[:, 3]]

    # Volume = |dot((p1-p0), cross((p2-p0), (p3-p0)))| / 6
    volumes_mm3 = np.abs(
        np.einsum(
            "ij,ij->i",
            p1 - p0,
            np.cross(p2 - p0, p3 - p0),
        )
    ) / 6.0

    volumes_m3 = volumes_mm3 * 1e-9

    return volumes_m3


def get_tetra_centroids_mm(msh, tetra_mask):
    """
    Computes tetrahedron centroids in millimeters.

    The centroid is the average of the four tetrahedron vertex coordinates.
    """

    node_coords_mm = np.asarray(msh.nodes.node_coord)
    tet_nodes_0 = get_tetra_node_indices(msh, tetra_mask)

    p0 = node_coords_mm[tet_nodes_0[:, 0]]
    p1 = node_coords_mm[tet_nodes_0[:, 1]]
    p2 = node_coords_mm[tet_nodes_0[:, 2]]
    p3 = node_coords_mm[tet_nodes_0[:, 3]]

    centroids_mm = (p0 + p1 + p2 + p3) / 4.0

    return centroids_mm


def normalize_field_to_tetra(field, tetra_mask, field_name):
    """
    Handles two common cases:
    1. Field has one value per mesh element.
    2. Field has one value per tetrahedron only.

    Returns field values only for tetrahedra.
    """

    n_total_elements = len(tetra_mask)
    n_tetra = np.count_nonzero(tetra_mask)

    field = np.asarray(field)

    if field.shape[0] == n_total_elements:
        return field[tetra_mask]

    if field.shape[0] == n_tetra:
        return field

    raise ValueError(
        f"Field '{field_name}' has incompatible shape {field.shape}. "
        f"Expected first dimension to be either total elements={n_total_elements} "
        f"or tetrahedra={n_tetra}."
    )


def make_full_element_array(tetra_values, tetra_mask, fill_value=0.0):
    """
    Creates a full element array for writing back to the mesh.
    Non-tetrahedral elements are filled with fill_value.
    """

    out = np.full(len(tetra_mask), fill_value, dtype=float)
    out[tetra_mask] = tetra_values
    return out


def make_elementwise_rho_c(tags_tetra):
    """
    Creates an elementwise volumetric heat capacity array.

    Default:
        rho*c = RHO_C

    Metal rod:
        rho*c = RHO_C_TITANIUM for tag METAL_ROD_TAG

    Units:
        J/(m^3*C)
    """

    rho_c = np.full(len(tags_tetra), RHO_C, dtype=float)
    rho_c[tags_tetra == METAL_ROD_TAG] = RHO_C_TITANIUM

    return rho_c


def summarize_region(name, q, delta_T, power, volumes_m3, mask):
    """
    Computes useful statistics for one region.
    """

    if np.count_nonzero(mask) == 0:
        return None

    q_region = q[mask]
    dt_region = delta_T[mask]
    power_region = power[mask]
    vol_region = volumes_m3[mask]

    return {
        "region": name,
        "n_tetra": int(np.count_nonzero(mask)),
        "volume_m3": float(np.sum(vol_region)),
        "total_power_W": float(np.sum(power_region)),

        "q_mean_W_m3": float(np.mean(q_region)),
        "q_max_W_m3": float(np.max(q_region)),
        "q_p99_W_m3": float(np.percentile(q_region, 99)),
        "q_p99_9_W_m3": float(np.percentile(q_region, 99.9)),

        "deltaT_mean_C": float(np.mean(dt_region)),
        "deltaT_max_C": float(np.max(dt_region)),
        "deltaT_p99_C": float(np.percentile(dt_region, 99)),
        "deltaT_p99_9_C": float(np.percentile(dt_region, 99.9)),
    }


def compute_near_metal_shell_mask(
    centroids_mm,
    tags_tetra,
    metal_tag,
    radius_mm,
    exclude_metal=True,
    exclude_electrodes_and_gel=True,
):
    """
    Creates a mask for tetrahedra whose centroid is within radius_mm
    of any metal tetrahedron centroid.

    Important:
    - This is a centroid-to-centroid distance test.
    - It is simple and robust.
    - It may miss some tetrahedra that touch the metal boundary but whose
      centroids are slightly outside the radius.
    - For a safety screen, use a radius at least as large as the local mesh size,
      or test several radii such as 1, 2, and 5 mm.
    """

    metal_mask = tags_tetra == metal_tag

    n_metal = np.count_nonzero(metal_mask)

    if n_metal == 0:
        print(f"\nWarning: No tetrahedra found with METAL_TAG={metal_tag}.")
        return np.zeros(len(tags_tetra), dtype=bool)

    metal_centroids = centroids_mm[metal_mask]

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(metal_centroids)
        distances_mm, _ = tree.query(centroids_mm, k=1)

    except ImportError:
        print("\nscipy not found. Falling back to slower NumPy distance calculation.")
        print("If your mesh is large, install scipy for a much faster near-metal shell test.")

        distances_mm = np.empty(len(centroids_mm), dtype=float)

        chunk_size = 5000
        for start in range(0, len(centroids_mm), chunk_size):
            end = min(start + chunk_size, len(centroids_mm))
            chunk = centroids_mm[start:end]

            # Squared distances from each chunk centroid to all metal centroids.
            diff = chunk[:, None, :] - metal_centroids[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            distances_mm[start:end] = np.sqrt(np.min(dist2, axis=1))

    shell_mask = distances_mm <= radius_mm

    if exclude_metal:
        shell_mask &= ~metal_mask

    if exclude_electrodes_and_gel:
        shell_mask &= ~np.isin(tags_tetra, [100, 500])

    return shell_mask


def add_near_metal_shell_rows(
    rows,
    shell_mask,
    tags_tetra,
    q,
    delta_T,
    power,
    volumes_m3,
    radius_mm,
    by_tissue_tag=True,
):
    """
    Adds near-metal shell statistics to the same output table as the global
    and tissue-tag summaries.
    """

    shell_name = f"Near-metal tissue shell <= {radius_mm:g} mm"

    row = summarize_region(
        name=shell_name,
        q=q,
        delta_T=delta_T,
        power=power,
        volumes_m3=volumes_m3,
        mask=shell_mask,
    )

    if row is not None:
        rows.append(row)

    if not by_tissue_tag:
        return rows

    shell_tags = sorted(set(tags_tetra[shell_mask].tolist()))

    for tag in shell_tags:
        label = TAGS_TO_REPORT.get(int(tag), f"Tag {int(tag)}")
        region_name = f"Near-metal shell <= {radius_mm:g} mm: {label} [tag {int(tag)}]"
        mask = shell_mask & (tags_tetra == tag)

        row = summarize_region(
            name=region_name,
            q=q,
            delta_T=delta_T,
            power=power,
            volumes_m3=volumes_m3,
            mask=mask,
        )

        if row is not None:
            rows.append(row)

    return rows


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    print(f"Reading mesh:\n  {MESH_PATH}")
    msh = mesh_io.read_msh(str(MESH_PATH))

    list_element_fields(msh)

    tetra_mask = get_tetra_mask(msh)
    tags_all = get_element_tags(msh)
    tags_tetra = tags_all[tetra_mask]

    n_total = len(tetra_mask)
    n_tetra = np.count_nonzero(tetra_mask)

    print(f"\nTotal elements: {n_total}")
    print(f"Tetrahedra:      {n_tetra}")

    E_raw = get_element_field(msh, E_FIELD_NAME)
    J_raw = get_element_field(msh, J_FIELD_NAME)

    E = normalize_field_to_tetra(E_raw, tetra_mask, E_FIELD_NAME)
    J = normalize_field_to_tetra(J_raw, tetra_mask, J_FIELD_NAME)

    if E.ndim != 2 or E.shape[1] != 3:
        raise ValueError(f"E field must have shape (n, 3). Got {E.shape}")

    if J.ndim != 2 or J.shape[1] != 3:
        raise ValueError(f"J field must have shape (n, 3). Got {J.shape}")

    print(f"\nUsing E field '{E_FIELD_NAME}' with shape {E.shape}")
    print(f"Using J field '{J_FIELD_NAME}' with shape {J.shape}")

    # --------------------------------------------------------
    # Joule heating
    # q = J dot E
    # Units:
    # J: A/m^2
    # E: V/m
    # q: W/m^3
    # --------------------------------------------------------

    q = np.einsum("ij,ij->i", J, E)

    # Numerical safety:
    # In passive ohmic tissue q should generally be non-negative.
    # Tiny negative values can occur from numerical precision or field conventions.
    n_negative = np.count_nonzero(q < 0)
    if n_negative > 0:
        print(f"\nWarning: {n_negative} tetrahedra have negative J·E values.")
        print("This may be numerical noise or a sign convention issue.")
        print("The script keeps the raw values. Check if large negatives occur.")

        q_negative_min = np.min(q)
        print(f"Most negative q value: {q_negative_min:.6g} W/m^3")

    # --------------------------------------------------------
    # Elementwise volumetric heat capacity
    #
    # Default:
    #   rho*c = 3.5e6 J/(m^3*C)
    #
    # Metal rod, tag 51:
    #   rho*c = 2.33e6 J/(m^3*C)
    #
    # This changes only the adiabatic delta_T calculation.
    # It does not change q = J dot E.
    # --------------------------------------------------------

    rho_c = make_elementwise_rho_c(tags_tetra)

    n_rod_tetra = np.count_nonzero(tags_tetra == METAL_ROD_TAG)
    print("\nVolumetric heat capacity:")
    print(f"  Default rho*c:        {RHO_C:.6g} J/(m^3*C)")
    print(f"  Titanium rho*c:       {RHO_C_TITANIUM:.6g} J/(m^3*C)")
    print(f"  Metal rod tag:        {METAL_ROD_TAG}")
    print(f"  Rod tetrahedra found: {n_rod_tetra}")

    if METAL_TAG != METAL_ROD_TAG:
        print("\nWarning:")
        print(f"  METAL_TAG for shell test is {METAL_TAG}")
        print(f"  METAL_ROD_TAG for titanium heat capacity is {METAL_ROD_TAG}")
        print("  These are different. Make sure this is intentional.")

    # --------------------------------------------------------
    # Adiabatic temperature rise
    # delta_T = q*t/(rho*c)
    # --------------------------------------------------------

    delta_T = q * DURATION_S / rho_c

    # --------------------------------------------------------
    # Element volumes, element power, and centroids
    # P_e = q_e * V_e
    # --------------------------------------------------------

    volumes_m3 = get_tetra_volumes_m3(msh, tetra_mask)
    power = q * volumes_m3
    centroids_mm = get_tetra_centroids_mm(msh, tetra_mask)

    print("\nBasic global tetrahedral results:")
    print(f"  q max:       {np.max(q):.6g} W/m^3")
    print(f"  q 99.9%:     {np.percentile(q, 99.9):.6g} W/m^3")
    print(f"  q 99%:       {np.percentile(q, 99):.6g} W/m^3")
    print(f"  q mean:      {np.mean(q):.6g} W/m^3")
    print(f"  ΔT max:      {np.max(delta_T):.6g} °C")
    print(f"  ΔT 99.9%:    {np.percentile(delta_T, 99.9):.6g} °C")
    print(f"  ΔT 99%:      {np.percentile(delta_T, 99):.6g} °C")
    print(f"  ΔT mean:     {np.mean(delta_T):.6g} °C")
    print(f"  Total power: {np.sum(power):.6g} W")

    # --------------------------------------------------------
    # Regional statistics
    # --------------------------------------------------------

    rows = []

    rows.append(
        summarize_region(
            name="All tetrahedra",
            q=q,
            delta_T=delta_T,
            power=power,
            volumes_m3=volumes_m3,
            mask=np.ones(n_tetra, dtype=bool),
        )
    )

    unique_tags = sorted(set(tags_tetra.tolist()))

    for tag in unique_tags:
        label = TAGS_TO_REPORT.get(int(tag), f"Tag {int(tag)}")
        region_name = f"{label} [tag {int(tag)}]"
        mask = tags_tetra == tag

        row = summarize_region(
            name=region_name,
            q=q,
            delta_T=delta_T,
            power=power,
            volumes_m3=volumes_m3,
            mask=mask,
        )

        if row is not None:
            rows.append(row)

    # --------------------------------------------------------
    # Near-metal shell statistics
    # --------------------------------------------------------

    shell_mask = np.zeros(n_tetra, dtype=bool)

    if DO_NEAR_METAL_SHELL_TEST:
        print("\nNear-metal shell test:")
        print(f"  Metal tag:          {METAL_TAG}")
        print(f"  Shell radius:       {NEAR_METAL_RADIUS_MM:g} mm")
        print(f"  Exclude metal:      {EXCLUDE_METAL_FROM_SHELL}")
        print(f"  Exclude electrodes: {EXCLUDE_ELECTRODES_AND_GEL_FROM_SHELL}")

        shell_mask = compute_near_metal_shell_mask(
            centroids_mm=centroids_mm,
            tags_tetra=tags_tetra,
            metal_tag=METAL_TAG,
            radius_mm=NEAR_METAL_RADIUS_MM,
            exclude_metal=EXCLUDE_METAL_FROM_SHELL,
            exclude_electrodes_and_gel=EXCLUDE_ELECTRODES_AND_GEL_FROM_SHELL,
        )

        n_shell = np.count_nonzero(shell_mask)
        shell_volume_m3 = np.sum(volumes_m3[shell_mask])

        print(f"  Shell tetrahedra:   {n_shell}")
        print(f"  Shell volume:       {shell_volume_m3:.6g} m^3")

        if n_shell == 0:
            print("  Warning: near-metal shell is empty.")
            print("  Try increasing NEAR_METAL_RADIUS_MM.")
        else:
            print(f"  Shell q max:        {np.max(q[shell_mask]):.6g} W/m^3")
            print(f"  Shell q 99.9%:      {np.percentile(q[shell_mask], 99.9):.6g} W/m^3")
            print(f"  Shell ΔT max:       {np.max(delta_T[shell_mask]):.6g} °C")
            print(f"  Shell ΔT 99.9%:     {np.percentile(delta_T[shell_mask], 99.9):.6g} °C")
            print(f"  Shell total power:  {np.sum(power[shell_mask]):.6g} W")

        rows = add_near_metal_shell_rows(
            rows=rows,
            shell_mask=shell_mask,
            tags_tetra=tags_tetra,
            q=q,
            delta_T=delta_T,
            power=power,
            volumes_m3=volumes_m3,
            radius_mm=NEAR_METAL_RADIUS_MM,
            by_tissue_tag=SHELL_BY_TISSUE_TAG,
        )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT_CSV_PATH, index=False)

    print(f"\nSaved statistics CSV:\n  {OUT_CSV_PATH}")

    print("\nRegional summary:")
    print(
        stats_df[
            [
                "region",
                "n_tetra",
                "total_power_W",
                "q_p99_9_W_m3",
                "q_max_W_m3",
                "deltaT_p99_9_C",
                "deltaT_max_C",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Add fields back to mesh and save
    # --------------------------------------------------------

    q_full = make_full_element_array(q, tetra_mask, fill_value=0.0)
    delta_T_full = make_full_element_array(delta_T, tetra_mask, fill_value=0.0)
    power_full = make_full_element_array(power, tetra_mask, fill_value=0.0)
    rho_c_full = make_full_element_array(rho_c, tetra_mask, fill_value=0.0)

    msh.add_element_field(q_full, Q_FIELD_NAME)
    msh.add_element_field(delta_T_full, DT_FIELD_NAME)
    msh.add_element_field(power_full, POWER_FIELD_NAME)
    msh.add_element_field(rho_c_full, RHO_C_FIELD_NAME)

    if DO_NEAR_METAL_SHELL_TEST:
        shell_full = make_full_element_array(shell_mask.astype(float), tetra_mask, fill_value=0.0)
        msh.add_element_field(shell_full, SHELL_FIELD_NAME)

    print(f"\nWriting output mesh:\n  {OUT_MESH_PATH}")
    mesh_io.write_msh(msh, str(OUT_MESH_PATH))

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()