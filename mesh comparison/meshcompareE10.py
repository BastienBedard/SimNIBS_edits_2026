from simnibs import mesh_io
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from pathlib import Path
from datetime import datetime
import re
import matplotlib.pyplot as plt
import csv

#Folder with all .msh files to compare 
meshcompare_folder = Path.home() / "Desktop" / "simnibs_compare"

# Folder where the VTK and NIfTI error maps will be saved
map_output_folder = Path.home() / "Desktop" / "meshcompare_vtk_error_maps"

# Folder where convergence CSV files will be saved
csv_output_folder = Path.home() / "Desktop" / "meshcomparegraphdata"

# Options:
# "convergence" = line graph and CSV for all comparisons
# "summary" = tissue-wise P95/P99 bar graphs
# "3d_map" = VTK point cloud and NIfTI sliceable volume typically for one comparison only
ANALYSIS_MODE = "convergence"

# If True, only keep tetrahedra corresponding to the tissue types in the list.
# If False, keep all tetrahedra.
COMPARE_ONLY_LIST = True

USED_TAGS = [1,2,3,4,5,6,7,8,9,10,100,500]
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

# Options for reference mesh:
# "most_tetrahedra" = use the file with the most kept tetrahedra as reference
# "specific_file" = use REFERENCE_FILE_NAME as reference
# "next_finer" = compare each mesh to the mesh with the next higher tetrahedra count
REFERENCE_MODE = "specific_file"

# Used only if REFERENCE_MODE = "specific_file"
REFERENCE_FILE_NAME = "m2m_ernie5_EJV_1_scalar.msh"

# ------------------------------------------------------------
# Default settings
# ------------------------------------------------------------

# Number of nearest lower-resolution tetrahedra used for interpolation
K_NEIGHBORS = 8

# Distance weighting exponent:
# 0 = distance is not factored in, neighbors are weighted equally
# 1 = smoother weighting
# 2 = closest points dominate more strongly
# 3 etc..
DISTANCE_POWER = 2

# If True, only compare tetrahedra with the same tissue tag.
# If False, ignore tissue tags and allow comparison across tissue types
MATCH_TISSUE_TAGS = True

# Used only for REFERENCE_MODE = "most_tetrahedra" or "specific_file"
# If True, the reference file is also compared to itself
# This should give 0%.
INCLUDE_REFERENCE_SELF_COMPARISON = False

# Small value to prevent division by zero in local relative error
RELATIVE_ERROR_EPSILON = 1e-12

# If True, writes a sliceable 3D NIfTI volume for each comparison
WRITE_NIFTI_VOLUME = True

# Voxel size for the sliceable volume, in mm
VOXEL_SIZE_MM = 0.5

# Maximum distance from a voxel center to a tetrahedron center.
# Voxels farther than this are treated as background.
# Increase this if the output volume has holes.
MAX_ASSIGN_DISTANCE_MM = 2.5

# Background value outside the assigned error region.
BACKGROUND_VALUE = 0.0

if ANALYSIS_MODE not in ["convergence", "summary", "3d_map"]:
    raise ValueError(
        "Invalid ANALYSIS_MODE. Use 'convergence', 'summary', or '3d_map'."
    )

if not meshcompare_folder.exists():
    raise FileNotFoundError(f"Folder not found: {meshcompare_folder}")

# Take every non-hidden file directly inside simnibs_compare.
# Ignore folders.
# Do not filter by extension/name.
files = [
    str(path)
    for path in meshcompare_folder.iterdir()
    if path.is_file() and not path.name.startswith(".")
]

files.sort()

if len(files) == 0:
    raise FileNotFoundError(f"No usable files found in: {meshcompare_folder}")

print("Files that will be compared:")
for f in files:
    print(f)


def safe_name(name):
    """
    Makes a filename-safe string.
    """
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


TISSUE_NAMES = {
    1: "White matter",
    2: "Gray matter",
    3: "CSF",
    4: "Bone",
    5: "Scalp",
    6: "Eyeballs",
    7: "Compact bone",
    8: "Spongy bone",
    9: "Blood",
    10: "Muscle",
    100: "Electrode",
    500: "Saline/gel",
}


def tissue_label_from_tag(tag):
    """
    Converts a tissue tag to a readable label.
    """
    tag_int = int(tag)
    name = TISSUE_NAMES.get(tag_int, "Unknown")
    return f"{tag_int}: {name}"


def weighted_percentile(values, weights, percentile):
    """
    Computes a volume-weighted percentile.

    percentile should be given as 95 or 99, not 0.95 or 0.99.

    Interpretation:
        weighted_percentile(values, volumes, 95)
        returns the value below which 95% of the physical volume lies.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]

    cumulative_weight = np.cumsum(weights_sorted)
    cutoff = percentile / 100.0 * cumulative_weight[-1]

    index = np.searchsorted(cumulative_weight, cutoff, side="left")
    index = min(index, len(values_sorted) - 1)

    return values_sorted[index]


def compute_tissue_percentile_summary(error_data):
    """
    Computes global and tissue-wise volume-weighted P95/P99 values
    for local absolute error and local relative error.
    """

    abs_error = error_data["local_abs_error"]
    rel_error = error_data["local_rel_error_percent"]
    volumes = error_data["reference_volumes"]
    tags = error_data["reference_tags"]

    summary = []

    # First row: global values across all valid reference tetrahedra
    summary.append({
        "label": "Global",
        "n_tetrahedra": len(abs_error),
        "total_volume": float(np.sum(volumes)),
        "abs_p95": weighted_percentile(abs_error, volumes, 95),
        "abs_p99": weighted_percentile(abs_error, volumes, 99),
        "rel_p95": weighted_percentile(rel_error, volumes, 95),
        "rel_p99": weighted_percentile(rel_error, volumes, 99),
    })

    # Tissue-wise values
    for tag in np.unique(tags):
        mask = tags == tag

        summary.append({
            "label": tissue_label_from_tag(tag),
            "n_tetrahedra": int(np.sum(mask)),
            "total_volume": float(np.sum(volumes[mask])),
            "abs_p95": weighted_percentile(abs_error[mask], volumes[mask], 95),
            "abs_p99": weighted_percentile(abs_error[mask], volumes[mask], 99),
            "rel_p95": weighted_percentile(rel_error[mask], volumes[mask], 95),
            "rel_p99": weighted_percentile(rel_error[mask], volumes[mask], 99),
        })

    return summary


def plot_tissue_percentile_summary(summary,
                                   test_file_name,
                                   reference_file_name):
    """
    Shows one figure with four bar graphs:
        1. P95 local absolute error
        2. P99 local absolute error
        3. P95 local relative error
        4. P99 local relative error

    The first bar in each subplot is the global value.
    The remaining bars are all tissue tags present in the valid comparison.
    No PNG is saved.
    """

    labels = [row["label"] for row in summary]

    abs_p95 = [row["abs_p95"] for row in summary]
    abs_p99 = [row["abs_p99"] for row in summary]
    rel_p95 = [row["rel_p95"] for row in summary]
    rel_p99 = [row["rel_p99"] for row in summary]

    plot_data = [
        (
            abs_p95,
            "Volume-weighted P95 local absolute error",
            "Absolute error |E_ref - E_test|"
        ),
        (
            abs_p99,
            "Volume-weighted P99 local absolute error",
            "Absolute error |E_ref - E_test|"
        ),
        (
            rel_p95,
            "Volume-weighted P95 local relative error",
            "Relative error (%)"
        ),
        (
            rel_p99,
            "Volume-weighted P99 local relative error",
            "Relative error (%)"
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.ravel()

    x = np.arange(len(labels))

    for ax, (values, title, ylabel) in zip(axes, plot_data):
        ax.bar(x, values)

        global_value = values[0]

        ax.axhline(
            global_value,
            linestyle="--",
            linewidth=1.5,
            label=f"Global = {global_value:.4g}"
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

        # Add value labels above bars
        for xi, value in zip(x, values):
            if np.isfinite(value):
                ax.text(
                    xi,
                    value,
                    f"{value:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    fig.suptitle(
        "Tissue-wise error percentiles\n"
        f"Test: {test_file_name}\n"
        f"Reference: {reference_file_name}",
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # One figure window only. No PNG saved.
    plt.show()


def get_tetra_data(mesh,
                   compare_only_gray_white=COMPARE_ONLY_LIST,
                   gray_white_tags=USED_TAGS):

    # Keep only tetrahedra
    tet_mask = mesh.elm.elm_type == 4

    # Tissue tags for tetrahedra
    tags_tet_all = mesh.elm.tag1[tet_mask]

    # Optionally keep only selected tissue types
    if compare_only_gray_white:
        tissue_mask = np.isin(tags_tet_all, gray_white_tags)
    else:
        tissue_mask = np.ones_like(tags_tet_all, dtype=bool)

    # Electric field vector
    # Usually shape is (number_of_elements, 3)
    E_all = mesh.field["E"].value
    E_tet_all = E_all[tet_mask, :3]
    E_tet = E_tet_all[tissue_mask]

    # Filter tissue tags
    tags_tet = tags_tet_all[tissue_mask]

    # Tetrahedron node numbers
    # SimNIBS/Gmsh uses 1-based node numbering, Python uses 0-based indexing
    tet_nodes_all = mesh.elm.node_number_list[tet_mask, :4] - 1
    tet_nodes = tet_nodes_all[tissue_mask]

    # Node coordinates
    coords = mesh.nodes.node_coord

    p0 = coords[tet_nodes[:, 0]]
    p1 = coords[tet_nodes[:, 1]]
    p2 = coords[tet_nodes[:, 2]]
    p3 = coords[tet_nodes[:, 3]]

    # Tetrahedron centers
    centers = (p0 + p1 + p2 + p3) / 4.0

    # Tetrahedron volumes
    volumes = np.abs(
        np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    ) / 6.0

    return centers, volumes, E_tet, tags_tet, tet_nodes, coords


def inverse_distance_weights(distances, distance_power):
    """
    Computes normalized inverse-distance weights.

    If a reference point has one or more zero-distance neighbours,
    only those exact matches are used.

    Otherwise, ordinary inverse-distance weighting is used:

        w_j = (1 / d_j^p) / sum_j(1 / d_j^p)

    where p = distance_power.
    """

    weights = np.zeros_like(distances, dtype=float)

    zero_distance_mask = distances == 0.0
    has_zero_distance = np.any(zero_distance_mask, axis=1)

    # Case 1: exact match exists.
    # Give equal weight to the zero-distance point(s), zero to all other points.
    if np.any(has_zero_distance):
        zero_rows = has_zero_distance

        weights[zero_rows] = zero_distance_mask[zero_rows].astype(float)

        weights[zero_rows] = (
            weights[zero_rows]
            / np.sum(weights[zero_rows], axis=1, keepdims=True)
        )

    # Case 2: no exact match.
    # Use ordinary inverse-distance weighting.
    nonzero_rows = ~has_zero_distance

    if np.any(nonzero_rows):
        d = distances[nonzero_rows]

        weights[nonzero_rows] = 1.0 / d**distance_power

        weights[nonzero_rows] = (
            weights[nonzero_rows]
            / np.sum(weights[nonzero_rows], axis=1, keepdims=True)
        )

    return weights


def estimate_test_E_with_knn(reference_centers,
                             reference_tags,
                             test_centers,
                             test_E,
                             test_tags,
                             k,
                             distance_power,
                             match_tissue_tags):
    """
    Estimates the test electric field at the reference tetrahedron centers.

    If match_tissue_tags is True:
        For each reference center, only test tetrahedra with the same tissue tag
        are used.

    If match_tissue_tags is False:
        Tissue tags are ignored.

    The test field is estimated using distance-weighted k-nearest neighbours.
    """

    test_E_on_reference = np.zeros((len(reference_centers), 3))
    valid_reference = np.zeros(len(reference_centers), dtype=bool)

    if match_tissue_tags:
        unique_reference_tags = np.unique(reference_tags)

        for tag in unique_reference_tags:
            # Reference tetrahedra with this tissue tag
            ref_mask = reference_tags == tag

            # Test tetrahedra with the same tissue tag
            test_mask = test_tags == tag

            if np.sum(test_mask) == 0:
                print(
                    f"Warning: no matching test tetrahedra for tissue tag {tag}. "
                    "These points are skipped."
                )
                continue

            ref_centers_tag = reference_centers[ref_mask]
            test_centers_tag = test_centers[test_mask]
            test_E_tag = test_E[test_mask]

            # Use at most the number of available test tetrahedra
            k_effective = min(k, len(test_centers_tag))

            tree = cKDTree(test_centers_tag)

            distances, indices = tree.query(ref_centers_tag, k=k_effective)

            # If k_effective = 1, scipy returns 1D arrays.
            # Convert to 2D so the rest of the code works the same.
            if k_effective == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]

            weights = inverse_distance_weights(distances, distance_power)

            # Electric fields of the neighboring test tetrahedra
            neighboring_E = test_E_tag[indices]

            # Distance-weighted estimate of the test field
            estimated_E = np.sum(weights[:, :, np.newaxis] * neighboring_E, axis=1)

            test_E_on_reference[ref_mask] = estimated_E
            valid_reference[ref_mask] = True

    else:
        # Ignore tissue tags completely
        k_effective = min(k, len(test_centers))

        tree = cKDTree(test_centers)

        distances, indices = tree.query(reference_centers, k=k_effective)

        if k_effective == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        weights = inverse_distance_weights(distances, distance_power)

        neighboring_E = test_E[indices]
        estimated_E = np.sum(weights[:, :, np.newaxis] * neighboring_E, axis=1)

        test_E_on_reference = estimated_E
        valid_reference[:] = True

    return test_E_on_reference, valid_reference


def compute_local_error_data(reference_centers,
                             reference_volumes,
                             reference_E,
                             reference_tags,
                             test_centers,
                             test_E,
                             test_tags,
                             k=K_NEIGHBORS,
                             distance_power=DISTANCE_POWER,
                             match_tissue_tags=MATCH_TISSUE_TAGS):
    """
    Compares test_E to reference_E.

    The test field is estimated at each reference tetrahedron center using
    distance-weighted k-nearest neighbours.

    This function returns both:
        1. The same global volume-weighted relative L2 error as the original script.
        2. Local per-tetrahedron error values for VTK visualization.
    """

    test_E_on_reference, valid_reference = estimate_test_E_with_knn(
        reference_centers=reference_centers,
        reference_tags=reference_tags,
        test_centers=test_centers,
        test_E=test_E,
        test_tags=test_tags,
        k=k,
        distance_power=distance_power,
        match_tissue_tags=match_tissue_tags
    )

    # Keep only reference points that had valid comparison points
    reference_E_valid = reference_E[valid_reference]
    test_E_valid = test_E_on_reference[valid_reference]
    reference_volumes_valid = reference_volumes[valid_reference]
    reference_tags_valid = reference_tags[valid_reference]

    if len(reference_E_valid) == 0:
        raise ValueError("No valid reference points were available for comparison.")

    # Vector difference
    diff = reference_E_valid - test_E_valid

    # Local absolute vector error
    local_abs_error = np.linalg.norm(diff, axis=1)

    # Reference magnitude
    reference_E_magn = np.linalg.norm(reference_E_valid, axis=1)

    # Local relative vector error, in percent
    local_rel_error_percent = (
        local_abs_error
        / np.maximum(reference_E_magn, RELATIVE_ERROR_EPSILON)
    ) * 100.0

    # Squared vector norms
    diff_squared = np.sum(diff**2, axis=1)
    reference_squared = np.sum(reference_E_valid**2, axis=1)

    # Volume-weighted relative L2 error
    numerator = np.sqrt(np.sum(diff_squared * reference_volumes_valid))
    denominator = np.sqrt(np.sum(reference_squared * reference_volumes_valid))
    global_relative_error_percent = numerator / denominator * 100.0

    return {
        "global_relative_error_percent": global_relative_error_percent,
        "valid_reference": valid_reference,
        "local_abs_error": local_abs_error,
        "local_rel_error_percent": local_rel_error_percent,
        "reference_tags": reference_tags_valid,
        "reference_volumes": reference_volumes_valid,
    }


def write_vtk_tetra_error_map(output_path,
                              coords,
                              tet_nodes,
                              valid_reference,
                              error_data):
    """
    Writes a legacy ASCII VTK POLYDATA point cloud.

    Each valid reference tetrahedron is represented by one point at its center.
    Error values are saved as POINT_DATA.

    This is much lighter and more Slicer-friendly than writing the full
    tetrahedral UNSTRUCTURED_GRID.
    """

    valid_tet_nodes = tet_nodes[valid_reference]

    if len(valid_tet_nodes) == 0:
        raise ValueError("No valid tetrahedra to write to VTK.")

    # Compute centers of the valid reference tetrahedra
    p0 = coords[valid_tet_nodes[:, 0]]
    p1 = coords[valid_tet_nodes[:, 1]]
    p2 = coords[valid_tet_nodes[:, 2]]
    p3 = coords[valid_tet_nodes[:, 3]]
    centers = (p0 + p1 + p2 + p3) / 4.0

    n_points = len(centers)

    with open(output_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SimNIBS localized electric-field comparison error point cloud\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n_points} float\n")
        for p in centers:
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")

        # One vertex per point
        f.write(f"VERTICES {n_points} {n_points * 2}\n")
        for i in range(n_points):
            f.write(f"1 {i}\n")

        # Point data
        f.write(f"POINT_DATA {n_points}\n")

        def write_scalar(name, values):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in values:
                f.write(f"{float(value):.9g}\n")

        def write_int_scalar(name, values):
            f.write(f"SCALARS {name} int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in values:
                f.write(f"{int(value)}\n")

        write_scalar("E_vector_relative_error_percent", error_data["local_rel_error_percent"])
        write_int_scalar("tissue_tag", error_data["reference_tags"])


def write_nifti_error_volume(output_path,
                             coords,
                             tet_nodes,
                             valid_reference,
                             error_data,
                             voxel_size_mm=VOXEL_SIZE_MM,
                             max_assign_distance_mm=MAX_ASSIGN_DISTANCE_MM,
                             background_value=BACKGROUND_VALUE):
    """
    Writes a sliceable 3D NIfTI volume.

    The error values are originally defined at reference tetrahedron centers.
    This function rasterizes them onto a regular voxel grid using nearest-neighbor
    assignment from tetrahedron centers to voxel centers.

    Output can be opened directly in 3D Slicer as a scalar volume.
    """

    valid_tet_nodes = tet_nodes[valid_reference]

    if len(valid_tet_nodes) == 0:
        raise ValueError("No valid tetrahedra to write to NIfTI.")

    # Compute centers of the valid reference tetrahedra
    p0 = coords[valid_tet_nodes[:, 0]]
    p1 = coords[valid_tet_nodes[:, 1]]
    p2 = coords[valid_tet_nodes[:, 2]]
    p3 = coords[valid_tet_nodes[:, 3]]
    centers = (p0 + p1 + p2 + p3) / 4.0

    values = error_data["local_rel_error_percent"].astype(float)

    # Bounding box of the whole reference mesh
    min_xyz = np.floor(coords.min(axis=0) / voxel_size_mm) * voxel_size_mm
    max_xyz = np.ceil(coords.max(axis=0) / voxel_size_mm) * voxel_size_mm

    x = np.arange(min_xyz[0], max_xyz[0] + voxel_size_mm, voxel_size_mm)
    y = np.arange(min_xyz[1], max_xyz[1] + voxel_size_mm, voxel_size_mm)
    z = np.arange(min_xyz[2], max_xyz[2] + voxel_size_mm, voxel_size_mm)

    nx, ny, nz = len(x), len(y), len(z)

    print(f"  Creating NIfTI volume shape: {nx} x {ny} x {nz}")

    # Create voxel-center coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    voxel_points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ])

    # Nearest tetrahedron center for each voxel
    tree = cKDTree(centers)
    distances, indices = tree.query(voxel_points, k=1)

    volume = np.full(len(voxel_points), background_value, dtype=np.float32)

    inside_mask = distances <= max_assign_distance_mm
    volume[inside_mask] = values[indices[inside_mask]]

    volume = volume.reshape((nx, ny, nz))

    # Affine maps voxel indices to SimNIBS mesh coordinates in mm
    affine = np.array([
        [voxel_size_mm, 0,             0,             min_xyz[0]],
        [0,             voxel_size_mm, 0,             min_xyz[1]],
        [0,             0,             voxel_size_mm, min_xyz[2]],
        [0,             0,             0,             1]
    ], dtype=float)

    img = nib.Nifti1Image(volume, affine)
    img.header.set_xyzt_units("mm")

    nib.save(img, str(output_path))


# ------------------------------------------------------------
# Read all simulations
# ------------------------------------------------------------

tetra_counts = []
centers_list = []
volumes_list = []
E_list = []
tags_list = []
tet_nodes_list = []
coords_list = []

for file in files:
    print("Reading:", file)

    mesh = mesh_io.read_msh(file)

    centers, volumes, E, tags, tet_nodes, coords = get_tetra_data(
        mesh,
        compare_only_gray_white=COMPARE_ONLY_LIST,
        gray_white_tags=USED_TAGS
    )

    tetra_counts.append(len(E))
    centers_list.append(centers)
    volumes_list.append(volumes)
    E_list.append(E)
    tags_list.append(tags)
    tet_nodes_list.append(tet_nodes)
    coords_list.append(coords)

    print("Tetrahedra kept:", len(E))


# ------------------------------------------------------------
# Choose reference simulation / comparison mode
# ------------------------------------------------------------

if REFERENCE_MODE == "most_tetrahedra":
    reference_index = int(np.argmax(tetra_counts))

    print()
    print("Reference mode:", REFERENCE_MODE)
    print("Reference file:", files[reference_index])
    print("Reference tetrahedra kept:", tetra_counts[reference_index])
    print("K nearest neighbors:", K_NEIGHBORS)
    print("Distance power:", DISTANCE_POWER)
    print("Match tissue tags:", MATCH_TISSUE_TAGS)
    print("Compare only tissue types in list:", COMPARE_ONLY_LIST)
    print("Used tissue tags:", USED_TAGS)
    print()

elif REFERENCE_MODE == "specific_file":
    matching_indices = [
        i
        for i, file in enumerate(files)
        if Path(file).name == REFERENCE_FILE_NAME
    ]

    if len(matching_indices) == 0:
        raise FileNotFoundError(
            f"Reference file '{REFERENCE_FILE_NAME}' was not found in:\n"
            f"{meshcompare_folder}\n\n"
            "Files found were:\n"
            + "\n".join(Path(file).name for file in files)
        )

    if len(matching_indices) > 1:
        raise ValueError(
            f"More than one file named '{REFERENCE_FILE_NAME}' was found. "
            "This should not happen if all files are directly in one folder."
        )

    reference_index = matching_indices[0]

    print()
    print("Reference mode:", REFERENCE_MODE)
    print("Reference file:", files[reference_index])
    print("Reference tetrahedra kept:", tetra_counts[reference_index])
    print("K nearest neighbors:", K_NEIGHBORS)
    print("Distance power:", DISTANCE_POWER)
    print("Match tissue tags:", MATCH_TISSUE_TAGS)
    print("Compare only tissue types in list:", COMPARE_ONLY_LIST)
    print("Used tissue tags:", USED_TAGS)
    print()

elif REFERENCE_MODE == "next_finer":
    print()
    print("Reference mode:", REFERENCE_MODE)
    print("Each file is compared to the file with the next higher tetrahedron count.")
    print("The finest file is skipped because it has no next finer reference.")
    print("K nearest neighbors:", K_NEIGHBORS)
    print("Distance power:", DISTANCE_POWER)
    print("Match tissue tags:", MATCH_TISSUE_TAGS)
    print("Compare only tissue types in list:", COMPARE_ONLY_LIST)
    print("Used tissue tags:", USED_TAGS)
    print()

else:
    raise ValueError(
        "Invalid REFERENCE_MODE. Use either "
        "'most_tetrahedra', 'specific_file', or 'next_finer'."
    )


# ------------------------------------------------------------
# Create output folder
# ------------------------------------------------------------

if ANALYSIS_MODE == "3d_map":
    map_output_folder.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if COMPARE_ONLY_LIST:
    tissue_label = "chosen_tissues"
else:
    tissue_label = "all_tissues"

if ANALYSIS_MODE == "3d_map":
    print(f"VTK files will be saved to: {map_output_folder}")
    print()


# ------------------------------------------------------------
# Compare simulations and write VTK files
# ------------------------------------------------------------

results = []

if REFERENCE_MODE in ["most_tetrahedra", "specific_file"]:

    for i, file in enumerate(files):
        if i == reference_index and not INCLUDE_REFERENCE_SELF_COMPARISON:
            continue

        print("Comparing:")
        print("  Test file:     ", file)
        print("  Reference file:", files[reference_index])

        error_data = compute_local_error_data(
            reference_centers=centers_list[reference_index],
            reference_volumes=volumes_list[reference_index],
            reference_E=E_list[reference_index],
            reference_tags=tags_list[reference_index],
            test_centers=centers_list[i],
            test_E=E_list[i],
            test_tags=tags_list[i],
            k=K_NEIGHBORS,
            distance_power=DISTANCE_POWER,
            match_tissue_tags=MATCH_TISSUE_TAGS
        )

        vtk_path = None

        if ANALYSIS_MODE == "3d_map":
            test_name = safe_name(file)
            reference_name = safe_name(files[reference_index])

            vtk_filename = (
                f"error_{tissue_label}_"
                f"test_{test_name}_"
                f"ref_{reference_name}_"
                f"{timestamp}.vtk"
            )

            vtk_path = map_output_folder / vtk_filename

            write_vtk_tetra_error_map(
                output_path=vtk_path,
                coords=coords_list[reference_index],
                tet_nodes=tet_nodes_list[reference_index],
                valid_reference=error_data["valid_reference"],
                error_data=error_data
            )

            if WRITE_NIFTI_VOLUME:
                nii_filename = vtk_filename.replace(".vtk", ".nii.gz")
                nii_path = map_output_folder / nii_filename

                write_nifti_error_volume(
                    output_path=nii_path,
                    coords=coords_list[reference_index],
                    tet_nodes=tet_nodes_list[reference_index],
                    valid_reference=error_data["valid_reference"],
                    error_data=error_data,
                    voxel_size_mm=VOXEL_SIZE_MM,
                    max_assign_distance_mm=MAX_ASSIGN_DISTANCE_MM,
                    background_value=BACKGROUND_VALUE
                )

                print(f"  NIfTI sliceable volume saved to: {nii_path}")

        global_error = error_data["global_relative_error_percent"]

        results.append((
            tetra_counts[i],
            global_error,
            Path(file).name,
            Path(files[reference_index]).name,
            vtk_path
        ))

        print(f"  Relative vector-field error: {global_error:.6f} %")
        if ANALYSIS_MODE == "3d_map":
            print(f"  VTK saved to: {vtk_path}")
        print()

        if ANALYSIS_MODE == "summary":
            tissue_summary = compute_tissue_percentile_summary(error_data)

            print("Volume-weighted tissue percentile summary:")
            for row in tissue_summary:
                print(
                    f"  {row['label']} | "
                    f"n={row['n_tetrahedra']} | "
                    f"volume={row['total_volume']:.6g} | "
                    f"abs P95={row['abs_p95']:.6g} | "
                    f"abs P99={row['abs_p99']:.6g} | "
                    f"rel P95={row['rel_p95']:.6g}% | "
                    f"rel P99={row['rel_p99']:.6g}%"
                )
            print()

            plot_tissue_percentile_summary(
                summary=tissue_summary,
                test_file_name=Path(file).name,
                reference_file_name=Path(files[reference_index]).name
            )

elif REFERENCE_MODE == "next_finer":

    # Sort the original file indices by tetrahedron count
    sorted_original_indices = np.argsort(tetra_counts)

    # Compare each mesh to the next finer mesh
    for position in range(len(sorted_original_indices) - 1):
        test_index = sorted_original_indices[position]
        reference_index = sorted_original_indices[position + 1]

        print("Comparing:")
        print("  Test file:     ", files[test_index])
        print("  Reference file:", files[reference_index])
        print("  Test tetrahedra:     ", tetra_counts[test_index])
        print("  Reference tetrahedra:", tetra_counts[reference_index])

        error_data = compute_local_error_data(
            reference_centers=centers_list[reference_index],
            reference_volumes=volumes_list[reference_index],
            reference_E=E_list[reference_index],
            reference_tags=tags_list[reference_index],
            test_centers=centers_list[test_index],
            test_E=E_list[test_index],
            test_tags=tags_list[test_index],
            k=K_NEIGHBORS,
            distance_power=DISTANCE_POWER,
            match_tissue_tags=MATCH_TISSUE_TAGS
        )

        vtk_path = None

        if ANALYSIS_MODE == "3d_map":
            test_name = safe_name(files[test_index])
            reference_name = safe_name(files[reference_index])

            vtk_filename = (
                f"error_{tissue_label}_"
                f"test_{test_name}_"
                f"ref_{reference_name}_"
                f"{timestamp}.vtk"
            )

            vtk_path = map_output_folder / vtk_filename

            write_vtk_tetra_error_map(
                output_path=vtk_path,
                coords=coords_list[reference_index],
                tet_nodes=tet_nodes_list[reference_index],
                valid_reference=error_data["valid_reference"],
                error_data=error_data
            )

            if WRITE_NIFTI_VOLUME:
                nii_filename = vtk_filename.replace(".vtk", ".nii.gz")
                nii_path = map_output_folder / nii_filename

                write_nifti_error_volume(
                    output_path=nii_path,
                    coords=coords_list[reference_index],
                    tet_nodes=tet_nodes_list[reference_index],
                    valid_reference=error_data["valid_reference"],
                    error_data=error_data,
                    voxel_size_mm=VOXEL_SIZE_MM,
                    max_assign_distance_mm=MAX_ASSIGN_DISTANCE_MM,
                    background_value=BACKGROUND_VALUE
                )

                print(f"  NIfTI sliceable volume saved to: {nii_path}")

        global_error = error_data["global_relative_error_percent"]

        results.append((
            tetra_counts[test_index],
            global_error,
            Path(files[test_index]).name,
            Path(files[reference_index]).name,
            vtk_path
        ))

        print(f"  Relative vector-field error: {global_error:.6f} %")
        if ANALYSIS_MODE == "3d_map":
            print(f"  VTK saved to: {vtk_path}")
        print()

        if ANALYSIS_MODE == "summary":
            tissue_summary = compute_tissue_percentile_summary(error_data)

            print("Volume-weighted tissue percentile summary:")
            for row in tissue_summary:
                print(
                    f"  {row['label']} | "
                    f"n={row['n_tetrahedra']} | "
                    f"volume={row['total_volume']:.6g} | "
                    f"abs P95={row['abs_p95']:.6g} | "
                    f"abs P99={row['abs_p99']:.6g} | "
                    f"rel P95={row['rel_p95']:.6g}% | "
                    f"rel P99={row['rel_p99']:.6g}%"
                )
            print()

            plot_tissue_percentile_summary(
                summary=tissue_summary,
                test_file_name=Path(files[test_index]).name,
                reference_file_name=Path(files[reference_index]).name
            )


# ------------------------------------------------------------
# Print sorted results
# ------------------------------------------------------------

results_sorted = sorted(results, key=lambda x: x[0])

print("Sorted results:")
for count, error, test_name, ref_name, vtk_path in results_sorted:
    if ANALYSIS_MODE == "3d_map":
        print(
            f"{count} tetrahedra kept | "
            f"{error:.6f} % | "
            f"test: {test_name} | "
            f"reference: {ref_name} | "
            f"vtk: {vtk_path.name}"
        )
    else:
        print(
            f"{count} tetrahedra kept | "
            f"{error:.6f} % | "
            f"test: {test_name} | "
            f"reference: {ref_name}"
        )

if ANALYSIS_MODE == "3d_map":
    print()
    print(f"All VTK files saved in: {map_output_folder}")

if ANALYSIS_MODE == "convergence":
    tetra_counts_sorted = np.array([row[0] for row in results_sorted])
    relative_errors_sorted = np.array([row[1] for row in results_sorted])
    file_names_sorted = np.array([row[2] for row in results_sorted])
    reference_file_names_sorted = np.array([row[3] for row in results_sorted])

    plt.plot(tetra_counts_sorted, relative_errors_sorted, marker="o")
    plt.xlabel("Number of tetrahedra kept")

    if REFERENCE_MODE == "next_finer":
        plt.ylabel("Relative vector-field difference to next finer mesh (%)")
        reference_label = "next finer mesh"
    elif REFERENCE_MODE == "specific_file":
        plt.ylabel("Relative vector-field error compared to reference (%)")
        reference_label = REFERENCE_FILE_NAME
    else:
        plt.ylabel("Relative vector-field error compared to reference (%)")
        reference_label = "mesh with most tetrahedra"

    if COMPARE_ONLY_LIST:
        plt.title(f"Electric-field difference in chosen tissue types\nReference: {reference_label}")
    else:
        plt.title(f"Electric-field difference\nReference: {reference_label}")

    plt.grid(True)
    plt.ylim(bottom=0)

    csv_output_folder.mkdir(parents=True, exist_ok=True)

    if REFERENCE_MODE == "specific_file":
        reference_name_for_file = Path(REFERENCE_FILE_NAME).stem
    elif REFERENCE_MODE == "next_finer":
        reference_name_for_file = "next_finer"
    else:
        reference_name_for_file = "most_tetrahedra"

    graph_data_filename = (
        f"meshcompare_{tissue_label}_ref_{reference_name_for_file}_{timestamp}.csv"
    )
    graph_data_path = csv_output_folder / graph_data_filename

    with open(graph_data_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "tetrahedra_kept",
            "relative_error_percent",
            "test_file",
            "reference_file",
            "reference_mode",
            "reference_label",
            "compare_only_list"
        ])

        for tetra_count, relative_error, test_name, ref_name in zip(
            tetra_counts_sorted,
            relative_errors_sorted,
            file_names_sorted,
            reference_file_names_sorted
        ):
            writer.writerow([
                tetra_count,
                relative_error,
                test_name,
                ref_name,
                REFERENCE_MODE,
                reference_label,
                COMPARE_ONLY_LIST
            ])

    print(f"Graph data saved to: {graph_data_path}")
    plt.show()