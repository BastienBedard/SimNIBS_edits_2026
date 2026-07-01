from simnibs import mesh_io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import csv
from datetime import datetime

#Similar to meshcompareE6.py, but compares current density J instead of E.

# ------------------------------------------------------------
# Adjustable settings
# ------------------------------------------------------------

# Number of nearest lower-resolution tetrahedra used for interpolation
K_NEIGHBORS = 8

# Distance weighting exponent:
# 0 = distance is not factored in, neighbors are weighted equally
# 1 = smoother weighting
# 2 = closest points dominate more strongly
DISTANCE_POWER = 2

# If True, only compare tetrahedra with the same tissue tag.
# If False, ignore tissue tags and allow comparison across tissue types.
MATCH_TISSUE_TAGS = True

# If True, only keep tetrahedra corresponding to the tissue types in the list.
# If False, keep all tetrahedra.
COMPARE_ONLY_LIST = True

USED_TAGS = [1, 2]
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


# ------------------------------------------------------------
# Reference-solution settings
# ------------------------------------------------------------

# Options:
# "most_tetrahedra" = use the file with the most kept tetrahedra as reference
# "specific_file" = use REFERENCE_FILE_NAME as reference
# "next_finer" = compare each mesh to the mesh with the next higher tetrahedra count
REFERENCE_MODE = "next_finer"

# Used only if REFERENCE_MODE = "specific_file"
REFERENCE_FILE_NAME = "m2m_ernie6_E_constantJV_1_scalar.msh"

# Used only for REFERENCE_MODE = "most_tetrahedra" or "specific_file".
# If True, the reference file is also compared to itself.
# This should give 0%.
# If False, the reference file is skipped in the final plot.
#
# This setting is ignored when REFERENCE_MODE = "next_finer",
# because the finest mesh has no next finer mesh to compare to.
INCLUDE_REFERENCE_SELF_COMPARISON = False


# ------------------------------------------------------------
# Folder containing all files to compare
# ------------------------------------------------------------

meshcompare_folder = Path.home() / "Desktop" / "simnibs_compare"

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


def get_tetra_data(mesh,
                   compare_only_list=COMPARE_ONLY_LIST,
                   used_tags=USED_TAGS):

    # Keep only tetrahedra
    tet_mask = mesh.elm.elm_type == 4

    # Tissue tags for tetrahedra
    tags_tet_all = mesh.elm.tag1[tet_mask]

    # Optionally keep only selected tissue types
    if compare_only_list:
        tissue_mask = np.isin(tags_tet_all, used_tags)
    else:
        tissue_mask = np.ones_like(tags_tet_all, dtype=bool)

    # Current density vector
    # Usually shape is (number_of_elements, 3)
    if "J" not in mesh.field:
        available_fields = list(mesh.field.keys())
        raise KeyError(
            "This .msh file does not contain a field named 'J'.\n"
            f"Available fields are: {available_fields}\n\n"
            "To compare current density directly, the simulation output must contain J.\n"
            "If your file only contains E, then J would need to be computed as J = sigma * E."
        )

    J_all = mesh.field["J"].value
    J_tet_all = J_all[tet_mask, :3]
    J_tet = J_tet_all[tissue_mask]

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

    return centers, volumes, J_tet, tags_tet


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


def estimate_test_J_with_knn(reference_centers,
                             reference_tags,
                             test_centers,
                             test_J,
                             test_tags,
                             k,
                             distance_power,
                             match_tissue_tags):
    """
    Estimates the test current density at the reference tetrahedron centers.

    If match_tissue_tags is True:
        For each reference center, only test tetrahedra with the same tissue tag
        are used.

    If match_tissue_tags is False:
        Tissue tags are ignored.

    The test field is estimated using distance-weighted k-nearest neighbours.
    """

    test_J_on_reference = np.zeros((len(reference_centers), 3))
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
            test_J_tag = test_J[test_mask]

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

            # Current densities of the neighboring test tetrahedra
            neighboring_J = test_J_tag[indices]

            # Distance-weighted estimate of the test field
            estimated_J = np.sum(weights[:, :, np.newaxis] * neighboring_J, axis=1)

            test_J_on_reference[ref_mask] = estimated_J
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

        neighboring_J = test_J[indices]
        estimated_J = np.sum(weights[:, :, np.newaxis] * neighboring_J, axis=1)

        test_J_on_reference = estimated_J
        valid_reference[:] = True

    return test_J_on_reference, valid_reference


def relative_vector_error(reference_centers,
                          reference_volumes,
                          reference_J,
                          reference_tags,
                          test_centers,
                          test_J,
                          test_tags,
                          k=K_NEIGHBORS,
                          distance_power=DISTANCE_POWER,
                          match_tissue_tags=MATCH_TISSUE_TAGS):
    """
    Compares test_J to reference_J.

    The test field is estimated at each reference tetrahedron center using
    distance-weighted k-nearest neighbours.

    The final relative error is volume-weighted using the reference mesh volumes.
    """

    test_J_on_reference, valid_reference = estimate_test_J_with_knn(
        reference_centers=reference_centers,
        reference_tags=reference_tags,
        test_centers=test_centers,
        test_J=test_J,
        test_tags=test_tags,
        k=k,
        distance_power=distance_power,
        match_tissue_tags=match_tissue_tags
    )

    # Keep only reference points that had valid comparison points
    reference_J_valid = reference_J[valid_reference]
    test_J_valid = test_J_on_reference[valid_reference]
    reference_volumes_valid = reference_volumes[valid_reference]

    if len(reference_J_valid) == 0:
        raise ValueError("No valid reference points were available for comparison.")

    # Vector difference
    diff = reference_J_valid - test_J_valid

    # Squared vector norms
    diff_squared = np.sum(diff**2, axis=1)
    reference_squared = np.sum(reference_J_valid**2, axis=1)

    # Volume-weighted relative L2 error
    numerator = np.sqrt(np.sum(diff_squared * reference_volumes_valid))
    denominator = np.sqrt(np.sum(reference_squared * reference_volumes_valid))

    return numerator / denominator * 100


# ------------------------------------------------------------
# Read all simulations
# ------------------------------------------------------------

tetra_counts = []
centers_list = []
volumes_list = []
J_list = []
tags_list = []

for file in files:
    print("Reading:", file)

    mesh = mesh_io.read_msh(file)

    centers, volumes, J, tags = get_tetra_data(
        mesh,
        compare_only_list=COMPARE_ONLY_LIST,
        used_tags=USED_TAGS
    )

    tetra_counts.append(len(J))
    centers_list.append(centers)
    volumes_list.append(volumes)
    J_list.append(J)
    tags_list.append(tags)

    print("Tetrahedra kept:", len(J))


# ------------------------------------------------------------
# Choose reference simulation / comparison mode
# ------------------------------------------------------------

if REFERENCE_MODE == "most_tetrahedra":
    reference_index = int(np.argmax(tetra_counts))

    reference_centers = centers_list[reference_index]
    reference_volumes = volumes_list[reference_index]
    reference_J = J_list[reference_index]
    reference_tags = tags_list[reference_index]

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

    reference_centers = centers_list[reference_index]
    reference_volumes = volumes_list[reference_index]
    reference_J = J_list[reference_index]
    reference_tags = tags_list[reference_index]

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
# Compare simulations
# ------------------------------------------------------------

relative_errors = []
tetra_counts_for_plot = []
file_names_for_plot = []
reference_file_names_for_plot = []

if REFERENCE_MODE in ["most_tetrahedra", "specific_file"]:

    for i, file in enumerate(files):
        if i == reference_index and not INCLUDE_REFERENCE_SELF_COMPARISON:
            continue

        print("Comparing:")
        print("  Test file:     ", file)
        print("  Reference file:", files[reference_index])

        error = relative_vector_error(
            reference_centers=reference_centers,
            reference_volumes=reference_volumes,
            reference_J=reference_J,
            reference_tags=reference_tags,
            test_centers=centers_list[i],
            test_J=J_list[i],
            test_tags=tags_list[i],
            k=K_NEIGHBORS,
            distance_power=DISTANCE_POWER,
            match_tissue_tags=MATCH_TISSUE_TAGS
        )

        relative_errors.append(error)
        tetra_counts_for_plot.append(tetra_counts[i])
        file_names_for_plot.append(Path(file).name)
        reference_file_names_for_plot.append(Path(files[reference_index]).name)

        print(f"  Relative current-density vector-field error: {error:.6f} %")
        print()

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

        error = relative_vector_error(
            reference_centers=centers_list[reference_index],
            reference_volumes=volumes_list[reference_index],
            reference_J=J_list[reference_index],
            reference_tags=tags_list[reference_index],
            test_centers=centers_list[test_index],
            test_J=J_list[test_index],
            test_tags=tags_list[test_index],
            k=K_NEIGHBORS,
            distance_power=DISTANCE_POWER,
            match_tissue_tags=MATCH_TISSUE_TAGS
        )

        relative_errors.append(error)
        tetra_counts_for_plot.append(tetra_counts[test_index])
        file_names_for_plot.append(Path(files[test_index]).name)
        reference_file_names_for_plot.append(Path(files[reference_index]).name)

        print(f"  Relative current-density vector-field error: {error:.6f} %")
        print()


# ------------------------------------------------------------
# Sort results by tetrahedron count
# ------------------------------------------------------------

tetra_counts_for_plot = np.array(tetra_counts_for_plot)
relative_errors = np.array(relative_errors)
file_names_for_plot = np.array(file_names_for_plot)
reference_file_names_for_plot = np.array(reference_file_names_for_plot)

sort_indices = np.argsort(tetra_counts_for_plot)

tetra_counts_sorted = tetra_counts_for_plot[sort_indices]
relative_errors_sorted = relative_errors[sort_indices]
file_names_sorted = file_names_for_plot[sort_indices]
reference_file_names_sorted = reference_file_names_for_plot[sort_indices]


# ------------------------------------------------------------
# Print sorted results
# ------------------------------------------------------------

print("Sorted results:")
for count, error, name, ref_name in zip(
    tetra_counts_sorted,
    relative_errors_sorted,
    file_names_sorted,
    reference_file_names_sorted
):
    print(
        f"{count} tetrahedra kept | "
        f"{error:.6f} % | "
        f"test: {name} | "
        f"reference: {ref_name}"
    )


# ------------------------------------------------------------
# Plot graph
# ------------------------------------------------------------

plt.plot(tetra_counts_sorted, relative_errors_sorted, marker="o")
plt.xlabel("Number of tetrahedra kept")

if REFERENCE_MODE == "next_finer":
    plt.ylabel("Relative current-density difference to next finer mesh (%)")
    reference_label = "next finer mesh"
elif REFERENCE_MODE == "specific_file":
    plt.ylabel("Relative current-density error compared to reference (%)")
    reference_label = REFERENCE_FILE_NAME
else:
    plt.ylabel("Relative current-density error compared to reference (%)")
    reference_label = "mesh with most tetrahedra"

if COMPARE_ONLY_LIST:
    plt.title(f"Current-density difference in chosen tissue types\nReference: {reference_label}")
else:
    plt.title(f"Current-density difference\nReference: {reference_label}")

plt.grid(True)

# Force y-axis to start at 0
plt.ylim(bottom=0)


# ------------------------------------------------------------
# Save graph data
# ------------------------------------------------------------

graph_data_folder = Path.home() / "Desktop" / "meshcomparegraphdata"
graph_data_folder.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if COMPARE_ONLY_LIST:
    tissue_label = "chosen_tissues"
else:
    tissue_label = "all_tissues"

if REFERENCE_MODE == "specific_file":
    reference_name_for_file = Path(REFERENCE_FILE_NAME).stem
elif REFERENCE_MODE == "next_finer":
    reference_name_for_file = "next_finer"
else:
    reference_name_for_file = "most_tetrahedra"

graph_data_filename = (
    f"meshcompare_J_{tissue_label}_ref_{reference_name_for_file}_{timestamp}.csv"
)

graph_data_path = graph_data_folder / graph_data_filename

with open(graph_data_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "tetrahedra_kept",
        "relative_error_percent",
        "test_file",
        "reference_file",
        "reference_mode",
        "reference_label",
        "compare_only_list",
        "field"
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
            COMPARE_ONLY_LIST,
            "J"
        ])

print(f"Graph data saved to: {graph_data_path}")

plt.show()