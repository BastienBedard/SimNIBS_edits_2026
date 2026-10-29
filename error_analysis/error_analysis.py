"""
Mesh-convergence comparison of SimNIBS results (E, J and V fields).

This module is meant to be imported (for example from a notebook) and is never
run directly. It contains no file locations: every path is given by the caller.

Typical use
-----------
    comparison = MeshConvergenceComparison(
        mesh_folder=MESHCOMPARE_FOLDER,
        simulation_folders=["sim_m2m_ernie1", "sim_m2m_ernie2"],
        csv_output_folder=CSV_OUTPUT_FOLDER,
        fields=("E", "J", "V"),
    )
    results = comparison.run("nearest_neighbor", field="E")   # DataFrame
    comparison.plot(results)
    comparison.save_csv(results)
"""
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from simnibs import mesh_io


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

# Tissue tags kept when compare_only_list=True
# 1 WM, 2 GM, 3 CSF, 4 bone, 5 scalp, 6 eyes, 7 compact bone, 8 spongy bone,
# 9 blood, 10 muscle, 100 electrode, 500 saline/gel
DEFAULT_USED_TAGS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 500)

# Fields that can be compared. Name used here -> (name in the SimNIBS mesh,
# kind of data, label for titles).
#   E, J : element (per-tetrahedron) vectors
#   V    : node scalar (stored as "v"); converted to one value per
#          tetrahedron by averaging its four nodes
FIELD_SOURCES = {
    "E": ("E", "element", "Electric field"),
    "J": ("J", "element", "Current density"),
    "V": ("v", "node", "Electric potential"),
}

# "nearest_neighbor"       = value of the nearest comparison-mesh tetrahedron center
# "knn"                    = distance-weighted k-nearest tetrahedron centers
# "containing_tetrahedron" = barycentric interpolation inside the comparison-mesh
#                            tetrahedron that contains each reference center
VALID_COMPARISON_METHODS = (
    "nearest_neighbor",
    "knn",
    "containing_tetrahedron",
)

# "most_tetrahedra" = the mesh with the most kept tetrahedra is the reference
# "specific_file"   = reference_file_name is the reference
# "next_finer"      = each mesh is compared to the next finer mesh
VALID_REFERENCE_MODES = (
    "most_tetrahedra",
    "specific_file",
    "next_finer",
)

# Methods that need the full SimNIBS mesh object (not only extracted arrays).
# The raw mesh is read from disk on demand for these methods only.
METHODS_NEEDING_MESH = frozenset({"containing_tetrahedron"})

# Result mesh searched for inside each simulation folder
RESULT_FILE_PATTERN = "*_scalar.msh"


def _check_field_names(names):
    unknown = [name for name in names if name not in FIELD_SOURCES]
    if unknown:
        raise ValueError(
            f"Unknown field(s) {unknown}. Use any of: {list(FIELD_SOURCES)}"
        )


# ------------------------------------------------------------
# Mesh data
# ------------------------------------------------------------

@dataclass
class TetraMeshData:
    """
    Retained tetrahedral data of one result mesh.

    `values` maps a field name ("E", "J", "V") to an array of shape
    (number of retained tetrahedra, 3) for vectors or (..., 1) for V.

    The raw SimNIBS mesh object is deliberately not stored (it is large).
    Everything else is a copy, so the raw mesh is freed after loading.
    """

    path: Path
    centers: np.ndarray
    volumes: np.ndarray
    tags: np.ndarray
    tet_nodes: np.ndarray
    tet_element_numbers: np.ndarray
    values: dict = dataclass_field(default_factory=dict)

    @property
    def tetrahedron_count(self):
        return len(self.tags)

    @classmethod
    def from_mesh(
        cls,
        mesh,
        path,
        compare_only_list=True,
        used_tags=None,
        fields=("E",),
        remove_mean_potential=True,
    ):
        """
        Extract retained tetrahedral geometry and the requested field data.

        remove_mean_potential: V is defined up to an arbitrary constant, so
        the volume-weighted mean over the retained tetrahedra is subtracted
        from V of each mesh.
        """
        fields = tuple(fields)
        _check_field_names(fields)

        if used_tags is None:
            used_tags = []

        tet_mask = mesh.elm.elm_type == 4
        tags_tet_all = mesh.elm.tag1[tet_mask]

        if compare_only_list:
            tissue_mask = np.isin(tags_tet_all, used_tags)
        else:
            tissue_mask = np.ones_like(tags_tet_all, dtype=bool)

        tags_tet = tags_tet_all[tissue_mask]

        # SimNIBS node and element numbers are 1-based; node indices are 0-based.
        tet_nodes = (mesh.elm.node_number_list[tet_mask, :4] - 1)[tissue_mask]
        tet_element_numbers = mesh.elm.elm_number[tet_mask][tissue_mask]

        coords = mesh.nodes.node_coord
        p0 = coords[tet_nodes[:, 0]]
        p1 = coords[tet_nodes[:, 1]]
        p2 = coords[tet_nodes[:, 2]]
        p3 = coords[tet_nodes[:, 3]]

        centers = (p0 + p1 + p2 + p3) / 4.0
        volumes = np.abs(
            np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
        ) / 6.0

        values = {}
        for name in fields:
            mesh_name, kind, _ = FIELD_SOURCES[name]

            if mesh_name not in mesh.field:
                raise KeyError(
                    f"Field {mesh_name!r} (for {name}) not found in {path}. "
                    "Check the 'fields' setting of the tDCS simulation."
                )

            data = mesh.field[mesh_name].value

            if kind == "element":
                values[name] = np.asarray(
                    data[tet_mask, :3][tissue_mask], dtype=float
                )
            else:
                nodal = np.asarray(data, dtype=float).reshape(-1)
                tet_values = nodal[tet_nodes].mean(axis=1)

                if remove_mean_potential:
                    tet_values = tet_values - np.average(
                        tet_values, weights=volumes
                    )

                values[name] = tet_values[:, np.newaxis]

        return cls(
            path=Path(path),
            centers=centers,
            volumes=volumes,
            tags=tags_tet,
            tet_nodes=tet_nodes,
            tet_element_numbers=tet_element_numbers,
            values=values,
        )

    @classmethod
    def from_file(
        cls,
        path,
        compare_only_list=True,
        used_tags=None,
        fields=("E",),
        remove_mean_potential=True,
    ):
        mesh = mesh_io.read_msh(str(path))
        return cls.from_mesh(
            mesh,
            path,
            compare_only_list=compare_only_list,
            used_tags=used_tags,
            fields=fields,
            remove_mean_potential=remove_mean_potential,
        )


# ------------------------------------------------------------
# Error metric
# ------------------------------------------------------------

def relative_l2_error(reference_values, test_values, volumes):
    """
    Volume-weighted relative L2 error (vector or scalar field), in percent.

        100 * sqrt(sum(V * |ref - test|^2)) / sqrt(sum(V * |ref|^2))

    Arrays have shape (n, 3) for vectors or (n, 1) for scalars, and must
    already be restricted to the valid points.
    """
    diff_squared = np.sum((reference_values - test_values) ** 2, axis=1)
    reference_squared = np.sum(reference_values**2, axis=1)

    numerator = np.sqrt(np.sum(diff_squared * volumes))
    denominator = np.sqrt(np.sum(reference_squared * volumes))

    if denominator == 0.0:
        raise ZeroDivisionError(
            "The volume-weighted reference field norm is zero, so a "
            "relative convergence error cannot be calculated."
        )

    return numerator / denominator * 100.0


# ------------------------------------------------------------
# Mapping methods
# ------------------------------------------------------------

class MeshComparisonMapper:
    """
    Maps a field from a comparison (test) mesh onto the tetrahedron centers
    of a reference mesh. Both meshes are TetraMeshData objects, and `field`
    selects which entry of `values` is mapped ("E", "J" or "V").

    Available methods
    -----------------
    nearest_neighbor
        Assign the value of the nearest comparison tetrahedron center.

    knn
        Distance-weighted k-nearest-neighbor interpolation.

    containing_tetrahedron
        Find the comparison tetrahedron that contains each reference point,
        then linearly interpolate nodal values using barycentric coordinates.
        Needs the full SimNIBS mesh object of the test mesh (test_mesh).

        Because the field is stored per tetrahedron, nodal values are first
        constructed separately for each tissue by volume-weighted averaging of
        the adjacent tetrahedral values. Keeping this recovery tissue-wise
        avoids averaging across tissue boundaries.

    Every method returns (values_on_reference, valid_reference).
    """

    def __init__(self, k=8, distance_power=2, match_tissue_tags=True, verbose=True):
        self.k = int(k)
        self.distance_power = float(distance_power)
        self.match_tissue_tags = bool(match_tissue_tags)
        self.verbose = bool(verbose)

        if self.k < 1:
            raise ValueError("k must be at least 1.")

        if self.distance_power < 0:
            raise ValueError("distance_power must be non-negative.")

    def _log(self, *args):
        if self.verbose:
            print(*args)

    @staticmethod
    def _inverse_distance_weights(distances, distance_power):
        """
        Compute normalized inverse-distance weights.

        Exact zero-distance matches receive all the weight. Otherwise:

            w_j = (1 / d_j**p) / sum_j(1 / d_j**p)
        """
        weights = np.zeros_like(distances, dtype=float)

        zero_distance_mask = distances == 0.0
        has_zero_distance = np.any(zero_distance_mask, axis=1)

        if np.any(has_zero_distance):
            zero_rows = has_zero_distance
            weights[zero_rows] = zero_distance_mask[zero_rows].astype(float)
            weights[zero_rows] /= np.sum(
                weights[zero_rows],
                axis=1,
                keepdims=True
            )

        nonzero_rows = ~has_zero_distance

        if np.any(nonzero_rows):
            d = distances[nonzero_rows]

            if distance_power == 0:
                weights[nonzero_rows] = 1.0
            else:
                weights[nonzero_rows] = 1.0 / d**distance_power

            weights[nonzero_rows] /= np.sum(
                weights[nonzero_rows],
                axis=1,
                keepdims=True
            )

        return weights

    def _interpolate_from_neighbors(
        self, query_centers, test_centers, test_values, k, distance_power
    ):
        """Distance-weighted values at query_centers from the k nearest test centers."""
        k_effective = min(k, len(test_centers))
        tree = cKDTree(test_centers)
        distances, indices = tree.query(query_centers, k=k_effective)

        if k_effective == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        weights = self._inverse_distance_weights(distances, distance_power)

        return np.sum(weights[:, :, np.newaxis] * test_values[indices], axis=1)

    def _center_based_mapping(self, reference, test, k, distance_power, field):
        """Shared implementation for nearest-neighbor and KNN mapping."""
        test_values = test.values[field]
        values_on_reference = np.zeros(
            (len(reference.centers), test_values.shape[1])
        )
        valid_reference = np.zeros(len(reference.centers), dtype=bool)

        if not self.match_tissue_tags:
            if len(test.centers) == 0:
                return values_on_reference, valid_reference

            values_on_reference = self._interpolate_from_neighbors(
                reference.centers, test.centers, test_values, k, distance_power
            )
            valid_reference[:] = True
            return values_on_reference, valid_reference

        for tag in np.unique(reference.tags):
            ref_mask = reference.tags == tag
            test_mask = test.tags == tag

            if not np.any(test_mask):
                self._log(
                    f"Warning: no matching test tetrahedra for tissue tag {tag}. "
                    "These points are skipped."
                )
                continue

            values_on_reference[ref_mask] = self._interpolate_from_neighbors(
                reference.centers[ref_mask],
                test.centers[test_mask],
                test_values[test_mask],
                k,
                distance_power,
            )
            valid_reference[ref_mask] = True

        return values_on_reference, valid_reference

    def nearest_neighbor(self, reference, test, test_mesh=None, field="E"):
        """Map the field from the single nearest comparison tetrahedron center."""
        return self._center_based_mapping(
            reference, test, k=1, distance_power=0, field=field
        )

    def knn(self, reference, test, test_mesh=None, field="E"):
        """Map the field using distance-weighted k-nearest-neighbor interpolation."""
        return self._center_based_mapping(
            reference,
            test,
            k=self.k,
            distance_power=self.distance_power,
            field=field,
        )

    @staticmethod
    def _recover_nodal_values(
        number_of_nodes,
        tet_nodes,
        tet_volumes,
        tet_values,
        tet_tags,
        match_tissue_tags,
    ):
        """
        Recover nodal values from per-tetrahedron values.

        When tissue matching is enabled, a separate nodal field is produced
        for each tissue tag. This prevents adjacent tissues from being blended.
        """
        nodal_fields = {}
        dimension = tet_values.shape[1]

        if match_tissue_tags:
            groups = np.unique(tet_tags)
        else:
            groups = [None]

        for tag in groups:
            if tag is None:
                mask = np.ones(len(tet_values), dtype=bool)
                key = None
            else:
                mask = tet_tags == tag
                key = int(tag)

            nodes = tet_nodes[mask]
            volumes = tet_volumes[mask]
            fields = tet_values[mask]

            weighted_sum = np.zeros((number_of_nodes, dimension), dtype=float)
            total_weight = np.zeros(number_of_nodes, dtype=float)

            repeated_fields = np.repeat(fields, 4, axis=0)
            repeated_volumes = np.repeat(volumes, 4)
            flattened_nodes = nodes.reshape(-1)

            np.add.at(
                weighted_sum,
                flattened_nodes,
                repeated_fields * repeated_volumes[:, np.newaxis]
            )
            np.add.at(
                total_weight,
                flattened_nodes,
                repeated_volumes
            )

            nodal_values = np.full((number_of_nodes, dimension), np.nan, dtype=float)
            used_nodes = total_weight > 0
            nodal_values[used_nodes] = (
                weighted_sum[used_nodes]
                / total_weight[used_nodes, np.newaxis]
            )

            nodal_fields[key] = nodal_values

        return nodal_fields

    def containing_tetrahedron(self, reference, test, test_mesh=None, field="E"):
        """
        Map the field by containing-tetrahedron barycentric interpolation.

        Reference centers outside the retained comparison volume, inside a
        tetrahedron excluded by the tissue filter, or inside a nonmatching
        tissue are marked invalid and excluded from the global error.

        test_mesh must be the full SimNIBS mesh object of the comparison mesh.
        """
        if test_mesh is None:
            raise ValueError(
                "containing_tetrahedron needs the full SimNIBS mesh object "
                "(test_mesh), but None was given."
            )

        reference_centers = reference.centers
        test_values = test.values[field]
        values_on_reference = np.zeros(
            (len(reference_centers), test_values.shape[1])
        )
        valid_reference = np.zeros(len(reference_centers), dtype=bool)

        containing_elements, barycentric = (
            test_mesh.find_tetrahedron_with_points(
                reference_centers,
                compute_baricentric=True
            )
        )

        containing_elements = np.asarray(containing_elements, dtype=int)
        barycentric = np.asarray(barycentric, dtype=float)

        if barycentric.shape != (len(reference_centers), 4):
            raise ValueError(
                "SimNIBS returned unexpected barycentric-coordinate shape: "
                f"{barycentric.shape}; expected "
                f"({len(reference_centers)}, 4)."
            )

        # Convert full-mesh element number -> index in the retained tetra arrays.
        element_to_local = {
            int(element_number): local_index
            for local_index, element_number
            in enumerate(test.tet_element_numbers)
        }

        nodal_fields = self._recover_nodal_values(
            number_of_nodes=test_mesh.nodes.nr,
            tet_nodes=test.tet_nodes,
            tet_volumes=test.volumes,
            tet_values=test_values,
            tet_tags=test.tags,
            match_tissue_tags=self.match_tissue_tags,
        )

        skipped_outside = 0
        skipped_filtered = 0
        skipped_tag = 0
        skipped_nodal = 0

        for ref_index, element_number in enumerate(containing_elements):
            if element_number == -1:
                skipped_outside += 1
                continue

            local_index = element_to_local.get(int(element_number))

            if local_index is None:
                skipped_filtered += 1
                continue

            containing_tag = int(test.tags[local_index])

            if (
                self.match_tissue_tags
                and containing_tag != int(reference.tags[ref_index])
            ):
                skipped_tag += 1
                continue

            field_key = containing_tag if self.match_tissue_tags else None
            vertex_values = nodal_fields[field_key][test.tet_nodes[local_index]]

            if not np.all(np.isfinite(vertex_values)):
                skipped_nodal += 1
                continue

            values_on_reference[ref_index] = barycentric[ref_index] @ vertex_values
            valid_reference[ref_index] = True

        if skipped_outside:
            self._log(
                f"Warning: {skipped_outside} reference centers were outside "
                "the comparison tetrahedral mesh and were skipped."
            )

        if skipped_filtered:
            self._log(
                f"Warning: {skipped_filtered} reference centers were inside "
                "comparison tetrahedra excluded by the selected tissue filter "
                "and were skipped."
            )

        if skipped_tag:
            self._log(
                f"Warning: {skipped_tag} reference centers were inside a "
                "different comparison-mesh tissue tag and were skipped."
            )

        if skipped_nodal:
            self._log(
                f"Warning: {skipped_nodal} reference centers could not be "
                "interpolated because one or more recovered nodal values "
                "were unavailable."
            )

        return values_on_reference, valid_reference

    def estimate(self, method, reference, test, test_mesh=None, field="E"):
        """
        Dispatch to one of the three mapping methods.

        This is the only place where the method name is validated.
        """
        if method not in VALID_COMPARISON_METHODS:
            raise ValueError(
                f"Invalid comparison method: {method!r}. "
                f"Use one of: {list(VALID_COMPARISON_METHODS)}"
            )

        return getattr(self, method)(
            reference, test, test_mesh=test_mesh, field=field
        )


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_convergence(results, group_by="method", ax=None, title=None):
    """
    Plot relative error against number of tetrahedra, one line per group.

    `results` is a DataFrame returned by MeshConvergenceComparison.run() or
    one of the run_all_* methods, or several such DataFrames (or CSV files
    read with pandas.read_csv) joined with pandas.concat. `group_by` is the
    column that defines the lines (for example "method", "reference_mode"
    or "field").
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for label, group in results.groupby(group_by, sort=False):
        group = group.sort_values("tetrahedra_kept")
        ax.plot(
            group["tetrahedra_kept"],
            group["relative_error_percent"],
            marker="o",
            label=str(label),
        )

    ax.set_xlabel("Number of tetrahedra kept")

    if (
        "reference_mode" in results.columns
        and (results["reference_mode"] == "next_finer").all()
    ):
        ax.set_ylabel("Relative difference to next finer mesh (%)")
    else:
        ax.set_ylabel("Relative error compared to reference (%)")

    ax.set_title(title or "Mesh convergence")
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.legend(title=group_by)

    return fig, ax


# ------------------------------------------------------------
# Convergence workflow
# ------------------------------------------------------------

class MeshConvergenceComparison:
    """
    Mesh-convergence workflow for a listed set of simulation folders.

    mesh_folder        Folder containing the simulation sub-folders.
    simulation_folders Names of the sub-folders of mesh_folder to compare.
                       Each must contain exactly one *_scalar.msh result.
    csv_output_folder  Where save_csv() writes (only needed for saving).
    fields             Fields extracted when the meshes are loaded, any of
                       "E", "J", "V". run() can only compare these.

    The extracted tetrahedral data of every mesh is kept in memory. The raw
    SimNIBS mesh is not; for methods that need it (METHODS_NEEDING_MESH) it is
    read from disk for one test mesh at a time and freed right after use.

    run() and the run_all_* methods return pandas DataFrames and keep no
    hidden results, so the caller decides what to keep, plot or save.
    """

    def __init__(
        self,
        mesh_folder,
        simulation_folders,
        csv_output_folder=None,
        fields=("E",),
        compare_only_list=True,
        used_tags=DEFAULT_USED_TAGS,
        reference_mode="most_tetrahedra",
        reference_file_name=None,
        k_neighbors=8,
        distance_power=2,
        match_tissue_tags=True,
        include_reference_self_comparison=False,
        verbose=True,
    ):
        self.mesh_folder = Path(mesh_folder)
        self.simulation_folders = list(simulation_folders)
        self.csv_output_folder = (
            Path(csv_output_folder) if csv_output_folder is not None else None
        )
        self.fields = tuple(fields)
        self.compare_only_list = bool(compare_only_list)
        self.used_tags = list(used_tags)
        self.reference_mode = reference_mode
        self.reference_file_name = reference_file_name
        self.k_neighbors = int(k_neighbors)
        self.distance_power = float(distance_power)
        self.match_tissue_tags = bool(match_tissue_tags)
        self.include_reference_self_comparison = bool(
            include_reference_self_comparison
        )
        self.verbose = bool(verbose)

        self.mapper = MeshComparisonMapper(
            k=self.k_neighbors,
            distance_power=self.distance_power,
            match_tissue_tags=self.match_tissue_tags,
            verbose=self.verbose,
        )

        self.meshes = []
        self.reference_index = None

        self._validate_settings()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _validate_settings(self):
        if not self.simulation_folders:
            raise ValueError("simulation_folders is empty.")

        if not self.fields:
            raise ValueError("fields is empty.")

        _check_field_names(self.fields)

        if self.reference_mode not in VALID_REFERENCE_MODES:
            raise ValueError(
                f"Invalid reference mode: {self.reference_mode!r}. "
                f"Use one of: {list(VALID_REFERENCE_MODES)}"
            )

        if self.reference_mode == "specific_file" and not self.reference_file_name:
            raise ValueError(
                "reference_file_name is required when reference_mode is "
                "'specific_file'."
            )

    def discover_files(self):
        """Find the single result mesh inside each listed simulation folder."""
        files = []

        for name in self.simulation_folders:
            folder = self.mesh_folder / name

            if not folder.is_dir():
                raise FileNotFoundError(f"Simulation folder not found: {folder}")

            matches = sorted(folder.rglob(RESULT_FILE_PATTERN))

            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No {RESULT_FILE_PATTERN} result found in: {folder}"
                )

            if len(matches) > 1:
                names = "\n".join(f"  {match}" for match in matches)
                raise ValueError(
                    f"More than one {RESULT_FILE_PATTERN} result found in "
                    f"{folder}:\n{names}"
                )

            files.append(matches[0])

        return files

    def load_meshes(self):
        """Read every mesh and keep only the extracted tetrahedral data."""
        files = self.discover_files()
        self.meshes = []

        # self._log("Files that will be compared:")
        # for path in files:
        #     self._log(f"  {path}")

        for path in files:
            self._log(f"Reading: {path.name}")
            mesh_data = TetraMeshData.from_file(
                path,
                compare_only_list=self.compare_only_list,
                used_tags=self.used_tags,
                fields=self.fields,
            )
            self.meshes.append(mesh_data)
            self._log(f"  Tetrahedra kept: {mesh_data.tetrahedron_count}")

        return self.meshes

    def select_reference(self):
        if not self.meshes:
            raise RuntimeError("No meshes are loaded. Call load_meshes() first.")

        if self.reference_mode == "most_tetrahedra":
            counts = [mesh.tetrahedron_count for mesh in self.meshes]
            self.reference_index = int(np.argmax(counts))

        elif self.reference_mode == "specific_file":
            matches = [
                index
                for index, mesh in enumerate(self.meshes)
                if mesh.path.name == self.reference_file_name
            ]

            if not matches:
                names = "\n".join(mesh.path.name for mesh in self.meshes)
                raise FileNotFoundError(
                    f"Reference file {self.reference_file_name!r} was not found "
                    f"among the loaded meshes:\n{names}"
                )

            if len(matches) > 1:
                raise ValueError(
                    f"More than one file named {self.reference_file_name!r} "
                    "was found."
                )

            self.reference_index = matches[0]

        else:
            self.reference_index = None

        return self.reference_index

    def _comparison_pairs(self):
        """Return the (test_index, reference_index) pairs to compare."""
        if self.reference_mode in {"most_tetrahedra", "specific_file"}:
            pairs = [
                (test_index, self.reference_index)
                for test_index in range(len(self.meshes))
                if (
                    test_index != self.reference_index
                    or self.include_reference_self_comparison
                )
            ]
        else:
            counts = [mesh.tetrahedron_count for mesh in self.meshes]
            order = np.argsort(counts)
            pairs = [
                (int(order[position]), int(order[position + 1]))
                for position in range(len(order) - 1)
            ]

        if not pairs:
            raise ValueError(
                "No mesh pairs to compare. At least two meshes are needed."
            )

        return pairs

    def _log_settings(self, method, field):
        self._log()
        self._log("Field:", field)
        self._log("Reference mode:", self.reference_mode)

        if self.reference_mode in {"most_tetrahedra", "specific_file"}:
            reference = self.meshes[self.reference_index]
            self._log("Reference file:", reference.path.name)
            self._log("Reference tetrahedra kept:", reference.tetrahedron_count)
        else:
            self._log(
                "Each mesh is compared to the mesh with the next higher "
                "tetrahedron count; the finest mesh is skipped."
            )

        self._log("Comparison method:", method)
        if method == "knn":
            self._log("K nearest neighbors:", self.k_neighbors)
            self._log("Distance power:", self.distance_power)
        self._log("Match tissue tags:", self.match_tissue_tags)
        self._log("Compare only tissue types in list:", self.compare_only_list)
        self._log("Used tissue tags:", self.used_tags)
        self._log()

    # --------------------------------------------------------
    # Error computation
    # --------------------------------------------------------

    def compute_error(self, reference, test, method, field="E"):
        """
        Map the test field onto the reference and compute the global error.

        Returns (error_percent, coverage_percent, coverage_volume_percent).
        Coverage is the share of reference tetrahedra (by count, and by
        volume) that could be mapped; only those enter the error.
        """
        # The full mesh is read from disk only for methods that need it.
        if method in METHODS_NEEDING_MESH:
            test_mesh = mesh_io.read_msh(str(test.path))
        else:
            test_mesh = None

        values_on_reference, valid_reference = self.mapper.estimate(
            method, reference, test, test_mesh=test_mesh, field=field
        )

        del test_mesh  # free the raw mesh before the error calculation

        if not np.any(valid_reference):
            raise ValueError("No valid reference points were available for comparison.")

        volumes_valid = reference.volumes[valid_reference]

        error_percent = relative_l2_error(
            reference.values[field][valid_reference],
            values_on_reference[valid_reference],
            volumes_valid,
        )
        coverage_percent = 100.0 * len(volumes_valid) / reference.tetrahedron_count
        coverage_volume_percent = (
            100.0 * np.sum(volumes_valid) / np.sum(reference.volumes)
        )

        return error_percent, coverage_percent, coverage_volume_percent

    def compare_pair(self, test_index, reference_index, method, field="E"):
        test = self.meshes[test_index]
        reference = self.meshes[reference_index]

        # self._log(f"Comparing {test.path.name}  vs reference {reference.path.name}")

        error, coverage, coverage_volume = self.compute_error(
            reference, test, method, field
        )

        # self._log(
        #     f"  Relative {field} error: {error:.6f} % | "
        #     f"coverage: {coverage:.2f} % of tetrahedra, "
        #     f"{coverage_volume:.2f} % of volume"
        # )

        return {
            "method": method,
            "field": field,
            "tetrahedra_kept": test.tetrahedron_count,
            "relative_error_percent": error,
            "coverage_percent": coverage,
            "coverage_volume_percent": coverage_volume,
            "test_file": test.path.name,
            "reference_file": reference.path.name,
            "reference_mode": self.reference_mode,
            "compare_only_list": self.compare_only_list,
            "match_tissue_tags": self.match_tissue_tags,
            "k_neighbors": self.k_neighbors if method == "knn" else None,
            "distance_power": self.distance_power if method == "knn" else None,
        }

    # --------------------------------------------------------
    # Public workflow
    # --------------------------------------------------------

    def run(self, method, field="E", reload_meshes=False):
        """
        Compare every mesh with the reference using one mapping method and
        one field ("E", "J" or "V").

        Meshes are read on the first call and reused afterwards. Pass
        reload_meshes=True after changing compare_only_list, used_tags or
        fields. Returns a DataFrame sorted by number of tetrahedra.

        The method name is validated by MeshComparisonMapper.estimate().
        """
        if field not in self.fields:
            raise ValueError(
                f"Field {field!r} was not requested. This comparison was "
                f"created with fields={self.fields}; create it with the "
                "field included."
            )

        if reload_meshes or not self.meshes:
            self.load_meshes()

        self.select_reference()
        # self._log_settings(method, field)

        rows = [
            self.compare_pair(test_index, reference_index, method, field)
            for test_index, reference_index in self._comparison_pairs()
        ]

        return (
            pd.DataFrame(rows)
            .sort_values("tetrahedra_kept")
            .reset_index(drop=True)
        )

    def run_all_methods(self, methods=None, field="E", reload_meshes=False):
        """
        Run several mapping methods (all three by default) on the same meshes.

        Returns one DataFrame with a "method" column.
        """
        methods = list(
            VALID_COMPARISON_METHODS if methods is None else methods
        )

        frames = [
            self.run(
                method,
                field=field,
                reload_meshes=reload_meshes and position == 0,
            )
            for position, method in enumerate(methods)
        ]

        return pd.concat(frames, ignore_index=True)

    def run_all_fields(self, method, fields=None, reload_meshes=False):
        """
        Run one mapping method on several fields (all loaded fields by default).

        Returns one DataFrame with a "field" column.
        """
        fields = list(self.fields if fields is None else fields)

        frames = [
            self.run(
                method,
                field=name,
                reload_meshes=reload_meshes and position == 0,
            )
            for position, name in enumerate(fields)
        ]

        return pd.concat(frames, ignore_index=True)

    def run_all_reference_modes(
        self,
        method,
        reference_modes=("most_tetrahedra", "next_finer"),
        field="E",
        reload_meshes=False,
    ):
        """
        Run one mapping method with several reference modes.

        Returns one DataFrame with a "reference_mode" column. The comparison's
        own reference_mode is restored afterwards.
        """
        invalid = [m for m in reference_modes if m not in VALID_REFERENCE_MODES]
        if invalid:
            raise ValueError(
                f"Invalid reference mode(s): {invalid}. "
                f"Use any of: {list(VALID_REFERENCE_MODES)}"
            )

        original_mode = self.reference_mode
        frames = []

        try:
            for position, mode in enumerate(reference_modes):
                self.reference_mode = mode
                frames.append(
                    self.run(
                        method,
                        field=field,
                        reload_meshes=reload_meshes and position == 0,
                    )
                )
        finally:
            self.reference_mode = original_mode

        return pd.concat(frames, ignore_index=True)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    def _reference_label(self, results):
        modes = list(results["reference_mode"].unique())

        if len(modes) > 1:
            return None
        if modes[0] == "next_finer":
            return "next finer mesh"
        if modes[0] == "specific_file":
            return str(results["reference_file"].iloc[0])
        return "mesh with most tetrahedra"

    def plot(self, results, group_by="method", ax=None):
        """Plot a results DataFrame; returns (fig, ax). Call plt.show() after."""
        labels = [
            FIELD_SOURCES[name][2] for name in results["field"].unique()
        ]
        title = " / ".join(labels) + " difference"

        if self.compare_only_list:
            title += " in chosen tissue types"

        reference_label = self._reference_label(results)
        if reference_label is not None:
            title += f"\nReference: {reference_label}"

        return plot_convergence(
            results,
            group_by=group_by,
            ax=ax,
            title=title,
        )

    def save_csv(self, results, filename=None):
        """Save a results DataFrame to csv_output_folder and return the path."""
        if self.csv_output_folder is None:
            raise ValueError(
                "csv_output_folder was not given when the comparison was created."
            )

        self.csv_output_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            methods = list(results["method"].unique())
            if set(methods) == set(VALID_COMPARISON_METHODS):
                method_label = "all_methods"
            else:
                method_label = "-".join(methods)

            field_label = "-".join(results["field"].unique())

            tissue_label = (
                "chosen_tissues" if self.compare_only_list else "all_tissues"
            )

            modes = list(results["reference_mode"].unique())
            if len(modes) > 1:
                reference_label = "multiple_refs"
            elif modes[0] == "specific_file":
                reference_label = Path(results["reference_file"].iloc[0]).stem
            else:
                reference_label = modes[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"meshcompare_{method_label}_{field_label}_{tissue_label}_ref_"
                f"{reference_label}_{timestamp}.csv"
            )

        output_path = self.csv_output_folder / filename
        results.to_csv(output_path, index=False)

        self._log(f"Results saved to: {output_path}")
        return output_path