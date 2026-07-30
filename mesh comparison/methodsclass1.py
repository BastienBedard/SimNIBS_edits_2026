from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from simnibs import mesh_io


#Settings:

# "nearest_neighbor" = use the nearest comparison-mesh tetrahedron center
# "knn" = distance-weighted k-nearest tetrahedron centers
# # "containing_tetrahedron" = find the comparison-mesh tetrahedron around
# each reference center and estimate E at that location using barycentric coordinates
COMPARISON_METHOD = "nearest_neighbor"

MESHCOMPARE_FOLDER = Path.home() / "Desktop" / "simnibs_compare"
CSV_OUTPUT_FOLDER = Path.home() / "Desktop" / "meshcomparegraphdata"

COMPARE_ONLY_LIST = True
USED_TAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 500]


# Options for reference mesh:
# "most_tetrahedra" = use the file with the most kept tetrahedra as reference
# "specific_file" = use REFERENCE_FILE_NAME as reference
# "next_finer" = compare each mesh to the mesh with the next higher tetrahedra count
REFERENCE_MODE = "most_tetrahedra"

# Used only if REFERENCE_MODE = "specific_file"
REFERENCE_FILE_NAME = "m2m_ernie5_EJV_1_scalar.msh"

MATCH_TISSUE_TAGS = True
INCLUDE_REFERENCE_SELF_COMPARISON = False

#KNN settings only
K_NEIGHBORS = 8
DISTANCE_POWER = 2

VALID_COMPARISON_METHODS = {
    "nearest_neighbor",
    "knn",
    "containing_tetrahedron",
}

VALID_REFERENCE_MODES = {
    "most_tetrahedra",
    "specific_file",
    "next_finer",
}


@dataclass
class TetraMeshData:
    """All retained tetrahedral data required for one comparison mesh."""

    path: Path
    mesh: object
    centers: np.ndarray
    volumes: np.ndarray
    E: np.ndarray
    tags: np.ndarray
    tet_nodes: np.ndarray
    tet_element_numbers: np.ndarray

    @property
    def tetrahedron_count(self):
        return len(self.E)


def extract_tetra_data(mesh, compare_only_list=True, used_tags=None):
    """Extract retained tetrahedral geometry and element E-field data."""
    if used_tags is None:
        used_tags = []

    tet_mask = mesh.elm.elm_type == 4
    tags_tet_all = mesh.elm.tag1[tet_mask]

    if compare_only_list:
        tissue_mask = np.isin(tags_tet_all, used_tags)
    else:
        tissue_mask = np.ones_like(tags_tet_all, dtype=bool)

    E_all = mesh.field["E"].value
    E_tet = E_all[tet_mask, :3][tissue_mask]
    tags_tet = tags_tet_all[tissue_mask]

    tet_nodes_all = mesh.elm.node_number_list[tet_mask, :4] - 1
    tet_nodes = tet_nodes_all[tissue_mask]

    tet_element_numbers_all = mesh.elm.elm_number[tet_mask]
    tet_element_numbers = tet_element_numbers_all[tissue_mask]

    coords = mesh.nodes.node_coord
    p0 = coords[tet_nodes[:, 0]]
    p1 = coords[tet_nodes[:, 1]]
    p2 = coords[tet_nodes[:, 2]]
    p3 = coords[tet_nodes[:, 3]]

    centers = (p0 + p1 + p2 + p3) / 4.0
    volumes = np.abs(
        np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
    ) / 6.0

    return centers, volumes, E_tet, tags_tet, tet_nodes, tet_element_numbers


class MeshComparisonMapper:
    """
    Maps an electric field from a comparison mesh onto reference points.

    Available methods
    -----------------
    nearest_neighbor
        Assign the E vector from the nearest comparison tetrahedron center.

    knn
        Use the original distance-weighted k-nearest-neighbor interpolation.

    containing_tetrahedron
        Find the comparison tetrahedron that contains each reference point,
        then linearly interpolate nodal E values using barycentric coordinates.

        Because SimNIBS stores E as element data, nodal E values are first
        constructed separately for each tissue by volume-weighted averaging of
        the adjacent tetrahedral E vectors. Keeping this recovery tissue-wise
        avoids averaging E across tissue boundaries.
    """

    def __init__(
        self,
        k=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
        match_tissue_tags=MATCH_TISSUE_TAGS,
    ):
        self.k = int(k)
        self.distance_power = float(distance_power)
        self.match_tissue_tags = bool(match_tissue_tags)

        if self.k < 1:
            raise ValueError("k must be at least 1.")

        if self.distance_power < 0:
            raise ValueError("distance_power must be non-negative.")

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

    def _center_based_mapping(
        self,
        reference_centers,
        reference_tags,
        test_centers,
        test_E,
        test_tags,
        k,
        distance_power,
    ):
        """
        Shared implementation for nearest-neighbor and KNN mapping.
        """
        test_E_on_reference = np.zeros((len(reference_centers), 3))
        valid_reference = np.zeros(len(reference_centers), dtype=bool)

        if self.match_tissue_tags:
            unique_reference_tags = np.unique(reference_tags)

            for tag in unique_reference_tags:
                ref_mask = reference_tags == tag
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

                k_effective = min(k, len(test_centers_tag))
                tree = cKDTree(test_centers_tag)
                distances, indices = tree.query(
                    ref_centers_tag,
                    k=k_effective
                )

                if k_effective == 1:
                    distances = distances[:, np.newaxis]
                    indices = indices[:, np.newaxis]

                weights = self._inverse_distance_weights(
                    distances,
                    distance_power
                )

                neighboring_E = test_E_tag[indices]
                estimated_E = np.sum(
                    weights[:, :, np.newaxis] * neighboring_E,
                    axis=1
                )

                test_E_on_reference[ref_mask] = estimated_E
                valid_reference[ref_mask] = True

        else:
            k_effective = min(k, len(test_centers))

            if k_effective == 0:
                return test_E_on_reference, valid_reference

            tree = cKDTree(test_centers)
            distances, indices = tree.query(
                reference_centers,
                k=k_effective
            )

            if k_effective == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]

            weights = self._inverse_distance_weights(
                distances,
                distance_power
            )

            neighboring_E = test_E[indices]
            test_E_on_reference = np.sum(
                weights[:, :, np.newaxis] * neighboring_E,
                axis=1
            )
            valid_reference[:] = True

        return test_E_on_reference, valid_reference

    def nearest_neighbor(
        self,
        reference_centers,
        reference_tags,
        test_centers,
        test_E,
        test_tags,
        **_,
    ):
        """
        Map E from the single nearest comparison tetrahedron center.
        """
        return self._center_based_mapping(
            reference_centers=reference_centers,
            reference_tags=reference_tags,
            test_centers=test_centers,
            test_E=test_E,
            test_tags=test_tags,
            k=1,
            distance_power=0,
        )

    def knn(
        self,
        reference_centers,
        reference_tags,
        test_centers,
        test_E,
        test_tags,
        **_,
    ):
        """
        Map E using the original distance-weighted KNN comparison logic.
        """
        return self._center_based_mapping(
            reference_centers=reference_centers,
            reference_tags=reference_tags,
            test_centers=test_centers,
            test_E=test_E,
            test_tags=test_tags,
            k=self.k,
            distance_power=self.distance_power,
        )

    @staticmethod
    def _recover_nodal_E(
        number_of_nodes,
        tet_nodes,
        tet_volumes,
        tet_E,
        tet_tags,
        match_tissue_tags,
    ):
        """
        Recover nodal E values from element E values.

        When tissue matching is enabled, a separate nodal field is produced
        for each tissue tag. This prevents adjacent tissues from being blended.
        """
        nodal_fields = {}

        if match_tissue_tags:
            groups = np.unique(tet_tags)
        else:
            groups = [None]

        for tag in groups:
            if tag is None:
                mask = np.ones(len(tet_E), dtype=bool)
                key = None
            else:
                mask = tet_tags == tag
                key = int(tag)

            nodes = tet_nodes[mask]
            volumes = tet_volumes[mask]
            fields = tet_E[mask]

            weighted_sum = np.zeros((number_of_nodes, 3), dtype=float)
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

            nodal_E = np.full((number_of_nodes, 3), np.nan, dtype=float)
            used_nodes = total_weight > 0
            nodal_E[used_nodes] = (
                weighted_sum[used_nodes]
                / total_weight[used_nodes, np.newaxis]
            )

            nodal_fields[key] = nodal_E

        return nodal_fields

    def containing_tetrahedron(
        self,
        reference_centers,
        reference_tags,
        test_mesh,
        test_E,
        test_tags,
        test_volumes,
        test_tet_nodes,
        test_tet_element_numbers,
        **_,
    ):
        """
        Map E by containing-tetrahedron barycentric interpolation.

        Reference centers outside the retained comparison volume, inside a
        tetrahedron excluded by USED_TAGS, or inside a nonmatching tissue are
        marked invalid and excluded from the global convergence error.
        """
        test_E_on_reference = np.zeros((len(reference_centers), 3))
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
            in enumerate(test_tet_element_numbers)
        }

        nodal_fields = self._recover_nodal_E(
            number_of_nodes=test_mesh.nodes.nr,
            tet_nodes=test_tet_nodes,
            tet_volumes=test_volumes,
            tet_E=test_E,
            tet_tags=test_tags,
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

            containing_tag = int(test_tags[local_index])

            if (
                self.match_tissue_tags
                and containing_tag != int(reference_tags[ref_index])
            ):
                skipped_tag += 1
                continue

            field_key = containing_tag if self.match_tissue_tags else None
            nodal_E = nodal_fields[field_key]
            nodes = test_tet_nodes[local_index]
            vertex_E = nodal_E[nodes]

            if not np.all(np.isfinite(vertex_E)):
                skipped_nodal += 1
                continue

            test_E_on_reference[ref_index] = (
                barycentric[ref_index] @ vertex_E
            )
            valid_reference[ref_index] = True

        if skipped_outside:
            print(
                "Warning: "
                f"{skipped_outside} reference centers were outside the "
                "comparison tetrahedral mesh and were skipped."
            )

        if skipped_filtered:
            print(
                "Warning: "
                f"{skipped_filtered} reference centers were inside comparison "
                "tetrahedra excluded by the selected tissue filter and were skipped."
            )

        if skipped_tag:
            print(
                "Warning: "
                f"{skipped_tag} reference centers were inside a different "
                "comparison-mesh tissue tag and were skipped."
            )

        if skipped_nodal:
            print(
                "Warning: "
                f"{skipped_nodal} reference centers could not be interpolated "
                "because one or more recovered nodal E values were unavailable."
            )

        return test_E_on_reference, valid_reference

    def estimate(self, method, **kwargs):
        """
        Dispatch to one of the three mapping methods.
        """
        if method not in VALID_COMPARISON_METHODS:
            raise ValueError(
                f"Invalid comparison method: {method!r}. "
                f"Use one of: {sorted(VALID_COMPARISON_METHODS)}"
            )

        return getattr(self, method)(**kwargs)

class MeshConvergenceComparison:
    """
    Complete mesh-convergence workflow for the three mapping methods.

    The class loads each mesh once, keeps the extracted tetrahedral data in
    memory, selects the requested reference mesh, computes the global
    volume-weighted relative L2 error, exports CSV data, and plots results.
    """

    def __init__(
        self,
        mesh_folder=MESHCOMPARE_FOLDER,
        csv_output_folder=CSV_OUTPUT_FOLDER,
        comparison_method=COMPARISON_METHOD,
        compare_only_list=COMPARE_ONLY_LIST,
        used_tags=USED_TAGS,
        reference_mode=REFERENCE_MODE,
        reference_file_name=REFERENCE_FILE_NAME,
        k_neighbors=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
        match_tissue_tags=MATCH_TISSUE_TAGS,
        include_reference_self_comparison=INCLUDE_REFERENCE_SELF_COMPARISON,
    ):
        self.mesh_folder = Path(mesh_folder)
        self.csv_output_folder = Path(csv_output_folder)
        self.comparison_method = comparison_method
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

        self.mapper = MeshComparisonMapper(
            k=self.k_neighbors,
            distance_power=self.distance_power,
            match_tissue_tags=self.match_tissue_tags,
        )

        self.meshes = []
        self.results = []
        self.reference_index = None
        self.timestamp = None

        self._validate_settings()

    def _validate_settings(self):
        if self.comparison_method not in VALID_COMPARISON_METHODS:
            raise ValueError(
                f"Invalid comparison method: {self.comparison_method!r}. "
                f"Use one of: {sorted(VALID_COMPARISON_METHODS)}"
            )

        if self.reference_mode not in VALID_REFERENCE_MODES:
            raise ValueError(
                f"Invalid reference mode: {self.reference_mode!r}. "
                f"Use one of: {sorted(VALID_REFERENCE_MODES)}"
            )

        if self.reference_mode == "specific_file" and not self.reference_file_name:
            raise ValueError(
                "reference_file_name is required when reference_mode is "
                "'specific_file'."
            )

    def discover_files(self):
        if not self.mesh_folder.exists():
            raise FileNotFoundError(f"Folder not found: {self.mesh_folder}")

        files = sorted(
            path
            for path in self.mesh_folder.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )

        if not files:
            raise FileNotFoundError(
                f"No usable files found in: {self.mesh_folder}"
            )

        return files

    def load_meshes(self):
        """Read and extract all meshes. Existing loaded data is replaced."""
        self.meshes = []
        files = self.discover_files()

        print("Files that will be compared:")
        for path in files:
            print(path)

        for path in files:
            print(f"Reading: {path}")
            mesh = mesh_io.read_msh(str(path))

            data = extract_tetra_data(
                mesh,
                compare_only_list=self.compare_only_list,
                used_tags=self.used_tags,
            )

            mesh_data = TetraMeshData(
                path=path,
                mesh=mesh,
                centers=data[0],
                volumes=data[1],
                E=data[2],
                tags=data[3],
                tet_nodes=data[4],
                tet_element_numbers=data[5],
            )

            self.meshes.append(mesh_data)
            print(f"Tetrahedra kept: {mesh_data.tetrahedron_count}")

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
                    f"in:\n{self.mesh_folder}\n\nFiles found were:\n{names}"
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

    def _print_settings(self):
        print()
        print("Reference mode:", self.reference_mode)

        if self.reference_mode in {"most_tetrahedra", "specific_file"}:
            reference = self.meshes[self.reference_index]
            print("Reference file:", reference.path)
            print("Reference tetrahedra kept:", reference.tetrahedron_count)
        else:
            print(
                "Each file is compared to the file with the next higher "
                "tetrahedron count."
            )
            print("The finest file is skipped because it has no next finer reference.")

        print("Comparison method:", self.comparison_method)
        if self.comparison_method == "knn":
            print("K nearest neighbors:", self.k_neighbors)
            print("Distance power:", self.distance_power)
        print("Match tissue tags:", self.match_tissue_tags)
        print("Compare only tissue types in list:", self.compare_only_list)
        print("Used tissue tags:", self.used_tags)
        print()

    def compute_error(self, reference, test):
        """Compute global volume-weighted relative L2 E-field error."""
        test_E_on_reference, valid_reference = self.mapper.estimate(
            method=self.comparison_method,
            reference_centers=reference.centers,
            reference_tags=reference.tags,
            test_centers=test.centers,
            test_volumes=test.volumes,
            test_E=test.E,
            test_tags=test.tags,
            test_mesh=test.mesh,
            test_tet_nodes=test.tet_nodes,
            test_tet_element_numbers=test.tet_element_numbers,
        )

        reference_E_valid = reference.E[valid_reference]
        test_E_valid = test_E_on_reference[valid_reference]
        reference_volumes_valid = reference.volumes[valid_reference]

        if len(reference_E_valid) == 0:
            raise ValueError("No valid reference points were available for comparison.")

        diff = reference_E_valid - test_E_valid
        diff_squared = np.sum(diff**2, axis=1)
        reference_squared = np.sum(reference_E_valid**2, axis=1)

        numerator = np.sqrt(np.sum(diff_squared * reference_volumes_valid))
        denominator = np.sqrt(
            np.sum(reference_squared * reference_volumes_valid)
        )

        if denominator == 0.0:
            raise ZeroDivisionError(
                "The volume-weighted reference E-field norm is zero, so a "
                "relative convergence error cannot be calculated."
            )

        return numerator / denominator * 100.0

    def compare_pair(self, test_index, reference_index):
        test = self.meshes[test_index]
        reference = self.meshes[reference_index]

        print("Comparing:")
        print("  Test file:     ", test.path)
        print("  Reference file:", reference.path)

        global_error = self.compute_error(reference, test)

        row = {
            "tetrahedra_kept": test.tetrahedron_count,
            "relative_error_percent": global_error,
            "test_file": test.path.name,
            "reference_file": reference.path.name,
        }
        self.results.append(row)

        print(f"  Relative vector-field error: {global_error:.6f} %")
        print()
        return row

    def run(self, reload_meshes=True):
        """Run the complete comparison and return sorted result dictionaries."""
        if reload_meshes or not self.meshes:
            self.load_meshes()

        self.select_reference()
        self._print_settings()
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.reference_mode in {"most_tetrahedra", "specific_file"}:
            for test_index in range(len(self.meshes)):
                if (
                    test_index == self.reference_index
                    and not self.include_reference_self_comparison
                ):
                    continue
                self.compare_pair(test_index, self.reference_index)

        else:
            counts = [mesh.tetrahedron_count for mesh in self.meshes]
            sorted_indices = np.argsort(counts)

            for position in range(len(sorted_indices) - 1):
                test_index = int(sorted_indices[position])
                reference_index = int(sorted_indices[position + 1])
                self.compare_pair(test_index, reference_index)

        self.results.sort(key=lambda row: row["tetrahedra_kept"])
        self.print_results()
        return self.results

    def run_method(self, method, reload_meshes=False):
        """Run another mapping method while reusing already loaded meshes."""
        if method not in VALID_COMPARISON_METHODS:
            raise ValueError(
                f"Invalid comparison method: {method!r}. "
                f"Use one of: {sorted(VALID_COMPARISON_METHODS)}"
            )

        self.comparison_method = method
        return self.run(reload_meshes=reload_meshes)

    def print_results(self):
        print("Sorted results:")
        for row in self.results:
            print(
                f"{row['tetrahedra_kept']} tetrahedra kept | "
                f"{row['relative_error_percent']:.6f} % | "
                f"test: {row['test_file']} | "
                f"reference: {row['reference_file']}"
            )

    def _reference_label(self):
        if self.reference_mode == "next_finer":
            return "next finer mesh"
        if self.reference_mode == "specific_file":
            return self.reference_file_name
        return "mesh with most tetrahedra"

    def _output_reference_name(self):
        if self.reference_mode == "specific_file":
            return Path(self.reference_file_name).stem
        return self.reference_mode

    def save_csv(self):
        if not self.results:
            raise RuntimeError("No results are available. Call run() first.")

        self.csv_output_folder.mkdir(parents=True, exist_ok=True)

        tissue_label = (
            "chosen_tissues" if self.compare_only_list else "all_tissues"
        )
        timestamp = self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"meshcompare_{self.comparison_method}_{tissue_label}_ref_"
            f"{self._output_reference_name()}_{timestamp}.csv"
        )
        output_path = self.csv_output_folder / filename

        with output_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "tetrahedra_kept",
                "relative_error_percent",
                "test_file",
                "reference_file",
                "reference_mode",
                "reference_label",
                "comparison_method",
                "compare_only_list",
            ])

            for row in self.results:
                writer.writerow([
                    row["tetrahedra_kept"],
                    row["relative_error_percent"],
                    row["test_file"],
                    row["reference_file"],
                    self.reference_mode,
                    self._reference_label(),
                    self.comparison_method,
                    self.compare_only_list,
                ])

        print(f"Graph data saved to: {output_path}")
        return output_path

    def plot(self, show=True):
        if not self.results:
            raise RuntimeError("No results are available. Call run() first.")

        tetra_counts = np.array(
            [row["tetrahedra_kept"] for row in self.results]
        )
        relative_errors = np.array(
            [row["relative_error_percent"] for row in self.results]
        )

        fig, ax = plt.subplots()
        ax.plot(tetra_counts, relative_errors, marker="o")
        ax.set_xlabel("Number of tetrahedra kept")

        if self.reference_mode == "next_finer":
            ax.set_ylabel(
                "Relative vector-field difference to next finer mesh (%)"
            )
        else:
            ax.set_ylabel("Relative vector-field error compared to reference (%)")

        if self.compare_only_list:
            title = "Electric-field difference in chosen tissue types"
        else:
            title = "Electric-field difference"

        ax.set_title(
            f"{title}\nMethod: {self.comparison_method} | "
            f"Reference: {self._reference_label()}"
        )
        ax.grid(True)
        ax.set_ylim(bottom=0)

        if show:
            plt.show()

        return fig, ax

    def run_and_export(self, show_plot=True, reload_meshes=True):
        """Convenience method for the original script's complete behavior."""
        self.run(reload_meshes=reload_meshes)
        csv_path = self.save_csv()
        self.plot(show=show_plot)
        return self.results, csv_path


def main():
    comparison = MeshConvergenceComparison()
    comparison.run_and_export()


if __name__ == "__main__":
    main()