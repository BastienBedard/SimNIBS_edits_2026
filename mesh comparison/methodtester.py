from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
from simnibs import mesh_io

from methodsclass1 import (
    TetraMeshData,
    extract_tetra_data,
    MeshComparisonMapper,
    VALID_COMPARISON_METHODS,
    MESHCOMPARE_FOLDER,
    CSV_OUTPUT_FOLDER,
    COMPARE_ONLY_LIST,
    USED_TAGS,
    REFERENCE_FILE_NAME,
    MATCH_TISSUE_TAGS,
    K_NEIGHBORS,
    DISTANCE_POWER,
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

# Run all three real methods from MeshComparisonMapper.
VALIDATION_METHODS = [
    "nearest_neighbor",
    "knn",
    "containing_tetrahedron",
]

# Mesh used as the locations and volume weights at which errors are measured.
VALIDATION_REFERENCE_FILE = "m2m_ernie5_EJV_1_scalar.msh"

# Mesh supplying the field that must be mapped onto the reference mesh.
# Set to a filename, or leave as None to automatically use the retained mesh
# with the fewest tetrahedra other than the reference.
VALIDATION_TEST_FILE = None

VALIDATION_OUTPUT_FOLDER = CSV_OUTPUT_FOLDER / "method_validation"

# Gaussian test settings, in millimetres.
GAUSSIAN_SIGMA_MM = 12.0
GAUSSIAN_AMPLITUDE = 20

# Numerical pass tolerances for tests with exact expected percentages.
EXACT_PERCENT_TOLERANCE = 1e-8

# These tests use the same analytical physical field on different meshes.
# Their true physical difference is zero, but their reported value measures
# mapping/interpolation error; therefore they are not assigned a strict pass
# threshold by default.
MAPPING_TEST_NAMES = {
    "constant_same_field_different_meshes",
    "linear_same_field_different_meshes",
    "gaussian_same_field_different_meshes",
}


@dataclass(frozen=True)
class ValidationCase:
    """One controlled comparison with independently known field functions."""

    name: str
    description: str
    reference_field_function: object
    test_field_function: object
    use_different_meshes: bool
    exact_expected_result: bool = False


class MeshComparisonMethodValidator:
    """
    Validate the real MeshComparisonMapper methods with controlled fields.

    The validator does not reimplement nearest-neighbour, KNN, or
    containing-tetrahedron mapping. It calls MeshComparisonMapper.estimate(),
    which dispatches to the actual methods used by the convergence workflow.

    For every test it calculates:

    expected_error_percent
        The true field difference, evaluated directly at reference centers.

    reported_error_percent
        The field difference after the selected method maps test-mesh values
        onto the reference centers.

    mapping_error_percent
        The difference between the mapped test field and the exact test field
        evaluated directly at the reference centers.

    coverage_percent
        Percentage of retained reference tetrahedra successfully mapped.
    """

    def __init__(
        self,
        mesh_folder=MESHCOMPARE_FOLDER,
        output_folder=VALIDATION_OUTPUT_FOLDER,
        methods=VALIDATION_METHODS,
        reference_file_name=VALIDATION_REFERENCE_FILE,
        test_file_name=VALIDATION_TEST_FILE,
        compare_only_list=COMPARE_ONLY_LIST,
        used_tags=USED_TAGS,
        k_neighbors=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
        match_tissue_tags=MATCH_TISSUE_TAGS,
    ):
        self.mesh_folder = Path(mesh_folder)
        self.output_folder = Path(output_folder)
        self.methods = list(methods)
        self.reference_file_name = reference_file_name
        self.test_file_name = test_file_name
        self.compare_only_list = bool(compare_only_list)
        self.used_tags = list(used_tags)

        invalid_methods = set(self.methods) - VALID_COMPARISON_METHODS
        if invalid_methods:
            raise ValueError(
                f"Invalid validation methods: {sorted(invalid_methods)}"
            )

        self.mapper = MeshComparisonMapper(
            k=k_neighbors,
            distance_power=distance_power,
            match_tissue_tags=match_tissue_tags,
        )

        self.loaded_meshes = {}
        self.reference = None
        self.test = None
        self.results = []
        self.timestamp = None

    def _load_one(self, path):
        print(f"Reading validation mesh: {path}")
        mesh = mesh_io.read_msh(str(path))
        data = extract_tetra_data(
            mesh,
            compare_only_list=self.compare_only_list,
            used_tags=self.used_tags,
        )
        return TetraMeshData(
            path=path,
            mesh=mesh,
            centers=data[0],
            volumes=data[1],
            E=data[2],
            tags=data[3],
            tet_nodes=data[4],
            tet_element_numbers=data[5],
        )

    def load_meshes(self):
        """Load all candidate meshes once and choose the validation pair."""
        if not self.mesh_folder.exists():
            raise FileNotFoundError(
                f"Validation mesh folder not found: {self.mesh_folder}"
            )

        paths = sorted(self.mesh_folder.glob("*.msh"))
        if not paths:
            raise FileNotFoundError(
                f"No .msh files were found in: {self.mesh_folder}"
            )

        self.loaded_meshes = {
            path.name: self._load_one(path)
            for path in paths
        }

        if self.reference_file_name not in self.loaded_meshes:
            available = "\n".join(sorted(self.loaded_meshes))
            raise FileNotFoundError(
                f"Validation reference file {self.reference_file_name!r} "
                f"was not found.\nAvailable files:\n{available}"
            )

        self.reference = self.loaded_meshes[self.reference_file_name]

        if self.test_file_name is not None:
            if self.test_file_name not in self.loaded_meshes:
                available = "\n".join(sorted(self.loaded_meshes))
                raise FileNotFoundError(
                    f"Validation test file {self.test_file_name!r} "
                    f"was not found.\nAvailable files:\n{available}"
                )
            self.test = self.loaded_meshes[self.test_file_name]
        else:
            alternatives = [
                mesh
                for name, mesh in self.loaded_meshes.items()
                if name != self.reference_file_name
            ]
            if not alternatives:
                raise ValueError(
                    "At least two .msh files are required for the "
                    "different-mesh validation tests."
                )
            self.test = min(
                alternatives,
                key=lambda mesh: mesh.tetrahedron_count,
            )

        print()
        print("Validation reference:", self.reference.path.name)
        print(
            "Reference retained tetrahedra:",
            self.reference.tetrahedron_count,
        )
        print("Validation test mesh:", self.test.path.name)
        print("Test retained tetrahedra:", self.test.tetrahedron_count)
        print()

    @staticmethod
    def _domain_parameters(reference, test):
        """Return shared center and length scale for analytical fields."""
        all_centers = np.vstack([reference.centers, test.centers])
        center = np.mean(all_centers, axis=0)
        extent = np.ptp(all_centers, axis=0)
        length_scale = float(np.max(extent))
        if length_scale == 0.0:
            length_scale = 1.0
        return center, length_scale

    def build_cases(self):
        center, length_scale = self._domain_parameters(
            self.reference,
            self.test,
        )

        constant_vector = np.array([1.0, 0.5, -0.25], dtype=float)

        def constant_field(points, tags=None):
            return np.repeat(
                constant_vector[np.newaxis, :],
                len(points),
                axis=0,
            )

        def scaled_constant_field(points, tags=None):
            return 1.05 * constant_field(points, tags)

        def opposite_constant_field(points, tags=None):
            return -constant_field(points, tags)

        def rotated_constant_field(points, tags=None):
            # Exact 90-degree rotation about the z axis.
            base = constant_field(points, tags)
            rotated = base.copy()
            rotated[:, 0] = -base[:, 1]
            rotated[:, 1] = base[:, 0]
            return rotated

        def linear_field(points, tags=None):
            q = (points - center) / length_scale
            x, y, z = q[:, 0], q[:, 1], q[:, 2]
            return np.column_stack([
                1.0 + 0.8 * x + 0.2 * y,
                0.5 - 0.3 * x + 0.7 * z,
                -0.25 + 0.4 * y - 0.6 * z,
            ])

        def gaussian_field(points, tags=None):
            displacement = points - center
            radius_squared = np.sum(displacement**2, axis=1)
            hotspot = GAUSSIAN_AMPLITUDE * np.exp(
                -radius_squared / (2.0 * GAUSSIAN_SIGMA_MM**2)
            )
            field = constant_field(points, tags)
            field[:, 0] += hotspot
            field[:, 1] += 0.5 * hotspot
            return field

        return [
            ValidationCase(
                name="identical_constant_field",
                description=(
                    "Identical mesh and identical constant vector field; "
                    "expected error is 0%."
                ),
                reference_field_function=constant_field,
                test_field_function=constant_field,
                use_different_meshes=False,
                exact_expected_result=True,
            ),
            ValidationCase(
                name="uniform_scale_5_percent",
                description=(
                    "Identical mesh; test field is 1.05 times the reference; "
                    "expected relative vector L2 error is 5%."
                ),
                reference_field_function=constant_field,
                test_field_function=scaled_constant_field,
                use_different_meshes=False,
                exact_expected_result=True,
            ),
            ValidationCase(
                name="opposite_vectors",
                description=(
                    "Identical mesh and opposite vectors; expected relative "
                    "vector L2 error is 200%."
                ),
                reference_field_function=constant_field,
                test_field_function=opposite_constant_field,
                use_different_meshes=False,
                exact_expected_result=True,
            ),
            ValidationCase(
                name="rotation_90_degrees",
                description=(
                    "Identical mesh; every vector is rotated 90 degrees about "
                    "the z axis. The expected value is calculated directly."
                ),
                reference_field_function=constant_field,
                test_field_function=rotated_constant_field,
                use_different_meshes=False,
                exact_expected_result=True,
            ),
            ValidationCase(
                name="constant_same_field_different_meshes",
                description=(
                    "Different meshes with the same constant field. The true "
                    "physical difference is 0%; reported error is mapping error."
                ),
                reference_field_function=constant_field,
                test_field_function=constant_field,
                use_different_meshes=True,
            ),
            ValidationCase(
                name="linear_same_field_different_meshes",
                description=(
                    "Different meshes with the same spatially linear field. "
                    "The true physical difference is 0%."
                ),
                reference_field_function=linear_field,
                test_field_function=linear_field,
                use_different_meshes=True,
            ),
            ValidationCase(
                name="gaussian_same_field_different_meshes",
                description=(
                    "Different meshes with the same smooth Gaussian hotspot. "
                    "The true physical difference is 0%."
                ),
                reference_field_function=gaussian_field,
                test_field_function=gaussian_field,
                use_different_meshes=True,
            ),
        ]

    @staticmethod
    def _relative_l2_percent(reference_E, comparison_E, volumes):
        if len(reference_E) == 0:
            raise ValueError("Cannot calculate an error from zero points.")

        difference_squared = np.sum(
            (reference_E - comparison_E) ** 2,
            axis=1,
        )
        reference_squared = np.sum(reference_E**2, axis=1)

        numerator = np.sqrt(np.sum(difference_squared * volumes))
        denominator = np.sqrt(np.sum(reference_squared * volumes))

        if denominator == 0.0:
            raise ZeroDivisionError(
                "The analytical reference field has zero weighted norm."
            )

        return float(numerator / denominator * 100.0)

    def _map_with_actual_method(
        self,
        method,
        reference_mesh_data,
        test_mesh_data,
        test_E,
    ):
        """Call the unchanged method through MeshComparisonMapper.estimate()."""
        return self.mapper.estimate(
            method=method,
            reference_centers=reference_mesh_data.centers,
            reference_tags=reference_mesh_data.tags,
            test_centers=test_mesh_data.centers,
            test_volumes=test_mesh_data.volumes,
            test_E=test_E,
            test_tags=test_mesh_data.tags,
            test_mesh=test_mesh_data.mesh,
            test_tet_nodes=test_mesh_data.tet_nodes,
            test_tet_element_numbers=(
                test_mesh_data.tet_element_numbers
            ),
        )

    def run_case(self, case, method):
        if case.use_different_meshes:
            test_mesh_data = self.test
        else:
            test_mesh_data = self.reference

        reference_E = case.reference_field_function(
            self.reference.centers,
            self.reference.tags,
        )
        test_E_at_test_centers = case.test_field_function(
            test_mesh_data.centers,
            test_mesh_data.tags,
        )

        # Independent truth: evaluate the intended test field directly at the
        # reference locations without using any mapping method.
        exact_test_E_at_reference = case.test_field_function(
            self.reference.centers,
            self.reference.tags,
        )

        expected_error = self._relative_l2_percent(
            reference_E,
            exact_test_E_at_reference,
            self.reference.volumes,
        )

        mapped_E, valid = self._map_with_actual_method(
            method=method,
            reference_mesh_data=self.reference,
            test_mesh_data=test_mesh_data,
            test_E=test_E_at_test_centers,
        )

        valid_count = int(np.count_nonzero(valid))
        total_count = len(valid)
        if valid_count == 0:
            raise ValueError(
                f"{method} produced no valid mapped points for {case.name}."
            )

        ref_valid = reference_E[valid]
        exact_test_valid = exact_test_E_at_reference[valid]
        mapped_valid = mapped_E[valid]
        volumes_valid = self.reference.volumes[valid]

        # Recalculate expected error on exactly the same valid subset used by
        # the method, so exclusion/coverage cannot distort the comparison.
        expected_on_valid = self._relative_l2_percent(
            ref_valid,
            exact_test_valid,
            volumes_valid,
        )
        reported_error = self._relative_l2_percent(
            ref_valid,
            mapped_valid,
            volumes_valid,
        )
        mapping_error = self._relative_l2_percent(
            exact_test_valid,
            mapped_valid,
            volumes_valid,
        )

        absolute_deviation = abs(reported_error - expected_on_valid)
        if expected_on_valid == 0.0:
            relative_deviation = np.nan
        else:
            relative_deviation = (
                absolute_deviation / expected_on_valid * 100.0
            )

        coverage = valid_count / total_count * 100.0

        if case.exact_expected_result:
            passed = absolute_deviation <= EXACT_PERCENT_TOLERANCE
            pass_label = "PASS" if passed else "FAIL"
        else:
            passed = None
            pass_label = "MEASURE"

        row = {
            "test_name": case.name,
            "method": method,
            "mesh_pair": (
                f"{self.reference.path.name} <- {test_mesh_data.path.name}"
            ),
            "expected_error_percent_all_points": expected_error,
            "expected_error_percent_valid_points": expected_on_valid,
            "reported_error_percent": reported_error,
            "absolute_deviation_percentage_points": absolute_deviation,
            "relative_deviation_percent": relative_deviation,
            "mapping_error_percent": mapping_error,
            "coverage_percent": coverage,
            "valid_reference_points": valid_count,
            "total_reference_points": total_count,
            "status": pass_label,
            "description": case.description,
        }

        print(
            f"{case.name:43s} | {method:23s} | "
            f"expected {expected_on_valid:10.6f}% | "
            f"reported {reported_error:10.6f}% | "
            f"mapping {mapping_error:10.6f}% | "
            f"coverage {coverage:7.3f}% | {pass_label}"
        )
        return row

    def run(self, reload_meshes=True):
        if reload_meshes or self.reference is None or self.test is None:
            self.load_meshes()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

        print("Running controlled validation tests")
        print("-" * 145)

        for case in self.build_cases():
            for method in self.methods:
                self.results.append(self.run_case(case, method))

        print("-" * 145)
        return self.results

    def save_csv(self):
        if not self.results:
            raise RuntimeError("No validation results exist. Call run() first.")

        self.output_folder.mkdir(parents=True, exist_ok=True)
        timestamp = self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            self.output_folder
            / f"mesh_comparison_method_validation_{timestamp}.csv"
        )

        fieldnames = list(self.results[0].keys())
        with output_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print()
        print(f"Validation CSV saved to: {output_path}")
        return output_path

    def plot(self, show=True):
        if not self.results:
            raise RuntimeError("No validation results exist. Call run() first.")

        mapping_rows = [
            row
            for row in self.results
            if row["test_name"] in MAPPING_TEST_NAMES
        ]

        test_names = [
            "constant_same_field_different_meshes",
            "linear_same_field_different_meshes",
            "gaussian_same_field_different_meshes",
        ]
        x = np.arange(len(test_names), dtype=float)
        width = 0.24

        fig, ax = plt.subplots(figsize=(11, 6))

        for method_index, method in enumerate(self.methods):
            values = []
            for test_name in test_names:
                matching = [
                    row["mapping_error_percent"]
                    for row in mapping_rows
                    if row["method"] == method
                    and row["test_name"] == test_name
                ]
                values.append(matching[0] if matching else np.nan)

            offset = (method_index - (len(self.methods) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=method)

        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Constant", "Linear", "Gaussian hotspot"]
        )
        ax.set_ylabel("Mapping error (%)")
        ax.set_title(
            "Actual mesh-comparison methods on controlled analytical fields"
        )
        ax.grid(True, axis="y")
        ax.legend()
        fig.tight_layout()

        output_path = (
            self.output_folder
            / f"mesh_comparison_method_validation_{self.timestamp}.png"
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Validation plot saved to: {output_path}")

        if show:
            plt.show()

        return fig, ax, output_path

    def run_and_export(self, show_plot=True):
        self.run()
        csv_path = self.save_csv()
        _, _, plot_path = self.plot(show=show_plot)
        return self.results, csv_path, plot_path


def main():
    validator = MeshComparisonMethodValidator()
    validator.run_and_export(show_plot=True)


if __name__ == "__main__":
    main()