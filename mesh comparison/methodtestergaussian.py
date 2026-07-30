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
    MATCH_TISSUE_TAGS,
    K_NEIGHBORS,
    DISTANCE_POWER,
)


# ============================================================
# GAUSSIAN HOTSPOT VALIDATION SETTINGS
# ============================================================

# Run the Gaussian hotspot test with these actual mapping methods.
VALIDATION_METHODS = [
    "nearest_neighbor",
    "knn",
    "containing_tetrahedron",
]

# Mesh whose tetrahedron centers and volumes are used for measurement.
VALIDATION_REFERENCE_FILE = "m2m_ernie5_EJV_1_scalar.msh"

# Mesh supplying the Gaussian field that is mapped to the reference mesh.
# Set a filename explicitly, or leave as None to automatically select the
# retained mesh with the fewest tetrahedra other than the reference mesh.
VALIDATION_TEST_FILE = None

VALIDATION_OUTPUT_FOLDER = CSV_OUTPUT_FOLDER / "gaussian_hotspot_validation"

# Gaussian hotspot settings, in millimetres.
GAUSSIAN_SIGMA_MM = 12.0
GAUSSIAN_AMPLITUDE = 20.0


class GaussianHotspotValidator:
    """Validate mesh-mapping methods using one smooth Gaussian hotspot."""

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
        """Load the reference mesh and select the test mesh."""
        if not self.mesh_folder.exists():
            raise FileNotFoundError(
                f"Validation mesh folder not found: {self.mesh_folder}"
            )

        paths = sorted(self.mesh_folder.glob("*.msh"))
        if not paths:
            raise FileNotFoundError(
                f"No .msh files were found in: {self.mesh_folder}"
            )

        meshes = {path.name: self._load_one(path) for path in paths}

        if self.reference_file_name not in meshes:
            available = "\n".join(sorted(meshes))
            raise FileNotFoundError(
                f"Validation reference file {self.reference_file_name!r} "
                f"was not found.\nAvailable files:\n{available}"
            )

        self.reference = meshes[self.reference_file_name]

        if self.test_file_name is not None:
            if self.test_file_name not in meshes:
                available = "\n".join(sorted(meshes))
                raise FileNotFoundError(
                    f"Validation test file {self.test_file_name!r} "
                    f"was not found.\nAvailable files:\n{available}"
                )
            self.test = meshes[self.test_file_name]
        else:
            alternatives = [
                mesh
                for name, mesh in meshes.items()
                if name != self.reference_file_name
            ]
            if not alternatives:
                raise ValueError(
                    "At least two .msh files are required for the Gaussian "
                    "hotspot validation test."
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

    def _shared_center(self):
        """Return one center calculated from both retained meshes."""
        all_centers = np.vstack([
            self.reference.centers,
            self.test.centers,
        ])
        return np.mean(all_centers, axis=0)

    @staticmethod
    def _constant_field(points):
        constant_vector = np.array([1.0, 0.5, -0.25], dtype=float)
        return np.repeat(
            constant_vector[np.newaxis, :],
            len(points),
            axis=0,
        )

    def _gaussian_field(self, points, center):
        """Return the same analytical Gaussian hotspot on either mesh."""
        displacement = points - center
        radius_squared = np.sum(displacement**2, axis=1)
        hotspot = GAUSSIAN_AMPLITUDE * np.exp(
            -radius_squared / (2.0 * GAUSSIAN_SIGMA_MM**2)
        )

        field = self._constant_field(points)
        field[:, 0] += hotspot
        field[:, 1] += 0.5 * hotspot
        return field

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

    def _map_with_actual_method(self, method, test_E):
        """Call the unchanged mapping implementation in methodsclass1.py."""
        return self.mapper.estimate(
            method=method,
            reference_centers=self.reference.centers,
            reference_tags=self.reference.tags,
            test_centers=self.test.centers,
            test_volumes=self.test.volumes,
            test_E=test_E,
            test_tags=self.test.tags,
            test_mesh=self.test.mesh,
            test_tet_nodes=self.test.tet_nodes,
            test_tet_element_numbers=self.test.tet_element_numbers,
        )

    def run_method(self, method, reference_E, test_E, exact_test_E):
        mapped_E, valid = self._map_with_actual_method(method, test_E)

        valid_count = int(np.count_nonzero(valid))
        total_count = len(valid)
        if valid_count == 0:
            raise ValueError(
                f"{method} produced no valid mapped points for the "
                "Gaussian hotspot test."
            )

        reference_valid = reference_E[valid]
        exact_test_valid = exact_test_E[valid]
        mapped_valid = mapped_E[valid]
        volumes_valid = self.reference.volumes[valid]

        # The analytical field is identical on both meshes, so the true
        # physical error is zero. Any reported difference is mapping error.
        expected_error = self._relative_l2_percent(
            reference_valid,
            exact_test_valid,
            volumes_valid,
        )
        reported_error = self._relative_l2_percent(
            reference_valid,
            mapped_valid,
            volumes_valid,
        )
        mapping_error = self._relative_l2_percent(
            exact_test_valid,
            mapped_valid,
            volumes_valid,
        )
        coverage = valid_count / total_count * 100.0

        row = {
            "test_name": "gaussian_same_field_different_meshes",
            "method": method,
            "mesh_pair": (
                f"{self.reference.path.name} <- {self.test.path.name}"
            ),
            "gaussian_sigma_mm": GAUSSIAN_SIGMA_MM,
            "gaussian_amplitude": GAUSSIAN_AMPLITUDE,
            "expected_error_percent": expected_error,
            "reported_error_percent": reported_error,
            "mapping_error_percent": mapping_error,
            "coverage_percent": coverage,
            "valid_reference_points": valid_count,
            "total_reference_points": total_count,
        }

        print(
            f"{method:23s} | "
            f"expected {expected_error:10.6f}% | "
            f"reported {reported_error:10.6f}% | "
            f"mapping {mapping_error:10.6f}% | "
            f"coverage {coverage:7.3f}%"
        )
        return row

    def run(self, reload_meshes=True):
        if reload_meshes or self.reference is None or self.test is None:
            self.load_meshes()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

        center = self._shared_center()

        reference_E = self._gaussian_field(
            self.reference.centers,
            center,
        )
        test_E = self._gaussian_field(
            self.test.centers,
            center,
        )

        # Independent analytical truth at the reference locations.
        exact_test_E = self._gaussian_field(
            self.reference.centers,
            center,
        )

        print("Running Gaussian hotspot validation test")
        print("-" * 105)

        for method in self.methods:
            self.results.append(
                self.run_method(
                    method=method,
                    reference_E=reference_E,
                    test_E=test_E,
                    exact_test_E=exact_test_E,
                )
            )

        print("-" * 105)
        return self.results

    def save_csv(self):
        if not self.results:
            raise RuntimeError("No validation results exist. Call run() first.")

        self.output_folder.mkdir(parents=True, exist_ok=True)
        timestamp = self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            self.output_folder
            / f"gaussian_hotspot_validation_{timestamp}.csv"
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

        methods = [row["method"] for row in self.results]
        values = [row["mapping_error_percent"] for row in self.results]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(methods, values)
        ax.set_ylabel("Mapping error (%)")
        ax.set_title("Gaussian hotspot mapping error by comparison method")
        ax.grid(True, axis="y")
        fig.tight_layout()

        self.output_folder.mkdir(parents=True, exist_ok=True)
        output_path = (
            self.output_folder
            / f"gaussian_hotspot_validation_{self.timestamp}.png"
        )
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
    validator = GaussianHotspotValidator()
    validator.run_and_export(show_plot=True)


if __name__ == "__main__":
    main()