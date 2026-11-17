"""
Validation of the mesh-comparison mapping methods with controlled fields.

The methods of error_analysis.py (nearest_neighbor, knn, containing_tetrahedron)
map a field from a test mesh onto the tetrahedron centers of a reference mesh.
An error reported by a convergence run therefore mixes two things: a real
difference between the two simulated fields, and an error introduced by the
mapping itself. This module isolates the second one: it builds analytical
fields whose correct answer is known, and sends them through the real
MeshComparisonMapper.

Only the geometry and tissue tags of the meshes are used. The field stored in
a mesh file is never read, so the meshes do not have to contain a simulation
result: the head mesh of an m2m folder works as well as a *_scalar.msh file.

This module is meant to be imported (for example from a notebook) and is never
run directly. It contains no file locations: every path is given by the caller.

Typical use
-----------
    validator = MeshComparisonMethodValidator(
        reference_path=REFERENCE_MESH,
        test_path=TEST_MESH,
        output_folder=VALIDATION_OUTPUT_FOLDER,
    )
    results = validator.run()                       # DataFrame
    fig, ax = validator.plot(results)
    validator.save_csv(results)

    # Only the Gaussian hotspot case
    results = validator.run(cases=["gaussian_same_field_different_meshes"])
"""
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simnibs import mesh_io

from error_analysis import (
    DEFAULT_USED_TAGS,
    METHODS_NEEDING_MESH,
    VALID_COMPARISON_METHODS,
    MeshComparisonMapper,
    TetraMeshData,
    relative_l2_error,
)


# Name under which the synthetic field is stored in TetraMeshData.values
FIELD_NAME = "E"


# ------------------------------------------------------------
# Validation cases
# ------------------------------------------------------------

@dataclass(frozen=True)
class ValidationCase:
    """
    One controlled comparison with independently known field functions.

    reference_field and test_field are functions (points, tags) -> array of
    shape (n, 3). exact_expected_result marks the cases whose reported error
    must equal the true error, which get a PASS/FAIL status. The other cases
    have a true error of zero, so what they report is the mapping error.
    """

    name: str
    label: str
    description: str
    reference_field: Callable
    test_field: Callable
    different_meshes: bool
    exact_expected_result: bool = False


def build_cases(
    center,
    length_scale,
    gaussian_sigma_mm=12.0,
    gaussian_amplitude=20.0,
):
    """Return the list of validation cases for a given domain center and size."""
    constant_vector = np.array([1.0, 0.5, -0.25], dtype=float)

    def constant_field(points, tags):
        return np.tile(constant_vector, (len(points), 1))

    def scaled_constant_field(points, tags):
        return 1.05 * constant_field(points, tags)

    def opposite_constant_field(points, tags):
        return -constant_field(points, tags)

    def rotated_constant_field(points, tags):
        # Exact 90-degree rotation about the z axis.
        base = constant_field(points, tags)
        rotated = base.copy()
        rotated[:, 0] = -base[:, 1]
        rotated[:, 1] = base[:, 0]
        return rotated

    def linear_field(points, tags):
        q = (points - center) / length_scale
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        return np.column_stack([
            1.0 + 0.8 * x + 0.2 * y,
            0.5 - 0.3 * x + 0.7 * z,
            -0.25 + 0.4 * y - 0.6 * z,
        ])

    def gaussian_field(points, tags):
        radius_squared = np.sum((points - center) ** 2, axis=1)
        hotspot = gaussian_amplitude * np.exp(
            -radius_squared / (2.0 * gaussian_sigma_mm**2)
        )
        field = constant_field(points, tags)
        field[:, 0] += hotspot
        field[:, 1] += 0.5 * hotspot
        return field

    return [
        ValidationCase(
            name="identical_constant_field",
            label="Identical",
            description=(
                "Same mesh, same constant vector field; expected error is 0%."
            ),
            reference_field=constant_field,
            test_field=constant_field,
            different_meshes=False,
            exact_expected_result=True,
        ),
        ValidationCase(
            name="uniform_scale_5_percent",
            label="Scale x1.05",
            description=(
                "Same mesh; test field is 1.05 times the reference; expected "
                "relative vector L2 error is 5%."
            ),
            reference_field=constant_field,
            test_field=scaled_constant_field,
            different_meshes=False,
            exact_expected_result=True,
        ),
        ValidationCase(
            name="opposite_vectors",
            label="Opposite",
            description=(
                "Same mesh, opposite vectors; expected relative vector L2 "
                "error is 200%."
            ),
            reference_field=constant_field,
            test_field=opposite_constant_field,
            different_meshes=False,
            exact_expected_result=True,
        ),
        ValidationCase(
            name="rotation_90_degrees",
            label="Rotation 90 deg",
            description=(
                "Same mesh; every vector is rotated 90 degrees about the z "
                "axis. The expected value is calculated directly."
            ),
            reference_field=constant_field,
            test_field=rotated_constant_field,
            different_meshes=False,
            exact_expected_result=True,
        ),
        ValidationCase(
            name="constant_same_field_different_meshes",
            label="Constant",
            description=(
                "Different meshes, same constant field. The true difference "
                "is 0%; the reported error is mapping error."
            ),
            reference_field=constant_field,
            test_field=constant_field,
            different_meshes=True,
        ),
        ValidationCase(
            name="linear_same_field_different_meshes",
            label="Linear",
            description=(
                "Different meshes, same spatially linear field. The true "
                "difference is 0%."
            ),
            reference_field=linear_field,
            test_field=linear_field,
            different_meshes=True,
        ),
        ValidationCase(
            name="gaussian_same_field_different_meshes",
            label="Gaussian hotspot",
            description=(
                "Different meshes, same smooth Gaussian hotspot. The true "
                "difference is 0%."
            ),
            reference_field=gaussian_field,
            test_field=gaussian_field,
            different_meshes=True,
        ),
    ]


# ------------------------------------------------------------
# Validator
# ------------------------------------------------------------

class MeshComparisonMethodValidator:
    """
    Validate the real MeshComparisonMapper methods with controlled fields.

    reference_path   Mesh whose tetrahedron centers and volumes are used to
                     measure the errors.
    test_path        Mesh the synthetic field is mapped from, for the cases
                     that use different meshes. May equal reference_path.
    output_folder    Where save_csv() writes (only needed for saving).
    methods          Mapping methods to validate (all three by default).

    The mesh files only provide geometry and tissue tags; no field is read
    from them (they are loaded with fields=()). The synthetic field is injected
    into a copy of the test TetraMeshData, then the unchanged mapper is called.

    For every case and method, the results contain:

    expected_error_percent_*  true difference between the reference field and
                              the test field, evaluated directly at the
                              reference centers (all points / valid points).
    reported_error_percent    difference after the method maps the test field
                              onto the reference centers.
    mapping_error_percent     mapped test field vs the exact test field at the
                              reference centers: the error of the method alone.
    coverage_percent          share of reference tetrahedra that were mapped.
                              Only these enter the three errors above.

    The raw SimNIBS meshes are kept in memory only when a method needs them
    (METHODS_NEEDING_MESH), and only for the two meshes involved.
    """

    def __init__(
        self,
        reference_path,
        test_path,
        output_folder=None,
        methods=VALID_COMPARISON_METHODS,
        compare_only_list=True,
        used_tags=DEFAULT_USED_TAGS,
        k_neighbors=8,
        distance_power=2,
        match_tissue_tags=True,
        gaussian_sigma_mm=12.0,
        gaussian_amplitude=20.0,
        exact_tolerance=1e-8,
        verbose=True,
    ):
        self.reference_path = Path(reference_path)
        self.test_path = Path(test_path)
        self.output_folder = (
            Path(output_folder) if output_folder is not None else None
        )
        self.methods = list(methods)
        self.compare_only_list = bool(compare_only_list)
        self.used_tags = list(used_tags)
        self.gaussian_sigma_mm = float(gaussian_sigma_mm)
        self.gaussian_amplitude = float(gaussian_amplitude)
        # Pass tolerance, in percentage points, for the exact cases.
        self.exact_tolerance = float(exact_tolerance)
        self.verbose = bool(verbose)

        if not self.methods:
            raise ValueError("methods is empty.")

        # The mapper stays silent: coverage_percent already reports skipped points.
        self.mapper = MeshComparisonMapper(
            k=k_neighbors,
            distance_power=distance_power,
            match_tissue_tags=match_tissue_tags,
            verbose=False,
        )

        self._keep_raw_meshes = any(
            method in METHODS_NEEDING_MESH for method in self.methods
        )
        self._raw_meshes = {}
        self.reference = None
        self.test = None

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    def _load_one(self, path):
        self._log(f"Reading validation mesh: {path.name}")
        mesh = mesh_io.read_msh(str(path))

        # fields=(): no field is needed, the synthetic one is injected later.
        data = TetraMeshData.from_mesh(
            mesh,
            path,
            compare_only_list=self.compare_only_list,
            used_tags=self.used_tags,
            fields=(),
        )

        if self._keep_raw_meshes:
            self._raw_meshes[data.path] = mesh

        self._log(f"  Tetrahedra kept: {data.tetrahedron_count}")
        return data

    def load_meshes(self):
        """Read the reference and test meshes (once if they are the same file)."""
        self._raw_meshes = {}
        self.reference = self._load_one(self.reference_path)

        if self.test_path == self.reference_path:
            self.test = self.reference
        else:
            self.test = self._load_one(self.test_path)

        return self.reference, self.test

    def _domain_parameters(self):
        """Shared center and length scale of the analytical fields."""
        all_centers = np.vstack([self.reference.centers, self.test.centers])
        center = np.mean(all_centers, axis=0)
        length_scale = float(np.max(np.ptp(all_centers, axis=0)))

        if length_scale == 0.0:
            length_scale = 1.0

        return center, length_scale

    # --------------------------------------------------------
    # Running the cases
    # --------------------------------------------------------

    def run_case(self, case, method):
        """Run one validation case with one method; returns one result row."""
        reference = self.reference
        test = self.test if case.different_meshes else self.reference

        reference_field = case.reference_field(reference.centers, reference.tags)
        test_field = case.test_field(test.centers, test.tags)

        # Independent truth: the intended test field evaluated directly at the
        # reference locations, without any mapping method.
        exact_test_field = case.test_field(reference.centers, reference.tags)

        expected_all = relative_l2_error(
            reference_field, exact_test_field, reference.volumes
        )

        test_with_field = replace(test, values={FIELD_NAME: test_field})

        mapped_field, valid = self.mapper.estimate(
            method,
            reference,
            test_with_field,
            test_mesh=self._raw_meshes.get(test.path),
            field=FIELD_NAME,
        )

        valid_count = int(np.count_nonzero(valid))
        total_count = len(valid)

        if valid_count == 0:
            raise ValueError(
                f"{method} produced no valid mapped points for {case.name}."
            )

        volumes_valid = reference.volumes[valid]

        # Everything below uses exactly the points the method could map, so
        # coverage cannot distort the comparison with the expected value.
        expected_valid = relative_l2_error(
            reference_field[valid], exact_test_field[valid], volumes_valid
        )
        reported = relative_l2_error(
            reference_field[valid], mapped_field[valid], volumes_valid
        )
        mapping = relative_l2_error(
            exact_test_field[valid], mapped_field[valid], volumes_valid
        )

        absolute_deviation = abs(reported - expected_valid)
        relative_deviation = (
            np.nan
            if expected_valid == 0.0
            else absolute_deviation / expected_valid * 100.0
        )
        coverage = 100.0 * valid_count / total_count

        if case.exact_expected_result:
            status = (
                "PASS" if absolute_deviation <= self.exact_tolerance else "FAIL"
            )
        else:
            status = "MEASURE"

        self._log(
            f"{case.name:40s} | {method:23s} | "
            f"expected {expected_valid:10.6f}% | "
            f"reported {reported:10.6f}% | "
            f"mapping {mapping:10.6f}% | "
            f"coverage {coverage:7.3f}% | {status}"
        )

        return {
            "test_name": case.name,
            "label": case.label,
            "method": method,
            "different_meshes": case.different_meshes,
            "mesh_pair": f"{reference.path.name} <- {test.path.name}",
            "expected_error_percent_all_points": float(expected_all),
            "expected_error_percent_valid_points": float(expected_valid),
            "reported_error_percent": float(reported),
            "absolute_deviation_percentage_points": float(absolute_deviation),
            "relative_deviation_percent": float(relative_deviation),
            "mapping_error_percent": float(mapping),
            "coverage_percent": coverage,
            "valid_reference_points": valid_count,
            "total_reference_points": total_count,
            "status": status,
            "description": case.description,
        }

    def run(self, cases=None, reload_meshes=False):
        """
        Run the validation cases with every method; returns a DataFrame.

        cases is a list of case names (all cases by default), for example
        ["gaussian_same_field_different_meshes"]. Meshes are read on the first
        call and reused afterwards; pass reload_meshes=True to read them again.
        """
        if reload_meshes or self.reference is None:
            self.load_meshes()

        center, length_scale = self._domain_parameters()
        all_cases = build_cases(
            center,
            length_scale,
            gaussian_sigma_mm=self.gaussian_sigma_mm,
            gaussian_amplitude=self.gaussian_amplitude,
        )

        if cases is None:
            selected = all_cases
        else:
            by_name = {case.name: case for case in all_cases}
            unknown = [name for name in cases if name not in by_name]

            if unknown:
                raise ValueError(
                    f"Unknown case(s) {unknown}. Use any of: {list(by_name)}"
                )

            selected = [by_name[name] for name in cases]

        self._log()
        self._log("Reference:", self.reference.path.name)
        self._log("Test mesh: ", self.test.path.name)
        self._log("-" * 140)

        rows = [
            self.run_case(case, method)
            for case in selected
            for method in self.methods
        ]

        self._log("-" * 140)

        results = pd.DataFrame(rows)

        failed = int((results["status"] == "FAIL").sum())
        if failed:
            self._log(f"WARNING: {failed} exact test(s) FAILED.")

        return results

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    def plot(self, results, ax=None):
        """
        Bar chart of the mapping error of each method on the different-mesh
        cases; returns (fig, ax). Call plt.show() or fig.savefig() after.
        """
        mapping_rows = results[results["different_meshes"]]

        if mapping_rows.empty:
            raise ValueError(
                "The results contain no different-mesh case to plot."
            )

        case_names = list(dict.fromkeys(mapping_rows["test_name"]))
        methods = list(dict.fromkeys(mapping_rows["method"]))
        labels = dict(zip(mapping_rows["test_name"], mapping_rows["label"]))

        table = mapping_rows.pivot_table(
            index="test_name",
            columns="method",
            values="mapping_error_percent",
            aggfunc="first",
        ).reindex(index=case_names, columns=methods)

        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 6))
        else:
            fig = ax.figure

        x = np.arange(len(case_names), dtype=float)
        width = 0.8 / len(methods)

        for position, method in enumerate(methods):
            offset = (position - (len(methods) - 1) / 2) * width
            ax.bar(
                x + offset,
                table[method].to_numpy(),
                width=width,
                label=method,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([labels[name] for name in case_names])
        ax.set_ylabel("Mapping error (%)")
        ax.set_title("Mapping error of the comparison methods on analytical fields")
        ax.grid(True, axis="y")
        ax.legend(title="method")
        fig.tight_layout()

        return fig, ax

    def save_csv(self, results, filename=None):
        """Save a results DataFrame to output_folder and return the path."""
        if self.output_folder is None:
            raise ValueError(
                "output_folder was not given when the validator was created."
            )

        self.output_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"method_validation_{timestamp}.csv"

        output_path = self.output_folder / filename
        results.to_csv(output_path, index=False)

        self._log(f"Results saved to: {output_path}")
        return output_path