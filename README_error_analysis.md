# SimNIBS Mesh Comparison and Metal Implant Scripts

This repository contains Python code used for SimNIBS finite-element simulations, mesh-convergence analysis, validation of the mesh-comparison methods, and metal-implant modeling.

The code is organized into three groups:

* **Simulation and mesh comparison** (refactored modules)

  * `tDCS_simulation.py`
  * `error_analysis.py`
  * `method_comparison.py`

* **Metal implants** (`metal implants` folder)

  * `electrode placements for patient.py`
  * `retag_rod1.py`
  * `meshtemp3.py`

The three refactored modules replace the earlier standalone scripts (`simnibs2.py`, `simnibs2_spheres.py`, `meshcompareE10.py`, `meshcompareJ3.py`, `meshcompareV3.py`, `methodsclass1.py`, `methodtester.py`, `methodtestergaussian.py` and `overlaygraphs1.py`). Their functionality is described below.

---

# 1. Running the Simulations

## `tDCS_simulation.py`

Runs the same tDCS simulation on several meshes so that the resulting fields can be compared across mesh sizes.

This single script replaces the two earlier batch scripts, `simnibs2.py` (head models) and `simnibs2_spheres.py` (spherical meshes). The model type is selected with the `MODEL_TYPE` switch:

* `"head"`: SimNIBS head models stored as `m2m_*` folders. Electrodes are placed with EEG position names.
* `"sphere"`: standalone spherical `.msh` files (no `m2m` folder needed). Electrodes are placed geometrically.

### Models

The models to simulate are listed explicitly (the script no longer scans a folder for every `m2m_*`):

* `M2M_FOLDER_NAMES`: sub-folders of the head-mesh folder (`data/ernie`), used when `MODEL_TYPE = "head"`.
* `SPHERE_MESH_NAMES`: `.msh` files inside `data/m2m_spheres`, used when `MODEL_TYPE = "sphere"`.

The script fails early if a listed model does not exist.

### Electrodes

* **Head models:** electrodes at `C3` and `FC2`.
* **Sphere models:** one electrode at the `+x` extreme and one at the `-x` extreme of the sphere, at the middle of the y/z bounding box. The pad rotation is fixed by orienting each pad toward a point 20 mm away along `+y`.

Common settings:

* currents of `+1 mA` and `-1 mA`,
* 50 × 50 mm rectangular electrodes,
* 4 mm electrode thickness,
* saved fields: `v`, `e`, `E`, `j`, `J` (`FIELDS = "veEjJ"`).

The conductivity table used by each simulation is printed before it runs.

### Output

Each simulation is written directly under the results folder, in its own sub-folder named `sim_<model name>`:

* `results/results_multisized_mesh` for head models,
* `results/results_multisized_spheres` for sphere models.

No result file is copied elsewhere: `error_analysis.py` reads the `*_scalar.msh` file from each `sim_*` sub-folder directly.

---

# 2. Mesh Comparison

## `error_analysis.py`

Mesh-convergence comparison of SimNIBS results. This module replaces `meshcompareE10.py`, `meshcompareJ3.py`, `meshcompareV3.py` and `methodsclass1.py` with one importable module covering all fields and all mapping methods.

It is meant to be **imported** (for example from a notebook) and is never run directly. It contains no file locations: every path is given by the caller.

### Compared fields

| Field | Source in the mesh | Kind                 | Notes                                                                                                |
| ----- | ------------------ | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `E`   | `E`                | element vector       | Electric field                                                                                       |
| `J`   | `J`                | element vector       | Current density                                                                                      |
| `V`   | `v`                | node scalar          | Electric potential, averaged over the four nodes of each tetrahedron; the volume-weighted mean is removed from each mesh because `V` is defined up to an arbitrary constant |

The fields to extract are chosen when the comparison is created (`fields=("E", "J", "V")`). The `.msh` files must contain them (see `FIELDS` in `tDCS_simulation.py`).

### Mapping methods

The test mesh field is mapped onto the tetrahedron centers of the reference mesh, because the tetrahedra of different meshes do not occupy identical locations.

* `nearest_neighbor`: uses the value of the closest tetrahedron center in the comparison mesh.
* `knn`: distance-weighted average of the `k` nearest tetrahedron centers (settings: `k_neighbors`, `distance_power`).
* `containing_tetrahedron`: finds the comparison-mesh tetrahedron that physically contains each reference center and uses **barycentric interpolation** of nodal values. Because SimNIBS stores `E` and `J` per element, nodal values are first reconstructed by volume-weighted averaging of adjacent tetrahedra, separately for each tissue so that adjacent tissues are not blended. This is the only method that needs the full SimNIBS mesh object; it is read from disk for one test mesh at a time and freed afterwards.

With `match_tissue_tags=True`, only tetrahedra with the same tissue tag are used for mapping.

### Error metric

The global metric is the **volume-weighted relative L2 error** (in percent), computed by `relative_l2_error`:

`100 × sqrt(Σ V·|ref − test|²) / sqrt(Σ V·|ref|²)`

It works for vector fields (`E`, `J`) and scalar fields (`V`). Only reference tetrahedra that could be mapped enter the error. The results report **coverage** (share of reference tetrahedra, by count and by volume, that were mapped).

### Reference modes

* `most_tetrahedra`: every mesh is compared with the mesh that has the most kept tetrahedra.
* `specific_file`: every mesh is compared with the file given in `reference_file_name`.
* `next_finer`: every mesh is compared with the next finer mesh (the finest mesh is skipped).

### Tissue selection

With `compare_only_list=True`, only the tissue tags in `used_tags` are kept (default: tags 1–10, 100 and 500). With `compare_only_list=False`, all tissues are used.

### Workflow

The `MeshConvergenceComparison` class:

1. Finds the single `*_scalar.msh` result in each listed simulation sub-folder of the mesh folder.
2. Loads each mesh into a `TetraMeshData` object (centers, volumes, tags, node indices, field values). The raw mesh is not kept in memory.
3. Selects the reference mesh.
4. Maps the test field onto the reference and computes the error.

Public methods (all return pandas DataFrames and keep no hidden results):

* `run(method, field)`: one method, one field.
* `run_all_methods(...)`: several mapping methods (all three by default).
* `run_all_fields(method, ...)`: several fields with one method.
* `run_all_reference_modes(method, ...)`: several reference modes with one method.
* `plot(results, group_by=...)`: convergence graph of relative error versus number of tetrahedra; one line per `method`, `field` or `reference_mode`. Call `plt.show()` afterwards.
* `save_csv(results)`: saves the results to the CSV output folder with an automatically generated file name.

### Result columns

`method`, `field`, `tetrahedra_kept`, `relative_error_percent`, `coverage_percent`, `coverage_volume_percent`, `test_file`, `reference_file`, `reference_mode`, `compare_only_list`, `match_tissue_tags`, `k_neighbors`, `distance_power`.

### Comparing several curves

Results DataFrames (or CSV files read with `pandas.read_csv`) can be joined with `pandas.concat` and passed to `plot_convergence(results, group_by=...)`. This replaces `overlaygraphs1.py`.

### Memory use

`chunk_size` (default 500,000) sets how many tetrahedra (when loading) or query points (`nearest_neighbor`, `knn`) are processed at a time. A smaller value lowers peak memory use; the results do not depend on it.

### Typical use

```python
comparison = MeshConvergenceComparison(
    mesh_folder=MESHCOMPARE_FOLDER,
    simulation_folders=["sim_m2m_ernie1", "sim_m2m_ernie2"],
    csv_output_folder=CSV_OUTPUT_FOLDER,
    fields=("E", "J", "V"),
)
results = comparison.run("nearest_neighbor", field="E")
comparison.plot(results)
comparison.save_csv(results)
```

### Not part of the refactored module

The `summary` mode (volume-weighted P95/P99 local errors, tissue-specific bar graphs) and the `3d_map` mode (VTK/NIfTI export of localized error) of the earlier `meshcompareE10.py` have no equivalent in `error_analysis.py`.

---

## `method_comparison.py`

Validation of the mapping methods of `error_analysis.py` with controlled fields. This module replaces `methodtester.py` and `methodtestergaussian.py`.

An error reported by a convergence run mixes two things: a real difference between the two simulated fields, and an error introduced by the mapping itself. This module isolates the second one. It builds **analytical fields whose correct answer is known** and sends them through the real `MeshComparisonMapper` (the methods are not re-implemented).

Only the geometry and tissue tags of the meshes are used. The field stored in a mesh file is never read, so the meshes do not have to contain a simulation result: the head mesh of an `m2m` folder works as well as a `*_scalar.msh` file.

Like `error_analysis.py`, it is meant to be imported and contains no file locations.

### Validation cases

Cases with an exact analytical answer (same mesh; `PASS`/`FAIL` status):

* `identical_constant_field`: expected error 0 %.
* `uniform_scale_5_percent`: test field is 1.05 times the reference; expected error 5 %.
* `opposite_vectors`: expected error 200 %.
* `rotation_90_degrees`: every vector rotated 90° about the z axis; expected value calculated directly.

Cases on different meshes with a true difference of zero (`MEASURE` status; the reported error is the mapping error of the method):

* `constant_same_field_different_meshes`
* `linear_same_field_different_meshes`
* `gaussian_same_field_different_meshes`: smooth Gaussian hotspot (`gaussian_sigma_mm`, `gaussian_amplitude` are adjustable).

### Reported quantities

For each case and method:

* `expected_error_percent_all_points` and `expected_error_percent_valid_points`: true difference evaluated directly at the reference centers.
* `reported_error_percent`: difference after the method maps the test field.
* `mapping_error_percent`: mapped test field versus the exact test field; the error of the method alone.
* `absolute_deviation_percentage_points` and `relative_deviation_percent`: deviation from the expected result.
* `coverage_percent`, `valid_reference_points`, `total_reference_points`.
* `status`: `PASS`, `FAIL` (tolerance `exact_tolerance`, default `1e-8` percentage points) or `MEASURE`.

### Workflow

The `MeshComparisonMethodValidator` class takes a reference mesh and a test mesh (they may be the same file), loads their geometry, injects the synthetic field into a copy of the test mesh data, and calls the unchanged mapper.

* `run(cases=None)`: runs all cases (or a list of case names) with every method; returns a DataFrame.
* `plot(results)`: bar chart of the mapping error of each method on the different-mesh cases.
* `save_csv(results)`: saves the detailed validation CSV.

### Typical use

```python
validator = MeshComparisonMethodValidator(
    reference_path=REFERENCE_MESH,
    test_path=TEST_MESH,
    output_folder=VALIDATION_OUTPUT_FOLDER,
)
results = validator.run()
fig, ax = validator.plot(results)
validator.save_csv(results)
```

---

# 3. Metal Implants

This folder contains scripts used to create and analyze a metal implant inside a SimNIBS head model.

## `electrode placements for patient.py`

Batch SimNIBS tDCS simulation script used for head models, including models containing the metal implant.

The script:

1. Finds all `m2m_*` folders inside the selected directory.
2. Creates a SimNIBS tDCS simulation for each model.
3. Defines the electrode locations, dimensions, currents, and output fields.
4. Defines custom conductivities for the implant-related tissue tags.
5. Runs each simulation.
6. Copies the final scalar `.msh` result into a common comparison folder.

The current configuration uses:

* two rectangular electrodes,
* electrode locations `P10` and `AF3`,
* currents of `+2 mA` and `-2 mA`,
* a metal rod represented by tissue tag `51`,
* a surrounding refinement shell represented by tag `52`.

Tag `51` is assigned a very high electrical conductivity to represent the metal rod.

Tag `52` is not metal. It represents the artificial surrounding region used to maintain local mesh refinement and is assigned a tissue-like conductivity.

The simulation saves:

* electric potential `v`,
* electric-field magnitude `e`,
* electric-field vector `E`,
* current-density magnitude `j`,
* current-density vector `J`.

---

## `retag_rod1.py`

Creates the metal rod and surrounding refinement shell inside a SimNIBS tissue-label NIfTI volume.

The implant is defined as a cylinder between two specified voxel coordinates.

Two regions are created:

* **Tag 51:** actual metal rod
* **Tag 52:** larger surrounding refinement shell

The script:

1. Loads the tissue-labeling `.nii.gz` file.
2. Converts voxel coordinates to physical coordinates.
3. Builds cylindrical masks using specified endpoints and radii.
4. Restricts replacement to selected jaw-region tissues.
5. Writes tag `52` into the outer shell.
6. Writes tag `51` into the inner metal cylinder.
7. Saves the modified tissue-label volume.
8. Saves separate binary rod and shell masks.

The shell exists because a small isolated implant region can otherwise be removed or poorly represented during the CHARM meshing process.

The edited label volume can then be remeshed to produce a head mesh containing the implant.

---

## `meshtemp3.py`

Post-processing script used to estimate **Joule heating and temperature rise**, particularly around the metal implant.

The script requires a SimNIBS result mesh containing both:

* electric field `E`,
* current density `J`.

The volumetric Joule heating is calculated as:

`q = J · E`

where `q` is in W/m³.

An approximate adiabatic temperature rise is then calculated using:

`ΔT = q × t / (ρc)`

where:

* `t` is the stimulation duration,
* `ρc` is volumetric heat capacity.

The script uses a separate volumetric heat capacity for the titanium implant region.

It calculates:

* Joule-heating density,
* estimated adiabatic temperature rise,
* power per tetrahedron,
* total power,
* global statistics,
* statistics for individual tissue tags,
* statistics in tissue close to the metal rod.

A configurable **near-metal shell test** finds tissue tetrahedra whose centers are within a specified distance of the metal rod.

The results are saved as:

* a CSV containing regional statistics,
* a new `.msh` file containing Joule-heating and temperature-related element fields.

The temperature calculation is an adiabatic estimate and does not model heat conduction, blood perfusion, or other thermal transport effects.

---

# Typical Workflow

For ordinary mesh-convergence testing:

1. Generate several SimNIBS models at different mesh resolutions.
2. List the models in `tDCS_simulation.py` (`M2M_FOLDER_NAMES`) and run it with `MODEL_TYPE = "head"`. Each result is saved in its own `sim_<model name>` sub-folder.
3. In a notebook, import `error_analysis.py` and create a `MeshConvergenceComparison` with the list of `sim_*` sub-folders to compare.
4. Run the comparison (`run`, `run_all_methods`, `run_all_fields`, ...), plot the DataFrame, and save it with `save_csv`.
5. Combine several results DataFrames or CSV files with `pandas.concat` to overlay convergence curves.

For spherical test models:

1. Generate sphere `.msh` files at different resolutions and place them in `data/m2m_spheres`.
2. List them in `SPHERE_MESH_NAMES` and run `tDCS_simulation.py` with `MODEL_TYPE = "sphere"`.
3. Compare the results with `error_analysis.py`.

To evaluate the mapping methods themselves:

1. Choose a reference mesh and a test mesh (head meshes without a simulation field are sufficient).
2. Run `MeshComparisonMethodValidator` from `method_comparison.py`.
3. Check the `PASS`/`FAIL` status of the exact cases, and read `mapping_error_percent` for the different-mesh cases to see how much error each method adds by itself.

For the metal-implant model:

1. Create the rod and refinement shell with `retag_rod1.py`.
2. Remesh the modified tissue-label volume with SimNIBS/CHARM.
3. Run the tDCS simulation using `electrode placements for patient.py`.
4. Compare the implant simulation with the baseline using `error_analysis.py`.
5. Use `meshtemp3.py` to estimate Joule heating and approximate temperature rise around the implant.

# Main Data Types

The scripts primarily operate on:

* `.msh`: SimNIBS/Gmsh finite-element meshes and simulation results
* `.nii.gz`: NIfTI tissue-label volumes
* `.csv`: convergence and validation results

# Main Tissue Tags

Common tissue tags used throughout the scripts are:

| Tag | Tissue                    |
| --: | ------------------------- |
|   1 | White matter              |
|   2 | Gray matter               |
|   3 | CSF                       |
|   4 | Bone                      |
|   5 | Scalp                     |
|   6 | Eyeballs                  |
|   7 | Compact bone              |
|   8 | Spongy bone               |
|   9 | Blood                     |
|  10 | Muscle                    |
|  51 | Metal rod                 |
|  52 | Near-rod refinement shell |
| 100 | Electrode                 |
| 500 | Saline / gel              |

# Important Notes

* `tDCS_simulation.py` defines its folders relative to the project root (`PROJECT_ROOT`, the parent of the folder containing the script): `data/ernie`, `data/m2m_spheres` and `results/...`. `error_analysis.py` and `method_comparison.py` take every path from the caller.
* The metal-implant scripts may still contain absolute paths such as `/Users/noahholm/Desktop/...`. These must be changed if they are run on another computer or if the input/output folders are moved.
* The mapping methods of `error_analysis.py` expect exactly one `*_scalar.msh` file in each listed simulation folder.