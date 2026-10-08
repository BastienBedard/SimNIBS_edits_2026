# SimNIBS Mesh Comparison and Metal Implant Scripts

This repository contains Python scripts used for SimNIBS finite-element simulations, mesh-convergence analysis, validation of mesh-comparison methods, metal-implant modeling, and visualization of comparison results.

The scripts are organized into three main folders:

* `mesh comparison`
* `metal implants`
* `misc noah`

---

# 1. Mesh Comparison

This folder contains the scripts used to compare electric fields, current densities, and electric potentials between SimNIBS meshes of different resolutions.

## `meshcompareE10.py`

Main electric-field comparison script.

Compares the **electric-field vector `E`** between SimNIBS `.msh` files whose tetrahedra do not necessarily occupy identical locations.

The test mesh field is mapped onto the reference mesh using **distance-weighted k-nearest-neighbor interpolation**.

The global comparison metric is a **volume-weighted relative L2 vector-field error**.

The script has three analysis modes:

* `convergence`

  * Compares multiple meshes.
  * Produces a convergence graph of error versus number of tetrahedra.
  * Saves the graph data as a CSV file.

* `summary`

  * Calculates volume-weighted P95 and P99 local errors.
  * Produces tissue-specific and global bar graphs.

* `3d_map`

  * Calculates spatially localized electric-field error.
  * Exports a VTK point cloud.
  * Can also export a NIfTI volume for viewing error slices in software such as 3D Slicer.

The script can compare:

* only selected tissue tags or all tissues,
* only matching tissue types or unrestricted neighboring tetrahedra,
* each mesh against the mesh with the most tetrahedra,
* each mesh against a specified reference file,
* or each mesh against the next finer mesh.

This is the primary detailed electric-field comparison script.

---

## `meshcompareJ3.py`

Current-density equivalent of the electric-field convergence comparison.

Instead of comparing `E`, this script compares the **current-density vector `J`** stored in the SimNIBS mesh.

It uses the same basic distance-weighted KNN mapping approach:

1. Find the reference tetrahedron centers.
2. Find nearby tetrahedron centers in the comparison mesh.
3. Interpolate the comparison `J` vector onto the reference locations.
4. Calculate the volume-weighted relative L2 error.

The script produces:

* current-density convergence results,
* a convergence graph,
* and a CSV containing the comparison data.

The `.msh` files must contain a `J` element field.

---

## `meshcompareV3.py`

Electric-potential comparison script.

Uses the same general KNN mesh-mapping approach as the `E` and `J` comparison scripts, but compares the **electric potential `V`**.

Because SimNIBS stores electric potential at nodes, this script first assigns a value to each tetrahedron by averaging the potential at its four nodes.

Electric potential can contain an arbitrary constant offset, so the script can remove the **volume-weighted mean potential** from both meshes before calculating error.

The final metric is a volume-weighted relative L2 scalar-field error.

The script outputs:

* a convergence graph,
* terminal comparison results,
* and a CSV containing the convergence data.

---

## `methodsclass1.py`

Class-based implementation of the electric-field mesh-comparison workflow.

This script was created to place multiple fundamentally different mesh-mapping methods into one common framework.

It currently implements three methods:

### `nearest_neighbor`

For every reference tetrahedron center, uses the electric-field vector from the closest tetrahedron center in the comparison mesh.

### `knn`

Uses several nearby comparison-mesh tetrahedra and calculates a distance-weighted average of their electric-field vectors.

The main settings are:

* number of neighbors,
* distance-weighting exponent,
* tissue-tag matching.

### `containing_tetrahedron`

Finds the tetrahedron in the comparison mesh that physically contains each reference tetrahedron center.

Because SimNIBS normally stores `E` as element data rather than nodal data, the script first reconstructs nodal electric-field values using volume-weighted averaging of adjacent tetrahedra.

The electric field at the reference location is then calculated using **barycentric interpolation inside the containing tetrahedron**.

The class performs the complete convergence workflow:

* loads meshes,
* filters tissue tags,
* selects the reference,
* maps the comparison field,
* calculates global volume-weighted relative L2 error,
* saves CSV data,
* and plots the convergence result.

This file is also imported by `methodtester.py`.

---

## `methodtester.py`

Validation framework for the comparison methods implemented in `methodsclass1.py`.

Rather than relying on unknown SimNIBS field differences, this script creates **controlled analytical electric fields where the correct answer is known independently**.

It directly calls the real comparison methods from `MeshComparisonMapper` rather than recreating simplified versions of them.

The validation cases include:

* identical constant fields,
* a uniform 5% field increase,
* opposite vectors,
* a 90° vector rotation,
* the same constant field on different meshes,
* the same linear field on different meshes,
* the same Gaussian hotspot on different meshes.

For each method and validation case, the script reports:

* expected error,
* error reported by the mesh-comparison method,
* mapping/interpolation error,
* mapping coverage,
* deviation from the expected result.

Tests with exact analytical answers can produce `PASS` or `FAIL`.

Different-mesh tests with a true physical difference of zero are used primarily to measure the interpolation error introduced by each mapping method.

The script exports:

* a detailed validation CSV,
* and a graph comparing the mapping error of the methods.

`methodtestergaussian.py` is intentionally not documented here.

---

# 2. Metal Implants

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

# 3. Misc Noah

This folder contains supporting scripts that do not belong directly to one of the main analysis folders.

## `overlaygraphs1.py`

Utility for combining multiple mesh-convergence results into one figure.

The mesh-comparison scripts save convergence data as CSV files in:

`~/Desktop/meshcomparegraphdata`

This script:

1. Finds every CSV in that directory.
2. Reads:

   * `tetrahedra_kept`
   * `relative_error_percent`
3. Plots every dataset on the same graph.
4. Uses each CSV filename as the legend label.

This is useful for directly comparing convergence curves produced using different:

* mesh-comparison methods,
* tissue filters,
* reference choices,
* simulation conditions.

---

## `simnibs2_spheres.py`

Batch tDCS simulation script specifically designed for **standalone spherical `.msh` meshes**.

Unlike normal SimNIBS head-model simulations, these sphere meshes do not need to exist inside an `m2m_*` folder.

The script:

1. Finds all `.msh` sphere meshes in `m2m_spheres`.
2. Reads the coordinate bounds of each sphere.
3. Finds opposite points on the `+x` and `-x` sides.
4. Places one electrode at each side.
5. Runs a tDCS simulation on every sphere mesh.
6. Copies the final simulation `.msh` files into `simnibs_compare_spheres`.

The electrodes are positioned geometrically rather than using EEG names such as `C3` or `FC2`.

This makes the script suitable for simplified spherical convergence models.

The current stimulation uses:

* `+1 mA`,
* `-1 mA`,
* 50 × 50 mm rectangular electrodes,
* 4 mm electrode thickness.

This is the preferred simulation script for the standalone spherical meshes.

---

## `simnibs2.py`

Earlier batch tDCS simulation script for models stored as complete `m2m_*` folders.

The script:

1. Finds every `m2m_*` folder inside the selected directory.
2. Runs one tDCS simulation for each model.
3. Uses EEG-position names for electrode placement.
4. Saves the simulation outputs.
5. Copies the final scalar mesh to a common comparison directory.

The current electrode configuration uses:

* `C3`
* `FC2`

with:

* `+1 mA`,
* `-1 mA`,
* 50 × 50 mm rectangular electrodes,
* 4 mm thickness.

This version assumes a complete SimNIBS subject folder containing the information necessary to resolve EEG electrode positions.

For standalone sphere `.msh` files, use `simnibs2_spheres.py` instead.

---

# Typical Workflow

For ordinary mesh-convergence testing:

1. Generate several SimNIBS models at different mesh resolutions.
2. Run the same tDCS simulation on each model.
3. Copy the result meshes into a common comparison folder.
4. Run one of the mesh-comparison scripts.
5. Save convergence data to CSV.
6. Use `overlaygraphs1.py` when multiple convergence curves need to be viewed together.

For spherical test models:

1. Generate sphere `.msh` files at different resolutions.
2. Run `simnibs2_spheres.py`.
3. Compare the resulting meshes with `meshcompareE10.py` or `methodsclass1.py`.
4. Use `methodtester.py` to evaluate how accurately the mapping methods themselves behave.

For the metal-implant model:

1. Create the rod and refinement shell with `retag_rod1.py`.
2. Remesh the modified tissue-label volume with SimNIBS/CHARM.
3. Run the tDCS simulation using `electrode placements for patient.py`.
4. Compare the implant simulation with the baseline using the mesh-comparison tools.
5. Use `meshtemp3.py` to estimate Joule heating and approximate temperature rise around the implant.

# Main Data Types

The scripts primarily operate on:

* `.msh` — SimNIBS/Gmsh finite-element meshes and simulation results
* `.nii.gz` — NIfTI tissue-label and error-map volumes
* `.csv` — convergence and validation results
* `.vtk` — localized 3D mesh-comparison error maps

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

# Important Note

Many scripts contain absolute paths such as:

`/Users/noahholm/Desktop/...`

These paths must be changed if the repository is run on another computer or if the input/output folders are moved.
