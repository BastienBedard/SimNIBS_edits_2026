"""
Test 1: AAL3 vs Harvard-Oxford grid compatibility (affine, shape)
Test 2: Overlap check - are AAL3 cerebellum voxels all background (0) in `combined`?

This informs the choice between:
  Strategy A - fill AAL3-cerebellum labels only where combined == 0 (non-destructive)
  Strategy B - AAL3-cerebellum always overwrites within its mask

Cerebellum+Vermis codes in AAL3 = 95-120 (26 regions, see atlas_labels_AAL3.json)
New codes to assign in combined atlas = 50-75 (contiguous, 26 values)
"""

import numpy as np
import nibabel as nib
from nilearn import datasets
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
AAL3_PATH = PROJECT_ROOT / "atlas_AAL3.nii"
AAL3_JSON_PATH = PROJECT_ROOT / "atlas_labels_AAL3.json"
# ---- Load HO atlases (same as atlas_HO_creation.py) ----
atlas_cortl = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr25-1mm')
atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

img_cortl = atlas_cortl.maps
img_sub = atlas_sub.maps

data_cortl = img_cortl.get_fdata().astype(np.int32)
data_sub = img_sub.get_fdata().astype(np.int32)

# Rebuild `combined` exactly as in atlas_HO_creation.py
combined = data_sub.copy()
mask_use_cortl = np.isin(data_sub, [0, 2, 13])
combined[mask_use_cortl] = 0
combined[mask_use_cortl & (data_cortl > 0)] = data_cortl[mask_use_cortl & (data_cortl > 0)] + 100

# ---- Load AAL3 ----
img_aal3_native = nib.load(AAL3_PATH)
data_aal3_native = img_aal3_native.get_fdata().astype(np.int32)

# =========================================================
# TEST 1 : grid compatibility
# =========================================================
print("=" * 60)
print("TEST 1 : Grid compatibility (affine / shape)")
print("=" * 60)

print(f"HO sub   shape  : {data_sub.shape}")
print(f"AAL3     shape  : {data_aal3_native.shape}")
print(f"Shapes match    : {data_sub.shape == data_aal3_native.shape}")

print("\nHO sub affine:")
print(img_sub.affine)
print("\nAAL3 affine:")
print(img_aal3_native.affine)

affine_match = np.allclose(img_sub.affine, img_aal3_native.affine, atol=1e-3)
print(f"\nAffines match (atol=1e-3): {affine_match}")

grids_differ = (not affine_match) or (data_sub.shape != data_aal3_native.shape)

if grids_differ:
    print("\n>>> GRIDS DIFFER. Resampling AAL3 onto the HO grid (nearest-neighbor)...")
    from nilearn.image import resample_to_img
    img_aal3 = resample_to_img(
        img_aal3_native, img_sub, interpolation='nearest', force_resample=True, copy_header=True
    )
    print(">>> Resampling done. AAL3 is now on the HO grid.")
else:
    print("\n>>> Grids are compatible, no resampling needed.")
    img_aal3 = img_aal3_native

data_aal3 = img_aal3.get_fdata().astype(np.int32)
print(f"\nPost-resample AAL3 shape : {data_aal3.shape}")
print(f"Post-resample AAL3 affine matches HO sub : "
      f"{np.allclose(img_aal3.affine, img_sub.affine, atol=1e-3)}")

# =========================================================
# TEST 2 : overlap check for cerebellum voxels
# =========================================================
print("\n" + "=" * 60)
print("TEST 2 : AAL3 cerebellum overlap with existing `combined` labels")
print("=" * 60)

CEREB_CODES = list(range(95, 121))  # AAL3 cerebellum + vermis, 26 regions
cereb_mask = np.isin(data_aal3, CEREB_CODES)

n_cereb_voxels = cereb_mask.sum()
print(f"Total AAL3 cerebellum voxels : {n_cereb_voxels}")

if n_cereb_voxels == 0:
    print(">>> WARNING: 0 voxels found for codes 95-120 in this AAL3 volume.")
    print(">>> Check that the uploaded file is really AAL3 (not AAL, AAL2, or a")
    print(">>> different labeling convention) and that codes match the JSON.")
else:
    overlapping = combined[cereb_mask]
    n_nonzero = (overlapping != 0).sum()
    pct_nonzero = 100 * n_nonzero / n_cereb_voxels

    print(f"Voxels where combined != 0 (i.e. HO already claims something) : "
          f"{n_nonzero} ({pct_nonzero:.2f}%)")
    print(f"Voxels where combined == 0 (clean background)                 : "
          f"{n_cereb_voxels - n_nonzero} ({100 - pct_nonzero:.2f}%)")

    if n_nonzero > 0:
        print("\nBreakdown of what `combined` (HO) contains in the conflicting voxels:")
        conflict_vals, counts = np.unique(overlapping[overlapping != 0], return_counts=True)
        for val, cnt in sorted(zip(conflict_vals, counts), key=lambda x: -x[1]):
            origin = "HO subcortical" if val < 100 else "HO cortical (label - 100)"
            pct_of_region = 100 * cnt / (combined == val).sum()
            print(f"  combined == {int(val):4d} ({origin:25s}) : {cnt:5d} voxels "
                  f"({pct_of_region:.1f}% of that region's total volume)")

        # ---- Same conflict, but from the AAL3 side: which cerebellar
        #      regions are the ones losing/winning voxels against HO? ----
        import json
        with open(AAL3_JSON_PATH, "r", encoding="utf-8") as f:
            aal3_labels = json.load(f)

        conflict_mask_full = cereb_mask & (combined != 0)
        aal3_conflict_vals, aal3_counts = np.unique(
            data_aal3[conflict_mask_full], return_counts=True
        )
        print("\nBreakdown of which AAL3 cerebellar regions are involved in the conflict:")
        for val, cnt in sorted(zip(aal3_conflict_vals, aal3_counts), key=lambda x: -x[1]):
            name = aal3_labels.get(str(int(val)), "UNKNOWN")
            pct_of_region = 100 * cnt / (data_aal3 == val).sum()
            print(f"  AAL3 == {int(val):3d} ({name:22s}) : {cnt:5d} voxels "
                  f"({pct_of_region:.1f}% of that region's total volume)")

    print(f"\n>>> Recommendation:")
    if pct_nonzero < 1.0:
        print(f">>>   Overlap is negligible ({pct_nonzero:.2f}%). Strategy A and B")
        print(f">>>   will give nearly identical results. Strategy A (fill background")
        print(f">>>   only) is still the safer/more conservative default.")
    else:
        print(f">>>   Overlap is non-trivial ({pct_nonzero:.2f}%). Decide explicitly:")
        print(f">>>   - Strategy A keeps existing HO labels in those voxels (AAL3 skipped there)")
        print(f">>>   - Strategy B lets AAL3 cerebellum override those HO labels")
        print(f">>>   Inspect the breakdown above to judge which is more anatomically sound.")

# =========================================================
# Sanity check: confirm 26 distinct cerebellar codes present & label count
# =========================================================
print("\n" + "=" * 60)
print("TEST 3 (bonus) : label count sanity check")
print("=" * 60)
found_codes = sorted(np.unique(data_aal3[cereb_mask]).tolist())
print(f"Distinct AAL3 codes found in cerebellum mask ({len(found_codes)} of 26 expected):")
print(found_codes)
missing = set(CEREB_CODES) - set(found_codes)
if missing:
    print(f">>> WARNING: codes with 0 voxels in this volume: {sorted(missing)}")