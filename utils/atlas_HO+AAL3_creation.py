"""
Add-on to atlas_HO_creation.py : integrate AAL3 cerebellum + vermis labels
into the combined Harvard-Oxford atlas.

Strategy B :
  Within the AAL3 cerebellum+vermis mask (native codes 95-120), AAL3 always
  wins, overwriting whatever HO (Brain-Stem or boundary cortex) had assigned
  there. Justification: AAL3 is purpose-built for cerebellar/vermis
  substructure (HO's "Brain-Stem" carries no internal cerebellar detail at
  all), and at HO's 25% maxprob thresholds the conflicting voxels are
  concentrated in small, low-confidence boundary structures where preserving
  HO would fragment or nearly erase several AAL3 cerebellar subregions
  (e.g. Cerebellum_3_L, Vermis_1_2).

New label range: 50-75 (26 contiguous codes), mapped as:
    new_code = aal3_code - 95 + 50      # 95 -> 50, ..., 120 -> 75

In addition to dropping the generic "Cerebral Cortex" codes (2, 13), the two
Harvard-Oxford "Lateral Ventricle" codes (3, 14) are also excluded: they are
CSF, not a structure of interest here, so those voxels are simply zeroed out
(left as background) rather than relabeled.

Run this AFTER atlas_HO_creation.py has produced `combined` (either by
pasting this block at the end of that script, or by re-deriving `combined`
here as done below for a standalone, re-runnable version).
"""

import json
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_to_img
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
AAL3_PATH = PROJECT_ROOT / "atlas_AAL3.nii"
AAL3_JSON_PATH = PROJECT_ROOT / "atlas_labels_AAL3.json"
HO_AAL3_PATH = PROJECT_ROOT / "atlas_HO_AAL3_139.nii"
HO_AAL3_JSON_PATH = PROJECT_ROOT / "atlas_labels_HO_AAL3_139.json"

# =========================================================
# 1. Rebuild the HO `combined` atlas (same logic as atlas_HO_creation.py)
# =========================================================
atlas_cortl = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr25-1mm')
atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

img_cortl = atlas_cortl.maps
img_sub = atlas_sub.maps

data_cortl = img_cortl.get_fdata().astype(np.int32)
data_sub = img_sub.get_fdata().astype(np.int32)

VENTRICLE_CODES = (3, 14)
CORTEX_CODES = (2, 13)

combined = data_sub.copy()
mask_use_cortl = np.isin(data_sub, [0, 2, 3, 13, 14])
combined[mask_use_cortl] = 0
combined[mask_use_cortl & (data_cortl > 0)] = data_cortl[mask_use_cortl & (data_cortl > 0)] + 100


# =========================================================
# 2. Load AAL3, resample onto the HO grid (nearest-neighbor)
# =========================================================
img_aal3_native = nib.load(AAL3_PATH)

img_aal3 = resample_to_img(
    img_aal3_native, img_sub, interpolation='nearest',
    force_resample=True, copy_header=True
)
data_aal3 = img_aal3.get_fdata().astype(np.int32)

# =========================================================
# 3. Remap AAL3 cerebellum+vermis codes (95-120) -> new codes (50-75)
#    and overwrite `combined` within that mask 
# =========================================================
CEREB_CODES = list(range(95, 121))  # 26 AAL3 codes
cereb_mask = np.isin(data_aal3, CEREB_CODES)

n_before = cereb_mask.sum()
n_overwritten = ((combined != 0) & cereb_mask).sum()

new_codes = data_aal3[cereb_mask] - 95 + 50  # 95->50 ... 120->75
combined[cereb_mask] = new_codes

print(f"AAL3 cerebellum voxels integrated : {n_before}")
print(f"  of which overwrote an existing HO label : {n_overwritten} "
      f"({100 * n_overwritten / n_before:.2f}%)")

# =========================================================
# 4. Save the updated atlas
# =========================================================
out_img = nib.Nifti1Image(combined.astype(np.float32), img_sub.affine, img_sub.header)
nib.save(out_img, HO_AAL3_PATH)

# =========================================================
# 5. Extend the label JSON with the 26 new cerebellar regions
# =========================================================
with open(AAL3_JSON_PATH, "r", encoding="utf-8") as f:
    aal3_labels = json.load(f)

# JSON des labels
labels_cortl = atlas_cortl.labels  # 97 items (0=Background + 96 régions)
labels_sub = atlas_sub.labels      # 22 items (0=Background + 21 régions)

label_map = {}
for i in range(1, 22):
    if i in CORTEX_CODES or i in VENTRICLE_CODES:
        continue
    label_map[str(i)] = labels_sub[i]
for i in range(1, 97):
    label_map[str(100 + i)] = labels_cortl[i]

for aal3_code in CEREB_CODES:
    new_code = aal3_code - 45
    label_map[str(new_code)] = aal3_labels[str(aal3_code)]

with open(HO_AAL3_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=4, ensure_ascii=False)

print(f"{len(label_map)} régions au total (attendu : 17 + 96 + 26 = 139)")