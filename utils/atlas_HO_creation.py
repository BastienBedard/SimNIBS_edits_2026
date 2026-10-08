import numpy as np
import nibabel as nib
from nilearn import datasets
from scipy import ndimage

atlas_cortl = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr25-1mm')
atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

img_cortl = atlas_cortl.maps  # déjà un Nifti1Image, pas besoin de nib.load
img_sub = atlas_sub.maps

data_cortl = img_cortl.get_fdata().astype(np.int32)
data_sub = img_sub.get_fdata().astype(np.int32)

# Point de départ : le volume subcortical
combined = data_sub.copy()



# Là où le sub dit "cortex générique" (2 ou 13) OU rien (0),
# on prend le label cortical latéralisé, offsetté +100
mask_use_cortl = np.isin(data_sub, [0, 2, 13])
combined[mask_use_cortl] = 0
combined[mask_use_cortl & (data_cortl > 0)] = data_cortl[mask_use_cortl & (data_cortl > 0)] + 100




# Parmi les voxels où sub == 0 (background) et qu'on recode en cortl,
# combien tombent dans une région anatomiquement plausible comme cortex
# vs. profondément entourés de matière blanche ?

wm_mask = np.isin(data_sub, [1, 12])
bg_recoded_mask = (data_sub == 0) & (data_cortl > 0)

# Distance de chaque voxel "recodé" au voxel de matière blanche le plus proche
dist_to_wm = ndimage.distance_transform_edt(~wm_mask)
depths = dist_to_wm[bg_recoded_mask]

print(f"Voxels sub=0 recodés en cortl : {bg_recoded_mask.sum()}")
print(f"Distance moyenne à la WM la plus proche : {depths.mean():.2f} voxels")
print(f"Distance max : {depths.max():.2f} voxels")




out_img = nib.Nifti1Image(combined.astype(np.float32), img_sub.affine, img_sub.header)
nib.save(out_img, "atlas_HO_115.nii")

# JSON des labels
labels_cortl = atlas_cortl.labels  # 97 items (0=Background + 96 régions)
labels_sub = atlas_sub.labels      # 22 items (0=Background + 21 régions)

label_map = {}
for i in range(1, 22):
    if i in (2, 13):
        continue
    label_map[str(i)] = labels_sub[i]
for i in range(1, 97):
    label_map[str(100 + i)] = labels_cortl[i]

import json
with open("atlas_labels_HO_115.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=4, ensure_ascii=False)

print(f"{len(label_map)} régions (attendu : 19 + 96 = 115)")





# Voxels où le sub assigne une vraie structure (pas 0, pas 2/13 générique)
mask_sub_specific = ~np.isin(data_sub, [0, 2, 13])

# Parmi ceux-là, combien le cortl aurait aussi voulu réclamer ?
conflict_mask = mask_sub_specific & (data_cortl > 0)
n_conflict = conflict_mask.sum()
n_sub_specific = mask_sub_specific.sum()

print(f"{n_conflict} voxels en conflit sur {n_sub_specific} voxels subcorticaux "
      f"({100*n_conflict/n_sub_specific:.1f}%)")

# Détail par région subcorticale touchée
for label_val in np.unique(data_sub[conflict_mask]):
    n = (data_sub[conflict_mask] == label_val).sum()
    print(f"  Label sub {int(label_val)} : {n} voxels en conflit")


# Conflit réel, en excluant la matière blanche (labels 1, 12)
real_conflict_mask = mask_sub_specific & (data_cortl > 0) & ~np.isin(data_sub, [1, 12])
n_real_conflict = real_conflict_mask.sum()
n_real_sub_specific = (mask_sub_specific & ~np.isin(data_sub, [1, 12])).sum()

print(f"{n_real_conflict} voxels en conflit sur {n_real_sub_specific} voxels "
      f"({100*n_real_conflict/n_real_sub_specific:.2f}%)")