import numpy as np
import nibabel as nib
import os

M2M = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5metalrod"

template_path = os.path.join(M2M, "label_prep", "tissue_labeling_upsampled.nii.gz")
out_path = os.path.join(M2M, "metal_rod_mask.nii.gz")

img = nib.load(template_path)
affine = img.affine
shape = img.shape[:3]

print("Template shape:", shape)

# ------------------------------------------------------------
# Rod placement in VOXEL coordinates: [i, j, k]
#
# In your check image:
#   Coronal j=194 means j is fixed at around 194.
#   Horizontal image direction roughly corresponds to i.
#   Vertical image direction roughly corresponds to k.
#
# The current rod is too central/posterior.
# These coordinates are an initial attempt to move it toward the jaw.
# You should adjust after viewing the PNG output.
# ------------------------------------------------------------

p1 = np.array([140, 115, 125])
p2 = np.array([180, 115, 105])
radius_vox = 4.0

# Create voxel grid
i, j, k = np.indices(shape)
P = np.stack([i, j, k], axis=-1).astype(float)

v = p2 - p1
w = P - p1

t = np.clip(np.sum(w * v, axis=-1) / np.sum(v * v), 0, 1)
closest = p1 + t[..., None] * v
dist = np.linalg.norm(P - closest, axis=-1)

rod = dist <= radius_vox

mask = np.zeros(shape, dtype=np.uint8)
mask[rod] = 1

nib.save(nib.Nifti1Image(mask, affine, img.header), out_path)

print("Saved rod mask to:", out_path)
print("Rod voxels:", int(rod.sum()))
print("Rod p1 voxel:", p1)
print("Rod p2 voxel:", p2)
print("Rod radius voxels:", radius_vox)