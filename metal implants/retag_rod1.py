import numpy as np
import nibabel as nib
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_NII = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5metalrod2/label_prep/tissue_labeling_upsampled.nii.gz"

OUTPUT_NII = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5metalrod2/label_prep/tissue_labeling_upsampled.nii.gz"

ROD_MASK_OUTPUT = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5metalrod2/left_jaw_metal_rod_mask.nii.gz"
SHELL_MASK_OUTPUT = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5metalrod2/left_jaw_refinement_shell_mask.nii.gz"

ROD_TAG = 51
SHELL_TAG = 52

# Same cylinder axis as your correctly placed rod
P1_VOX = np.array([135, 115, 125], dtype=float)
P2_VOX = np.array([180, 115, 105], dtype=float)

# Actual metal rod radius
ROD_RADIUS_MM = 4.0

# Larger refinement region radius
SHELL_RADIUS_MM = 10.0

# Only overwrite likely jaw-region tissues
ONLY_OVERWRITE_SELECTED_TAGS = True

# 4 bone, 5 scalp, 7 compact bone, 8 spongy bone, 9 blood, 10 muscle
TAGS_ALLOWED_TO_OVERWRITE = [4, 5, 7, 8, 9, 10]


# ============================================================
# FUNCTIONS
# ============================================================

def voxel_to_world(vox_points, affine):
    vox_points = np.asarray(vox_points, dtype=float)

    if vox_points.ndim == 1:
        vox_h = np.append(vox_points, 1.0)
        return (affine @ vox_h)[:3]

    ones = np.ones((vox_points.shape[0], 1))
    vox_h = np.hstack([vox_points, ones])
    return (affine @ vox_h.T).T[:, :3]


def make_cylinder_mask(shape, affine, p1_vox, p2_vox, radius_mm):
    p1_mm = voxel_to_world(p1_vox, affine)
    p2_mm = voxel_to_world(p2_vox, affine)

    axis = p2_mm - p1_mm
    axis_len2 = np.dot(axis, axis)

    if axis_len2 == 0:
        raise ValueError("P1_VOX and P2_VOX are identical.")

    i, j, k = np.indices(shape)
    vox = np.stack([i, j, k], axis=-1).reshape(-1, 3)

    world = voxel_to_world(vox, affine)

    w = world - p1_mm
    t = np.dot(w, axis) / axis_len2

    inside_caps = (t >= 0.0) & (t <= 1.0)

    closest = p1_mm + t[:, None] * axis
    dist = np.linalg.norm(world - closest, axis=1)

    inside_radius = dist <= radius_mm

    return (inside_caps & inside_radius).reshape(shape)


# ============================================================
# MAIN
# ============================================================

def main():
    input_path = Path(INPUT_NII)
    output_path = Path(OUTPUT_NII)

    img = nib.load(str(input_path))
    data = img.get_fdata().astype(np.int16)

    affine = img.affine
    header = img.header.copy()

    print("Input:", input_path)
    print("Shape:", data.shape)
    print("Voxel sizes:", img.header.get_zooms()[:3])

    rod_mask = make_cylinder_mask(
        shape=data.shape,
        affine=affine,
        p1_vox=P1_VOX,
        p2_vox=P2_VOX,
        radius_mm=ROD_RADIUS_MM,
    )

    outer_mask = make_cylinder_mask(
        shape=data.shape,
        affine=affine,
        p1_vox=P1_VOX,
        p2_vox=P2_VOX,
        radius_mm=SHELL_RADIUS_MM,
    )

    shell_mask = outer_mask & ~rod_mask

    if ONLY_OVERWRITE_SELECTED_TAGS:
        allowed = np.isin(data, TAGS_ALLOWED_TO_OVERWRITE)
        rod_mask = rod_mask & allowed
        shell_mask = shell_mask & allowed

    print("Rod voxels:", int(np.sum(rod_mask)))
    print("Shell voxels:", int(np.sum(shell_mask)))

    out = data.copy()

    # Write shell first, then rod second so rod wins in overlap
    out[shell_mask] = SHELL_TAG
    out[rod_mask] = ROD_TAG

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_img = nib.Nifti1Image(out.astype(np.int16), affine, header)
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, str(output_path))

    rod_img = nib.Nifti1Image(rod_mask.astype(np.uint8), affine, header)
    rod_img.set_data_dtype(np.uint8)
    nib.save(rod_img, ROD_MASK_OUTPUT)

    shell_img = nib.Nifti1Image(shell_mask.astype(np.uint8), affine, header)
    shell_img.set_data_dtype(np.uint8)
    nib.save(shell_img, SHELL_MASK_OUTPUT)

    print("\nSaved edited tissue labeling:")
    print(output_path)

    print("\nSaved rod mask:")
    print(ROD_MASK_OUTPUT)

    print("\nSaved shell mask:")
    print(SHELL_MASK_OUTPUT)


if __name__ == "__main__":
    main()