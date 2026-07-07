import numpy as np
import nibabel as nib
from simnibs import mesh_io
from collections import defaultdict

ORIGINAL_MESH = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5/ernie5.msh"
ROD_MASK = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5metalrod/metal_rod_mask.nii.gz"
OUT_MESH = "/Users/noahholm/Desktop/m2m_folders/m2m_ernie5metalrod/ernie5metalrod.msh"

ROD_TAG = 51

# Main controls
TAG_IF_CENTER_INSIDE = True
TAG_IF_ANY_VERTEX_INSIDE = True

# Add neighboring tetrahedra around the selected rod region.
# 0 = no expansion
# 1 = add face-neighboring tetrahedra
# 2 = add neighbors of neighbors
ADJACENCY_LAYERS = 4

# Optional: only expand into these original tissue tags.
# This prevents the rod from expanding into air/electrodes/etc.
# For jaw, compact/spongy bone and muscle are usually the most relevant.
# SimNIBS tags:
# 7 = compact bone, 8 = spongy bone, 10 = muscle, 5 = scalp
ALLOW_EXPANSION_IN_TAGS = None

m = mesh_io.read_msh(ORIGINAL_MESH)

mask_img = nib.load(ROD_MASK)
mask = mask_img.get_fdata() > 0
inv_affine = np.linalg.inv(mask_img.affine)

nodes = m.nodes.node_coord

# Tetrahedra only
tet_idx = np.where(m.elm.elm_type == 4)[0]
tet_nodes = m.elm.node_number_list[tet_idx, :4] - 1
tet_original_tags = m.elm.tag1[tet_idx].copy()

def world_to_voxel(points):
    points_h = np.c_[points, np.ones(len(points))]
    vox = points_h @ inv_affine.T
    return np.round(vox[:, :3]).astype(int)

def inside_mask(points):
    ijk = world_to_voxel(points)

    valid = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < mask.shape[0]) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < mask.shape[1]) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < mask.shape[2])
    )

    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]]
    return inside

# ------------------------------------------------------------
# Initial selection: center and/or vertex inside rod mask
# ------------------------------------------------------------

selected = np.zeros(len(tet_idx), dtype=bool)

centers = nodes[tet_nodes].mean(axis=1)
center_inside = inside_mask(centers)

if TAG_IF_CENTER_INSIDE:
    selected |= center_inside

vertex_inside = np.zeros(len(tet_idx), dtype=bool)

if TAG_IF_ANY_VERTEX_INSIDE:
    for local_vertex in range(4):
        pts = nodes[tet_nodes[:, local_vertex]]
        vertex_inside |= inside_mask(pts)

    selected |= vertex_inside

print("Initial center-inside tetrahedra:", int(np.sum(center_inside)))
print("Initial vertex-inside tetrahedra:", int(np.sum(vertex_inside)))
print("Initial selected tetrahedra:", int(np.sum(selected)))

# ------------------------------------------------------------
# Build face adjacency between tetrahedra
# Two tetrahedra are neighbors if they share a triangular face.
# ------------------------------------------------------------

faces_by_key = defaultdict(list)

# Local faces of a tetrahedron, defined by its 4 local node indices
local_faces = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
]

for local_tet_id, conn in enumerate(tet_nodes):
    for face in local_faces:
        face_nodes = tuple(sorted(conn[list(face)]))
        faces_by_key[face_nodes].append(local_tet_id)

neighbors = [[] for _ in range(len(tet_idx))]

for face_nodes, attached_tets in faces_by_key.items():
    if len(attached_tets) == 2:
        a, b = attached_tets
        neighbors[a].append(b)
        neighbors[b].append(a)

# ------------------------------------------------------------
# Expand selected region by adjacency layers
# ------------------------------------------------------------

if ALLOW_EXPANSION_IN_TAGS is None:
    allowed_to_expand = np.ones(len(tet_idx), dtype=bool)
else:
    allowed_tags = np.array(ALLOW_EXPANSION_IN_TAGS)
    allowed_to_expand = np.isin(tet_original_tags, allowed_tags)

expanded = selected.copy()

for layer in range(ADJACENCY_LAYERS):
    current = np.where(expanded)[0]
    new_selected = expanded.copy()

    for t in current:
        for nb in neighbors[t]:
            if allowed_to_expand[nb]:
                new_selected[nb] = True

    added_this_layer = int(np.sum(new_selected) - np.sum(expanded))
    expanded = new_selected

    print(f"Adjacency layer {layer + 1}: added {added_this_layer} tetrahedra")
    print(f"Total after layer {layer + 1}: {int(np.sum(expanded))}")

rod_local_tets = np.where(expanded)[0]
rod_global_tets = tet_idx[rod_local_tets]

if len(rod_global_tets) == 0:
    raise RuntimeError("No tetrahedra selected for rod.")

# Retag selected tetrahedra
m.elm.tag1[rod_global_tets] = ROD_TAG
m.elm.tag2[rod_global_tets] = ROD_TAG

mesh_io.write_msh(m, OUT_MESH)

print()
print("Original mesh:", ORIGINAL_MESH)
print("Rod mask:", ROD_MASK)
print("Output mesh:", OUT_MESH)
print("Total tetrahedra:", len(tet_idx))
print("Final rod tetrahedra:", int(len(rod_global_tets)))
print("Rod tag:", ROD_TAG)
print("Expansion layers:", ADJACENCY_LAYERS)
print("Expansion allowed in original tags:", ALLOW_EXPANSION_IN_TAGS)