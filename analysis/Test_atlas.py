import nibabel as nib
import numpy as np
from pathlib import Path
from simnibs import mesh_io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Atlas ─────────────────────────────────────────────────────────────────────
atlas_path = Path(__file__).parent.parent / "utils" / "atlas.nii"

atlas = nib.load(atlas_path)

print("Atlas shape   :", atlas.shape)
print("Atlas zooms   :", atlas.header.get_zooms())
print("Atlas affine  :\n", atlas.affine)

# coins du volume atlas en mm
corners_vox = np.array([
    [0, 0, 0],
    [atlas.shape[0], 0, 0],
    [0, atlas.shape[1], 0],
    [0, 0, atlas.shape[2]],
    [atlas.shape[0], atlas.shape[1], atlas.shape[2]]
])
corners_mm = nib.affines.apply_affine(atlas.affine, corners_vox)
print("\nAtlas extent in mm :")
print(f"  X : [{corners_mm[:,0].min():.1f}, {corners_mm[:,0].max():.1f}]")
print(f"  Y : [{corners_mm[:,1].min():.1f}, {corners_mm[:,1].max():.1f}]")
print(f"  Z : [{corners_mm[:,2].min():.1f}, {corners_mm[:,2].max():.1f}]")

# ── Mesh ──────────────────────────────────────────────────────────────────────
msh = mesh_io.read_msh(Path("data/ernie/m2m_ernie/ernie.msh"))
coords = msh.nodes.node_coord
print("\nMesh extent in mm :")
print(f"  X : [{coords[:,0].min():.1f}, {coords[:,0].max():.1f}]")
print(f"  Y : [{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]")
print(f"  Z : [{coords[:,2].min():.1f}, {coords[:,2].max():.1f}]")


# ─── 5. (OPTIONNEL) EXPLORER UNE TRANCHE SPÉCIFIQUE ─────────────────────────

def afficher_tranche(volume, axe="z", index=None):
    """
    Affiche une tranche précise du volume.
    axe   : 'x', 'y', ou 'z'
    index : numéro de la tranche (None = milieu)
    """
    n    = len(np.unique(volume).astype(int))
    cmap = plt.colormaps["nipy_spectral"].resampled(n)   # corrigé ici aussi
    vmin = np.unique(volume).astype(int).min()
    vmax = np.unique(volume).astype(int).max()

    dim = {"x": 0, "y": 1, "z": 2}[axe]
    idx = index if index is not None else volume.shape[dim] // 2

    slc = (volume[idx, :, :] if axe == "x"
           else volume[:, idx, :] if axe == "y"
           else volume[:, :, idx])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(slc.T, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Label de région")
    ax.set_title(f"Tranche {axe.upper()} = {idx}")
    ax.axis("off")
    plt.show()

# Exemple d'utilisation :
data = atlas.get_fdata()
afficher_tranche(data, axe="z", index=108)