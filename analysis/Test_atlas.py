from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── 1. CHARGEMENT DE L'ATLAS ────────────────────────────────────────────────

atlas_path = Path(__file__).parent.parent / "utils" / "atlas.nii"

img = nib.load(atlas_path)

data = img.get_fdata()          # matrice NumPy
affine = img.affine             # matrice de transformation voxel → mm
header = img.header             # métadonnées

# ─── 2. INFORMATIONS GÉNÉRALES ───────────────────────────────────────────────

print("=" * 50)
print("INFOS GÉNÉRALES")
print("=" * 50)
print(f"Shape       : {data.shape}")
print(f"Dtype       : {data.dtype}")
print(f"Min valeur  : {data.min()}")
print(f"Max valeur  : {data.max()}")
regions = np.unique(data).astype(int)
print(f"\nNombre de régions uniques : {len(regions)}")
# print(f"Labels des régions        : {regions}")


# for field in header.keys():
#     try:
#         val = header[field]
#         # Décoder les champs bytes en string lisible
#         if isinstance(val, bytes):
#             val = val.decode("utf-8", errors="ignore").strip("\x00").strip()
#         elif hasattr(val, "tobytes"):
#             decoded = val.tobytes().decode("utf-8", errors="ignore").strip("\x00").strip()
#             val = f"{val}  →  '{decoded}'" if decoded else val
#         print(f"  {field:<20} : {val}")
#     except Exception as e:
#         print(f"  {field:<20} : [erreur: {e}]")

print("=" * 50)

# ─── 3. AFFICHAGE DES TRANCHES ───────────────────────────────────────────────
# L'atlas est en 3D (x, y, z) → on prend la tranche du milieu de chaque axe.

vol = data

slice_x = vol[vol.shape[0] // 2, :, :]
slice_y = vol[:, vol.shape[1] // 2, :]
slice_z = vol[:, :, vol.shape[2] // 2]

vmin, vmax = regions.min(), regions.max()
n_regions  = len(regions)

# ── Correction : matplotlib.colormaps à la place de plt.cm.get_cmap ──────────
cmap = plt.colormaps["nipy_spectral"].resampled(n_regions)

slices = [slice_x, slice_y, slice_z]
titles = ["Sagittale (X)", "Coronale (Y)", "Axiale (Z)"]

# ── Correction colorbar : axes séparés avec gridspec ─────────────────────────
fig = plt.figure(figsize=(17, 5))
fig.suptitle("Atlas cérébral — Tranches centrales", fontsize=14, fontweight="bold")

gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

axes_img = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_cbar  = fig.add_subplot(gs[0, 3])

for ax, slc, title in zip(axes_img, slices, titles):
    im = ax.imshow(
        slc.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )
    ax.set_title(title)
    ax.axis("off")

# Colorbar dans sa propre colonne, sans chevauchement
fig.colorbar(im, cax=ax_cbar, label="Label de région")

plt.show()

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
# afficher_tranche(vol, axe="x", index=150)
