import simnibs
from simnibs import mesh_io
import numpy as np
from pathlib import Path
from nibabel.affines import apply_affine

# ─── 1. CHARGEMENT DU FICHIER .msh du cerveau d'ernie ───────────────────────────────────────────

# msh_path = Path(__file__).parent.parent / "data" / "ernie" / "m2m_ernie" / "ernie.msh"
# msh = mesh_io.read_msh(msh_path)

# ─── 1.2 CHARGEMENT DU FICHIER .msh des résultats après simulation───────────────────────────────────────────

msh_path_res = Path(__file__).parent.parent / "results" / "results_tms_MA" / "addicott2024" / "ernie_TMS_1-0001_MagVenture_MCF-B65_new_scalar.msh"
msh = mesh_io.read_msh(msh_path_res)

# ─── 2. STRUCTURE GÉNÉRALE ───────────────────────────────────────────────────

print("=" * 55)
print("STRUCTURE GÉNÉRALE DU MAILLAGE")
print("=" * 55)
print(f"Nombre de noeuds        : {msh.nodes.nr}")
print(f"Nombre d'éléments       : {msh.elm.nr}")
print(f"Types d'éléments uniques: {np.unique(msh.elm.elm_type)}")


print("\n" + "=" * 55)
print("TAGS ANATOMIQUES PRÉSENTS")
print("=" * 55)

TAG_NAMES = {
    1: "Matière blanche (WM)",
    2: "Matière grise (GM)",
    3: "LCR / CSF",
    4: "Crâne (skull)",
    5: "Peau (scalp)",
    1001: "Surface WM",
    1002: "Surface GM",
    1003: "Surface CSF",
    1004: "Surface skull",
    1005: "Surface scalp",
}

unique_tags = np.unique(msh.elm.tag1)
for tag in unique_tags:
    mask  = msh.elm.tag1 == tag
    count = np.sum(mask)
    name  = TAG_NAMES.get(tag, "inconnu")
    print(f"  Tag {tag:>5}  {name:<21} : {count:>8} éléments")

# ─── 4. CHAMPS DISPONIBLES (DONNÉES SCALAIRES / VECTORIELLES) ────────────────

print("\n" + "=" * 55)
print("CHAMPS DE DONNÉES DISPONIBLES")
print("=" * 55)

print(f"\n  Sur les noeuds ({msh.nodedata.__class__.__name__}) :")
if msh.nodedata:
    for field in msh.nodedata:
        print(f"    - '{field.field_name}'  shape: {field.value.shape}")
else:
    print("    (aucun)")

print(f"\n  Sur les éléments ({msh.elmdata.__class__.__name__}) :")
if msh.elmdata:
    for field in msh.elmdata:
        print(f"    - '{field.field_name}'  shape: {field.value.shape}")
else:
    print("    (aucun)")

# ─── 5. EXTRACTION DU CHAMP ÉLECTRIQUE ───────────────────────────────────────
# SimNIBS génère typiquement :
#   'E'       → vecteur du champ électrique (V/m)  shape: (n, 3)
#   'magnE'   → magnitude du champ électrique       shape: (n,)
#   'J'       → densité de courant                  shape: (n, 3)
#   'magnJ'   → magnitude de la densité de courant  shape: (n,)

print("\n" + "=" * 55)
print("CHAMP ÉLECTRIQUE — magnE")
print("=" * 55)

# Récupère un champ par son nom (cherche dans nodedata et elmdata)
def get_field(msh, name):
    for field in msh.elmdata:
        if field.field_name == name:
            return field
    return None

magnE_field = get_field(msh, "magnE")

if magnE_field is not None:
    magnE = magnE_field.value
    print(f"  Shape           : {magnE.shape}")
    print(f"  Min  (V/m)      : {magnE.min():.4f}")
    print(f"  Max  (V/m)      : {magnE.max():.4f}")
    print(f"  Moyenne (V/m)   : {magnE.mean():.4f}")
    print(f"  Médiane (V/m)   : {np.median(magnE):.4f}")
    print(f"  Écart-type      : {magnE.std():.4f}")
    print(f"  P95  (V/m)      : {np.percentile(magnE, 95):.4f}")
    print(f"  P99  (V/m)      : {np.percentile(magnE, 99):.4f}")
else:
    print("  Champ 'magnE' non trouvé.")
    print("  → Noms disponibles :", [f.field_name for f in msh.nodedata + msh.elmdata])

# ─── 6. EXTRACTION PAR RÉGION ANATOMIQUE ─────────────────────────────────────
# Exemple : magnE dans la matière grise uniquement (tag 2)

print("\n" + "=" * 55)
print("magnE PAR RÉGION ANATOMIQUE")
print("=" * 55)

if magnE_field is not None:
    # Détermine si le champ est sur les noeuds ou les éléments
    is_elm_field = magnE_field in msh.elmdata

    for tag, name in TAG_NAMES.items():
        if tag >= 1001:          # on ignore les surfaces pour cette analyse
            continue
        mask = msh.elm.tag1 == tag
        if not np.any(mask):
            continue

        if is_elm_field:
            values = magnE_field.value[mask]
        else:
            # Champ sur noeuds : récupère les noeuds de chaque élément du tag
            elm_nodes = msh.elm.node_number_list[mask] - 1   # index 0-based
            node_idx  = np.unique(elm_nodes[elm_nodes >= 0])
            values    = magnE_field.value[node_idx]

        print(f"\n  {name} (tag {tag}) — {len(values)} valeurs")
        print(f"    Moyenne : {values.mean():.4f} V/m")
        print(f"    Max     : {values.max():.4f} V/m")
        print(f"    P95     : {np.percentile(values, 95):.4f} V/m")

# ─── 7. COORDONNÉES DES NOEUDS ────────────────────────────────────────────────

print("\n" + "=" * 55)
print("test truc centre")
print("=" * 55)
centers = msh.elements_baricenters()[:]
print(type(centers))
print(np.shape(centers))
print("-----------------------")
# print(dir(msh.elmdata[1].value))
for i, d in enumerate(msh.elmdata):
    print(i, d.field_name)
    print(i, msh.elmdata[i].field_name, msh.elmdata[i].value.shape)