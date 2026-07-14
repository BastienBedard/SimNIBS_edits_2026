import json
import os
from simnibs import sim_struct, run_simnibs, opt_struct, SubjectFiles
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent

SUBJECT_PATH = PROJECT_ROOT / "data" / "ernie" / "m2m_ernie"
COIL_DIR     = Path(sim_struct.__file__).parent.parent / "resources" / "coil_models"
JSON_PATH    = PROJECT_ROOT / "data" / "protocoles.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "results_tms_MA"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── CHARGEMENT DU JSON ───────────────────────────────────────────────────────
with open(JSON_PATH, "r", encoding="utf-8") as f:
    protocoles = json.load(f)

# ─── FONCTION DE SIMULATION ───────────────────────────────────────────────────
def run_tms_simulation(study_id, protocole):
    """
    Lance une simulation TMS SimNIBS pour un protocole donné.
    Gère les coils flexibles (H1/H4/H7, MST-Twin) via TmsFlexOptimization.
    """
    sim_cfg = protocole["simulation"]

    eeg_position, coil_direction = sim_cfg["coil_position_10_20"], sim_cfg["coil_direction"]

    coil_file = COIL_DIR / sim_cfg["coil_folder"] / sim_cfg["coil_file"]

    # Vérification que le fichier de bobine existe
    if not os.path.isfile(coil_file):
        print(f"  [ERREUR] Fichier de bobine introuvable : {coil_file}")
        return

    out_dir = os.path.join(RESULTS_DIR, study_id)
    print(f"\n  → Simulation {study_id} | Position: {eeg_position} | Bobine: {sim_cfg['coil_file']}")

    # Coil flexible (helmet) -> nécessite TmsFlexOptimization plutôt qu'un placement rigide
    is_flexible = sim_cfg["coil_folder"] == "flexible_coils"

    if is_flexible:
        _run_flex_coil(study_id, out_dir, coil_file, eeg_position, coil_direction)
    else:
        _run_rigid_coil(study_id, out_dir, coil_file, eeg_position, coil_direction)


def _run_rigid_coil(study_id, out_dir, coil_file, eeg_position, coil_direction):
    """Placement rigide standard (coils figure-8, plats)."""
    S = sim_struct.SESSION()
    S.subpath  = SUBJECT_PATH
    S.pathfem  = out_dir
    S.open_in_gmsh = False

    tms = S.add_tmslist()
    tms.fnamecoil = coil_file
    pos = tms.add_position()
    pos.centre   = eeg_position
    pos.distance = 4
    pos.pos_ydir = coil_direction

    try:
        run_simnibs(S)
        print(f"  [OK] Simulation terminée → {S.pathfem}")
    except Exception as e:
        print(f"  [ERREUR] {study_id} : {e}")


def _run_flex_coil(study_id, out_dir, coil_file, eeg_position, coil_direction):
    """Optimisation + simulation pour un coil flexible (H1/H4/H7, MST-Twin)."""
    tms_opt = opt_struct.TmsFlexOptimization()
    tms_opt.subpath = str(SUBJECT_PATH)
    tms_opt.path_optimization = out_dir
    tms_opt.fnamecoil = str(coil_file)

    # cap EEG explicite (pas hérité automatiquement hors SESSION)
    sub_files = SubjectFiles(subpath=str(SUBJECT_PATH))
    tms_opt.eeg_cap = sub_files.eeg_cap_1010

    tms_opt.method = "distance"
    tms_opt.distance = 0            # H4 = casque capitonné, 0 mm comme dans l'exemple officiel
    tms_opt.open_in_gmsh = False

    pos = tms_opt.add_position()
    pos.centre   = eeg_position
    pos.pos_ydir = coil_direction

    # bornes de recherche ajusté pour la position Fz orienté Fpz 
    tms_opt.global_translation_ranges = [[-5, 5], [-30, 30], [-50, 50]]
    tms_opt.global_rotation_ranges = [[-40, 40], [-2, 2], [-1, 1]]

    try:
        tms_opt.run()
        print(f"  [OK] Optimisation + simulation terminées → {out_dir}")
    except Exception as e:
        print(f"  [ERREUR] {study_id} : {e}")

def valider_protocole(study_id, protocole):
    """
    Vérifie que tous les champs requis pour la simulation sont présents et non "NR".
    Retourne True si le protocole est valide, False sinon.
    """
    # Champs obligatoires pour lancer une simulation
    # Format : ("section", "clé")
    champs_requis = [
        ("simulation", "coil_file"),
        ("simulation", "coil_folder"),
        ("simulation", "coil_position_10_20"),
        ("simulation", "coil_direction")
    ]

    champs_manquants = []

    for section, cle in champs_requis:
        valeur = protocole.get(section, {}).get(cle)
        if valeur is None or valeur == "NR":
            champs_manquants.append(f"{section}.{cle}")

    if champs_manquants:
        print(f"  [SKIP] {study_id} — champs manquants ou non renseignés (NR) :")
        for champ in champs_manquants:
            print(f"    - {champ}")
        return False

    return True

def afficher_notes(study_id, protocole):
    """
    Affiche toutes les notes présentes dans les sections simulation et dosimetry.
    """
    sections = ["simulation", "dosimetry"]
    notes_trouvees = []

    for section in sections:
        contenu = protocole.get(section, {})
        for cle, valeur in contenu.items():
            if cle == "note" or cle.endswith("_note"):
                notes_trouvees.append((section, cle, valeur))

    if notes_trouvees:
        print(f"  [NOTES]")
        for section, cle, valeur in notes_trouvees:
            print(f"    {section}.{cle} : {valeur}")

def get_simulations_a_faire(protocoles):
    """
    Compare les protocoles du JSON avec les dossiers déjà présents dans RESULTS_DIR.
    Retourne uniquement les protocoles qui n'ont pas encore été simulés.
    """
    print("\n" + "=" * 55)
    print("VÉRIFICATION DES SIMULATIONS EXISTANTES")
    print("=" * 55)

    a_faire = {}

    for study_id, protocole in protocoles.items():
        sim_cfg = protocole["simulation"]

        dossier = RESULTS_DIR / study_id
        deja_fait = dossier.exists()

        if deja_fait:
            print(f"  [SKIP] {study_id} — déjà simulé")
        else:
            print(f"  [TODO] {study_id} — à simuler")
            a_faire[study_id] = protocole

        # Affichage des notes pour tous les protocoles
        afficher_notes(study_id, protocole)

    print(f"\n  {len(a_faire)}/{len(protocoles)} protocoles à simuler")
    return a_faire

# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────

print("=" * 55)
print("LANCEMENT DES SIMULATIONS TMS")
print("=" * 55)

a_faire = get_simulations_a_faire(protocoles)

if not a_faire:
    print("\nToutes les simulations sont déjà effectuées.")
else:
    for study_id, protocole in a_faire.items():
        print(f"\n[{study_id}]")
        print(f"  Région   : {protocole['classification']['target_region']} "
            f"({protocole['classification']['hemisphere']})")
        print(f"  Protocole: {protocole['dosimetry']['protocol_type']} "
            f"| {protocole['dosimetry']['frequency_hz']} Hz")
        
        if not valider_protocole(study_id, protocole):
            continue   # passe au protocole suivant sans simuler

        run_tms_simulation(study_id, protocole)

print("\n" + "=" * 55)
print("TOUTES LES SIMULATIONS TERMINÉES")
print("=" * 55)