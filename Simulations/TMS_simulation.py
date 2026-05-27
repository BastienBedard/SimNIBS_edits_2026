import json
import os
from simnibs import sim_struct, run_simnibs
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
    Gère les cas unilatéraux et bilatéraux automatiquement.
    """
    sim_cfg = protocole["simulation"]

    # Déterminer si la simulation est bilatérale
    bilateral_method = sim_cfg.get("bilateral_method", None)

    if bilateral_method == "two_coils":
        positions = {
            "L": (sim_cfg["coil_position_10_20_L"], sim_cfg["coil_direction_L"]),
            "R": (sim_cfg["coil_position_10_20_R"], sim_cfg["coil_direction_R"]),
        }
    else:
        positions    = {"": (sim_cfg["coil_position_10_20"], sim_cfg["coil_direction"])}

    coil_file = COIL_DIR / sim_cfg["coil_folder"] / sim_cfg["coil_file"]

    # Vérification que le fichier de bobine existe
    if not os.path.isfile(coil_file):
        print(f"  [ERREUR] Fichier de bobine introuvable : {coil_file}")
        return

    # Boucle sur les positions (1 itération si unilatéral, 2 si bilatéral)
    for side, (eeg_position, coil_direction) in positions.items():

        suffix = f" ({side})" if side else ""
        print(f"\n  → Simulation {study_id}{suffix} | Position: {eeg_position} | Bobine: {sim_cfg['coil_file']}")

        # ── Configuration SimNIBS ─────────────────────────────────────────────
        S = sim_struct.SESSION()
        S.subpath  = SUBJECT_PATH
        S.pathfem  = os.path.join(RESULTS_DIR, study_id + (f"_{side}" if side else ""))
        S.open_in_gmsh = False   # ne pas ouvrir gmsh après chaque simulation

        # ── Paramètres TMS ────────────────────────────────────────────────────
        tms = S.add_tmslist()
        tms.fnamecoil = coil_file

        pos = tms.add_position()
        pos.centre   = eeg_position                  # label EEG → SimNIBS trouve les coordonnées
        pos.distance  = 4                            # distance bobine-scalp en mm (standard)
        pos.pos_ydir = coil_direction
        
        # ── Lancement ─────────────────────────────────────────────────────────
        try:
            run_simnibs(S)
            print(f"  [OK] Simulation terminée → {S.pathfem}")
        except Exception as e:
            print(f"  [ERREUR] {study_id}{suffix} : {e}")

def valider_protocole(study_id, protocole):
    """
    Vérifie que tous les champs requis pour la simulation sont présents et non "NR".
    Retourne True si le protocole est valide, False sinon.
    """
    sim_cfg  = protocole["simulation"]
    bilateral_method = sim_cfg.get("bilateral_method", None)
    # Champs obligatoires pour lancer une simulation
    # Format : ("section", "clé")
    champs_requis = [
        ("simulation", "coil_file"),
        ("simulation", "coil_folder"),
        ("dosimetry",  "intensity_pct_rMT")
    ]


    champs_manquants = []

    for section, cle in champs_requis:
        valeur = protocole.get(section, {}).get(cle)
        if valeur is None or valeur == "NR":
            if cle == "intensity_pct_rMT":
                print("Intensité manquante")
            else:
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
        bilateral = "coil_position_10_20_L" in sim_cfg

        if bilateral:
            # Les deux dossiers doivent exister pour considérer la sim complète
            dossier_L = RESULTS_DIR / f"{study_id}_L"
            dossier_R = RESULTS_DIR / f"{study_id}_R"
            deja_fait = dossier_L.exists() and dossier_R.exists()
        else:
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