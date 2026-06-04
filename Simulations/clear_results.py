import shutil
import os
from pathlib import Path

# ─── CHEMINS ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "results_tms_MA"

# ─── FONCTIONS ────────────────────────────────────────────────────────────────

def lister_simulations():
    """Affiche tous les dossiers de simulation présents."""
    dossiers = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    if not dossiers:
        print("  Aucune simulation trouvée dans le dossier.")
        return []
    print(f"\nSimulations présentes dans {RESULTS_DIR} :")
    for i, d in enumerate(dossiers):
        print(f"  [{i}] {d.name}")
    return dossiers


def supprimer_tout():
    """Supprime tous les dossiers de simulation."""
    dossiers = lister_simulations()
    if not dossiers:
        return

    confirmation = input("\nSupprimer TOUTES les simulations? (y/n) : ").strip().lower()
    if confirmation != "y":
        print("Annulé.")
        return

    for d in dossiers:
        shutil.rmtree(d)
        print(f"  [OK] Supprimé : {d.name}")
    print("\nTous les résultats ont été supprimés.")


def supprimer_un():
    """Supprime un seul dossier de simulation choisi par l'utilisateur."""
    dossiers = lister_simulations()
    if not dossiers:
        return

    choix = input("\nEntrer le numéro du dossier à supprimer : ").strip()
    try:
        index = int(choix)
        dossier = dossiers[index]
    except (ValueError, IndexError):
        print("Numéro invalide.")
        return

    confirmation = input(f"Supprimer '{dossier.name}'? (y/n) : ").strip().lower()
    if confirmation != "y":
        print("Annulé.")
        return

    shutil.rmtree(dossier)
    print(f"  [OK] Supprimé : {dossier.name}")


# ─── MENU PRINCIPAL ───────────────────────────────────────────────────────────

print("=" * 45)
print("GESTION DES RÉSULTATS DE SIMULATION")
print("=" * 45)
print("  [1] Supprimer toutes les simulations")
print("  [2] Supprimer une simulation spécifique")

action = input("\nChoix : ").strip()

if action == "1":
    supprimer_tout()
elif action == "2":
    supprimer_un()