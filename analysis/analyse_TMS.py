import simnibs
from simnibs import mesh_io
import numpy as np
from pathlib import Path


# ─── 1 CHARGEMENT DU FICHIER .msh des résultats après simulation───────────────────────────────────────────

class ProtocoleAnalysis:
    """
    Charge et analyse le champ électrique d'un protocole TMS simulé.
    Associe les tétraèdres du .msh aux régions de l'atlas.
    """
    def __init__(self, study_id, tissue_tags=[1, 2]):
        # study_id    → clé dans le JSON (ex: "eichhammer2003")
        # tissue_tags → [1, 2] = WM + GM par défaut
        
        self.study_id    = study_id
        self.tissue_tags = tissue_tags
        self.msh         = None   # maillage chargé
        self.atlas       = None   # atlas NIfTI
        self.magnE       = None   # valeurs du champ E filtrées
        self.region_labels = None # label atlas pour chaque tétraèdre
        
        self._load_msh()
        self._load_atlas()
        self._assign_regions()

    def _load_msh(self):
        """Charge le .msh et extrait magnE pour les tissus d'intérêt."""
        ...

    def _load_atlas(self):
        """Charge l'atlas .nii et prépare la matrice affine."""
        ...

    def _assign_regions(self):
        """
        Pour chaque tétraèdre du tissu d'intérêt,
        trouve le label de région correspondant dans l'atlas.
        """
        ...

    def mean_by_region(self):
        """Retourne un dict {region_label: moyenne_magnE}."""
        ...

    def top_percentile_volume(self, percentile=95):
        """
        Retourne un dict {region_label: valeur_magnE}
        au percentile donné pour chaque région.
        """
        ...

    def fraction_above_threshold(self, threshold_pct):
        """
        Pour chaque région, retourne le % de tétraèdres dont magnE
        dépasse threshold_pct% du champ moyen global GM+WM.
        """
        ...

    def top_regions(self, n=5, metric="mean"):
        """
        Retourne les n régions avec le champ E le plus élevé.
        metric : 'mean' ou 'p95'
        """
        ...