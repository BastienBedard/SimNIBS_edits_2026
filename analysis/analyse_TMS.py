import simnibs
from simnibs import mesh_io
import numpy as np
from pathlib import Path
import nibabel as nib
import json
from nibabel.affines import apply_affine
from simnibs import mni2subject_coords, subject2mni_coords

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "results_tms_MA"
ATLAS_PATH   = PROJECT_ROOT / "utils" / "atlas.nii"
SUBJECT_PATH = PROJECT_ROOT / "data" / "ernie" / "m2m_ernie" 
ATLAS_LABELS_PATH = PROJECT_ROOT / "utils" / "atlas_labels.json"

def _update_slice(plotter, mesh, axis, value, row, col):
    """
    Callback pour les sliders — coupe le mesh selon l'axe et la valeur donnés.
    """
    plotter.subplot(row, col)
    plotter.clear_actors()

    normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    origin = {"x": (value, 0, 0), "y": (0, value, 0), "z": (0, 0, value)}[axis]

    clipped = mesh.clip(normal=normal, origin=origin)
    plotter.add_mesh(clipped)
    plotter.render()

class ProtocoleAnalysis:
    """
    Charge et analyse le champ électrique d'un protocole TMS simulé.
    Associe les tétraèdres du .msh aux régions de l'atlas.
    """
    with open(ATLAS_LABELS_PATH, "r") as f:
        ATLAS_LABELS = {int(k): v for k, v in json.load(f).items()}
    def __init__(self, study_id, tissue_tags=None):
        # study_id    → clé dans le JSON (ex: "eichhammer2003")
        # tissue_tags → [1, 2] = WM + GM par défaut
        
        self.study_id      = study_id
        self.tissue_tags   = tissue_tags if tissue_tags is not None else [1, 2]
        self.msh           = None   # maillage chargé
        self.atlas         = None   # atlas NIfTI
        self.magnE         = None   # valeurs du champ E filtrées
        self.region_labels = None   # label atlas pour chaque tétraèdre
        self.tissue_mask   = None   # mask du mesh pour les tissus selectionné
        self.vols          = None   # volumes des tétraèdres
        

        self._load_msh()
        self._load_atlas()
        self._msh_in_tissue()
        self._assign_regions()

    def _load_msh(self):
        """Charge le .msh et extrait magnE pour les tissus d'intérêt."""
        if self.msh is None:
            study_folder = RESULTS_DIR / self.study_id

            fichiers_msh = [f for f in study_folder.glob("*.msh") if f.is_file()]

            if len(fichiers_msh) != 1:
                raise ValueError(f"Nombre inattendu de fichiers .msh dans {self.study_id} : {len(fichiers_msh)}")

            msh_file = fichiers_msh[0]
            self.msh = mesh_io.read_msh(msh_file)

    def _load_atlas(self):
        """Charge l'atlas .nii et prépare la matrice affine."""
        self.atlas = nib.load(ATLAS_PATH)

    def _assign_regions(self):
        """
        Associe chaque tétraèdre (GM/WM) à une région de l'atlas
        via le barycentre des éléments transformé en espace MNI.
        """
        # 1. barycentres des tétraèdres en espace natif
        centers = self.msh.elements_baricenters()[:]
        centers = centers[self.tissue_mask]

        # 2. transformation espace natif Ernie → espace MNI
        centers_mni = subject2mni_coords(centers, str(SUBJECT_PATH))

        # 3. passage en coordonnées voxel atlas
        inv_affine = np.linalg.inv(self.atlas.affine)
        vox = apply_affine(inv_affine, centers_mni)

        # 4. indices voxel entiers
        vox = np.round(vox).astype(int)

        # 5. atlas data
        atlas_data = self.atlas.get_fdata()
        shape = atlas_data.shape

        # 6. masque de validité
        valid = (
            (vox[:, 0] >= 0) & (vox[:, 0] < shape[0]) &
            (vox[:, 1] >= 0) & (vox[:, 1] < shape[1]) &
            (vox[:, 2] >= 0) & (vox[:, 2] < shape[2])
        )

        # 7. labels région
        labels = np.zeros(len(vox), dtype=int)
        labels[valid] = atlas_data[
            vox[valid, 0],
            vox[valid, 1],
            vox[valid, 2]
        ].astype(int)

        # 8. stockage
        self.region_labels = labels

    def _msh_in_tissue(self):
        """
        Retourne le mesh associé a un ou plusieurs tissus selon tissue_tags
        """
        elm_tags         = self.msh.elm.tag1 # tag1 : 1=WM, 2=GM (tétraèdres) — 1001/1002 = surfaces (ignorées automatiquement)
        self.tissue_mask = np.isin(elm_tags, self.tissue_tags) 
        self.magnE       = self.msh.field["magnE"][self.tissue_mask]
        self.vols        = self.msh.elements_volumes_and_areas()[self.tissue_mask]

    def _get_field(self, name):
        for field in self.msh.elmdata:
            if field.field_name == name:
                return field
        return None

    def get_region_label(self, region_id, use_names=None):
        """
        Retourne le label d'une région selon le format choisi.
        use_names : None  → "Frontal_Mid_2_L - 5" (défaut)
                    True  → "Frontal_Mid_2_L"
                    False → "5"
        """
        name = self.ATLAS_LABELS.get(region_id, f"Unknown_{region_id}")
        if use_names is None:
            return f"{name} - {region_id}"
        elif use_names:
            return name
        else:
            return str(region_id)

    def get_labels_for_regions(self, region_ids, use_names=None):
        """
        Retourne une liste de labels pour une liste de region_ids.
        use_names : None  → "nom - #" (défaut)
                    True  → nom anatomique seulement
                    False → numéro seulement
        """
        return [self.get_region_label(r, use_names) for r in region_ids]

    def region_stats(self):
        """
        Retourne un dict {region_label: {"n_elements": int, "volume_mm3": float}}
        pour chaque région de l'atlas.
        """
        labels = self.region_labels
        stats  = {}

        for r in np.unique(labels):
            if r == 0:
                continue

            mask = labels == r
            if not np.any(mask):
                continue

            stats[r] = {
                "n_elements": int(np.sum(mask)),
                "volume_mm3": float(np.sum(self.vols[mask]))
            }

        return stats

    def mean_by_region(self):
        """Retourne un dict {region_label: moyenne_magnE}."""
        
        labels = self.region_labels
        E = self.magnE
        vols = self.vols

        # dictionnaire résultat
        region_means = {}


        for r in np.unique(labels):
            if r == 0:
                continue  # hors atlas

            mask = labels == r

            if not np.any(mask):
                continue

            region_means[r] = np.average(E[mask], weights=vols[mask])

        return region_means

    def top_percentile_volume(self, percentile=95):
        """
        Retourne un dict {region_label: valeur_magnE au percentile donné}
        pour chaque région de l'atlas.
        
        percentile : float entre 0 et 100 (ex: 95 pour P95)
        """
        labels = self.region_labels
        E      = self.magnE

        region_percentiles = {}

        for r in np.unique(labels):
            if r == 0:
                continue

            mask = labels == r
            if not np.any(mask):
                continue

            region_percentiles[r] = np.percentile(E[mask], percentile)
        return region_percentiles

    def fraction_above_threshold(self, threshold_pct, by_volume=True):
        """
        Pour chaque région, retourne le % de tétraèdres ou de volume dont magnE
        dépasse threshold_pct% de la moyenne globale GM+WM.
        
        threshold_pct : float entre 0 et 100 (ex: 50 pour 50% de la moyenne globale)
        """
        labels = self.region_labels
        E      = self.magnE

        global_mean = np.average(E, weights=self.vols)
        threshold   = (threshold_pct / 100) * global_mean

        region_fractions = {}

        for r in np.unique(labels):
            if r == 0:
                continue

            mask = labels == r
            if not np.any(mask):
                continue

            above = E[mask] > threshold

            if by_volume:
                region_fractions[r] = (
                    np.sum(self.vols[mask][above]) /
                    np.sum(self.vols[mask]) * 100
                )
            else:
                region_fractions[r] = np.sum(above) / np.sum(mask) * 100

        return region_fractions

    def regions_below_threshold(self, threshold_pct, by_volume=True):
        """
        Pour chaque région, retourne le % de volume (ou de tétraèdres) dont magnE
        est SOUS threshold_pct% de la moyenne globale GM+WM.
        
        threshold_pct : float entre 0 et 100
        by_volume     : True  → % de volume (défaut)
                        False → % de tétraèdres
        """
        labels = self.region_labels
        E      = self.magnE
        vols   = self.vols

        global_mean = np.average(E, weights=vols)
        threshold   = (threshold_pct / 100) * global_mean

        region_fractions = {}

        for r in np.unique(labels):
            if r == 0:
                continue

            mask = labels == r
            if not np.any(mask):
                continue

            below = E[mask] < threshold

            if by_volume:
                region_fractions[r] = (
                    np.sum(vols[mask][below]) /
                    np.sum(vols[mask]) * 100
                )
            else:
                region_fractions[r] = np.sum(below) / np.sum(mask) * 100

        return region_fractions

    def rank_regions(self, n=5, metric="mean", percentile=95,
                 threshold_pct=50, by_volume=True, ascending=False):
        """
        Retourne les n régions classées selon la métrique choisie.

        metric        : 'mean'             → moyenne pondérée par volume
                        'percentile'       → valeur au P(percentile)
                        'focality'         → ratio P(percentile)/moyenne
                        'above_threshold'  → % de volume/tétraèdres au dessus du seuil
                        'below_threshold'  → % de volume/tétraèdres sous le seuil

        percentile    : utilisé si metric='percentile' ou 'focality' (défaut 95)
        threshold_pct : utilisé si metric='above_threshold' ou 'below_threshold' (défaut 50)
        by_volume     : utilisé si metric='above_threshold' ou 'below_threshold' (défaut True)
        ascending     : False → plus stimulées en premier (défaut)
                        True  → moins stimulées en premier

        Retourne : liste de tuples (region_label, valeur) triée
        """

        if metric == "mean":
            scores = self.mean_by_region()

        elif metric == "percentile":
            scores = self.top_percentile_volume(percentile)

        elif metric == "focality":
            means       = self.mean_by_region()
            percentiles = self.top_percentile_volume(percentile)
            scores = {
                r: percentiles[r] / means[r]
                for r in percentiles
                if r in means and means[r] > 0
            }

        elif metric == "above_threshold":
            scores = self.fraction_above_threshold(threshold_pct, by_volume)

        elif metric == "below_threshold":
            scores = self.regions_below_threshold(threshold_pct, by_volume)

        else:
            raise ValueError(f"metric '{metric}' invalid. "
                            f"Choose from: 'mean', 'percentile', 'focality', "
                            f"'above_threshold', 'below_threshold'")

        sorted_regions = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)

        return sorted_regions if n == "all" else sorted_regions[:n]
