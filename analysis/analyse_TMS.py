from simnibs import mesh_io
import numpy as np
from pathlib import Path
import nibabel as nib
import json
from nibabel.affines import apply_affine
from simnibs import mni2subject_coords, subject2mni_coords
from nilearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
ATLAS_PATH   = PROJECT_ROOT / "utils" / "atlas_HO_AAL3_139.nii"
ATLAS_LABELS_PATH = PROJECT_ROOT / "utils" / "atlas_labels_HO_AAL3_139.json"

# ─── HEAD MODELS ─────────────────────────────────────────────────────────────
# Un même protocole peut maintenant être simulé sur plusieurs morphologies de
# tête. Chaque entrée précise où trouver le dossier m2m (nécessaire pour la
# transformation espace natif → MNI) et le dossier de résultats associé à
# cette tête.
#
# IMPORTANT : adapte les chemins ci-dessous (noms de dossiers) à ta structure
# réelle sur disque pour smoker_m / smoker_f.
HEAD_MODELS = {
    "ernie": {
        "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_ernie",
        "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_2",
    },
    "ernie_big": {
            "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_erniebig",
            "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_big",
        },
    "ernie_new": {
                "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_ernie_new",
                "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_new",
            },
    "ernie_new_2": {
                "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_ernie_new_2",
                "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_new_2",
            },
    "ernie_small": {
                "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_ernie_small",
                "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_small",
            },
    "coils_setups": {
            "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_ernie",
            "results_dir":  PROJECT_ROOT / "results" / "results_tms_coils_setups",
        },
    "coils_setups_M": {
            "subject_path": PROJECT_ROOT / "data" / "Smoker_patient_M" / "m2m_smoker_men",
            "results_dir":  PROJECT_ROOT / "results" / "results_tms_coils_setups_M",
        },
    "coils_setups_F": {
            "subject_path": PROJECT_ROOT / "data" / "Smoker_patient_F" / "m2m_smoker_women",
            "results_dir":  PROJECT_ROOT / "results" / "results_tms_coils_setups_F",
        },
    "smoker_m": {
        "subject_path": PROJECT_ROOT / "data" / "Smoker_patient_M" / "m2m_smoker_men",
        "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_smoker_M",
    },
    "smoker_f": {
        "subject_path": PROJECT_ROOT / "data" / "Smoker_patient_F" / "m2m_smoker_women",
        "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_smoker_F",
    },
}


class ProtocoleAnalysis:
    """
    Charge et analyse le champ électrique d'un protocole TMS simulé.
    Associe les tétraèdres du .msh aux régions de l'atlas.

    Un protocole est maintenant identifié par (study_id, head_model) : le
    même study_id peut exister sous plusieurs têtes (ernie, smoker_m,
    smoker_f, ...), chacune avec son propre dossier m2m et son propre
    dossier de résultats.
    """
    with open(ATLAS_LABELS_PATH, "r") as f:
        ATLAS_LABELS = {int(k): v for k, v in json.load(f).items()}

    def __init__(self, study_id, tissue_tags=None, HO_atlas=False,
                 head_model="ernie"):
        """
        head_model   : clé de HEAD_MODELS ("ernie", "smoker_m", "smoker_f").
        subject_path : surcharge manuelle du dossier m2m (prioritaire sur head_model)
        results_dir  : surcharge manuelle du dossier de résultats (prioritaire sur head_model)
        """
        self.study_id      = study_id
        self.tissue_tags   = tissue_tags if tissue_tags is not None else [1, 2]  # WM + GM par défaut
        self.HO_atlas      = HO_atlas   # Harvard-Oxford atlas
        self.head_model    = head_model

        if head_model not in HEAD_MODELS:
            raise ValueError(
                f"head_model '{head_model}' inconnu dans HEAD_MODELS. "
                f"Options : {list(HEAD_MODELS)}, ou fournir subject_path "
                f"et results_dir directement."
            )

        self.subject_path = HEAD_MODELS[head_model]["subject_path"]
        self.results_dir = HEAD_MODELS[head_model]["results_dir"]

        self.msh           = None       # Maillage chargé
        self.atlas         = None       # Atlas NIfTI
        self.tissue_mask   = None       # Masque du maillage pour les tissus selectionnés
        self.magnE         = None       # Liste des magnitudes du champ E associées aux tétraèdre
        self.vols          = None       # Liste des volumes associés aux tétraèdres
        self.region_labels = None       # Liste des régions de l'atlas associées aux tétraèdres

        self._load_msh()                # Charge le mesh de simulation de l'étude
        self._load_atlas()              # Charge l'atlas
        self._msh_in_tissue()           # Crée le masque des tissus, puis découpe le champs E et les volumes avec lui
        self._assign_regions()          # Associe les tétraèdres à une région anatomique de l'atlas

    def _load_msh(self):
        """Charge le .msh (dans le dossier de résultats propre à cette tête) et extrait magnE."""
        if self.msh is None:
            study_folder = self.results_dir / self.study_id

            fichiers_msh = [f for f in study_folder.glob("*.msh") if f.is_file()]

            if len(fichiers_msh) != 1:
                raise ValueError(
                    f"Nombre inattendu de fichiers .msh dans {self.study_id} "
                    f"(head_model={self.head_model}) : {len(fichiers_msh)}"
                )

            msh_file = fichiers_msh[0]
            self.msh = mesh_io.read_msh(msh_file)

    def _load_atlas(self):
        """Charge l'atlas .nii et prépare la matrice affine."""
        if self.HO_atlas:
            atlas_cortical = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr25-1mm')

            self.ATLAS_LABELS = {int(k): v for k, v in enumerate(atlas_cortical.labels)}
            self.atlas = atlas_cortical.maps
        else:
            self.atlas = nib.load(ATLAS_PATH)

    def _msh_in_tissue(self):
        """
        Retourne le mesh associé a un ou plusieurs tissus selon tissue_tags
        """
        elm_tags         = self.msh.elm.tag1 # liste des tags dans le mesh 1=WM, 2=GM 
        self.tissue_mask = np.isin(elm_tags, self.tissue_tags) # masque de selection pour les régions désirées
        self.magnE       = self.msh.field["magnE"][self.tissue_mask] # découpage du champs avec le masque
        self.vols        = self.msh.elements_volumes_and_areas()[self.tissue_mask] # Volume des régions désirées

    def _assign_regions(self):
        """
        Associe chaque tétraèdre (GM/WM) à une région de l'atlas
        via le barycentre des éléments transformé en espace MNI.

        La transformation natif → MNI utilise le dossier m2m propre à la
        tête sur laquelle CE protocole a été simulé (self.subject_path),
        et non plus systématiquement celui d'ernie.
        """
        # 1. barycentres des tétraèdres en espace natif
        centers = self.msh.elements_baricenters()[:]
        centers = centers[self.tissue_mask]

        # 2. transformation espace natif (tête spécifique) → espace MNI
        centers_mni = subject2mni_coords(centers, str(self.subject_path))

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

    def _get_field(self, name):
        for field in self.msh.elmdata:
            if field.field_name == name:
                return field
        return None

    def _weighted_percentile(self, values, weights, percentile):
        """
        Calcule le percentile pondéré de `values` avec les poids `weights`.

        percentile : float entre 0 et 100 (ex: 50 pour la médiane, 95 pour P95)
        """
        values = np.asarray(values)
        weights = np.asarray(weights)

        # Tri des valeurs
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]

        # Somme cumulée des poids
        cumulative = np.cumsum(weights)

        # Premier indice dont le poids cumulé atteint la fraction visée
        cutoff = percentile / 100 * weights.sum()
        idx = np.searchsorted(cumulative, cutoff)
        idx = min(idx, len(values) - 1)  # sécurité si cutoff == somme totale

        return values[idx]

    def weighted_percentile_by_region(self, percentile):
        """
        Retourne un dict {region_label: percentile pondéré de magnE}
        pour chaque région de l'atlas.
        """
        labels = self.region_labels
        E = self.magnE
        vols = self.vols

        region_values = {}

        for r in np.unique(labels):
            if r == 0:
                continue

            mask = labels == r
            if not np.any(mask):
                continue

            region_values[r] = self._weighted_percentile(
                E[mask],
                vols[mask],
                percentile
            )

        return region_values
    
    def global_reference(self, metric="mean", percentile=95, ref_metric="mean", threshold_pct=75):
        """
        Retourne la valeur de référence globale utilisée pour les seuils.
        """
        if metric == "mean":
            return np.average(self.magnE, weights=self.vols)

        elif metric == "median":
            return self._weighted_percentile(self.magnE, self.vols, 50)

        elif metric == "percentile":
            return self._weighted_percentile(self.magnE, self.vols, percentile)
        
        elif metric == "focality":
            return self._weighted_percentile(self.magnE, self.vols, percentile)/np.average(self.magnE, weights=self.vols)
        
        elif metric == "above_threshold":
            return self.global_fraction_above_threshold(threshold_pct=threshold_pct, metric = ref_metric, percentile = percentile)
        
        raise ValueError(f"The metric {metric} is not an option")

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
    
    def median_by_region(self):
        """
        Retourne un dict {region_label: médiane pondérée de magnE}
        """
        return self.weighted_percentile_by_region(50)

    def mean_by_region(self):
        """Retourne un dict {region_label: moyenne pondérée de magnE}."""
        
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

    def global_fraction_above_threshold(self,threshold_pct,metric="mean",percentile=95,by_volume=True,):
        """
        Retourne le % de volume (ou de tétraèdres) du cerveau entier
        dont magnE dépasse threshold_pct% de la référence globale.

        threshold_pct : float entre 0 et 100
        by_volume     : True  → % de volume (défaut)
                        False → % de tétraèdres
        metric        : 'mean' ou 'percentile'
        """

        E = self.magnE
        vols = self.vols

        global_ref = self.global_reference(metric=metric, percentile=percentile)

        threshold = (threshold_pct / 100) * global_ref

        above = E > threshold

        if by_volume:
            return np.sum(vols[above]) / np.sum(vols) * 100
        else:
            return np.mean(above) * 100
    
    def fraction_above_threshold(self, threshold_pct, ref_metric="mean", percentile=95, by_volume=True):
        """
        Pour chaque région, retourne le % de tétraèdres ou de volume dont magnE
        dépasse threshold_pct% de la référence globale GM+WM.
        
        threshold_pct : float entre 0 et 100 
        by_volume     : True  → % de volume (défaut)
                        False → % de tétraèdres
        metric        : 'mean' → moyenne globale pondérée (défaut)
                        'percentile'  → P(percentile) global
        """
        labels = self.region_labels
        E      = self.magnE

        global_ref = self.global_reference(metric=ref_metric, percentile=percentile)

        threshold = (threshold_pct / 100) * global_ref

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

    def fraction_below_threshold(self, threshold_pct, ref_metric="mean", percentile=95, by_volume=True):
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

        global_ref = self.global_reference(metric=ref_metric, percentile=percentile)
        threshold   = (threshold_pct / 100) * global_ref

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
    
    def stimulation_ratio(self, metric="mean", percentile=95):
        """
        Retourne un dict {region_label: ratio métrique_région / métrique_globale}
        
        metric : 'mean'       → moyenne pondérée
                 'median'     → médiane
                 'percentile' → P(percentile)
        """
        labels = self.region_labels
        E      = self.magnE

        # calcul de la référence globale
        global_ref = self.global_reference(metric, percentile)

        # calcul par région
        ratios = {}
        for r in np.unique(labels):
            if r == 0:
                continue
            mask = labels == r
            if not np.any(mask):
                continue

            if metric == "mean":
                region_ref = np.average(E[mask], weights=self.vols[mask])
            elif metric == "median":
                region_ref = np.median(E[mask])
            elif metric == "percentile":
                region_ref = np.percentile(E[mask], percentile)

            ratios[r] = region_ref / global_ref

        return ratios

    def rank_regions(self, n=5, metric="mean", percentile=95,
                 threshold_pct=50, by_volume=True, ascending=False, reference_metric="mean"):
        """
        Retourne les n régions classées selon la métrique choisie.

        metric        : 'mean'             → moyenne pondérée par volume
                        'percentile'       → valeur au P(percentile)
                        'focality'         → ratio P(percentile)/moyenne
                        'above_threshold'  → % de volume/tétraèdres au dessus du seuil
                        'below_threshold'  → % de volume/tétraèdres sous le seuil
                        'stimulation_ratio'→ ratio régions/globale selon un métrique

        percentile    : utilisé si metric='percentile' ou 'focality' (défaut 95)
        threshold_pct : utilisé si metric='above_threshold' ou 'below_threshold' (défaut 50)
        by_volume     : utilisé si metric='above_threshold' ou 'below_threshold' (défaut True)
        ascending     : False → plus stimulées en premier (défaut)
                        True  → moins stimulées en premier
        reference_metric : Métrique de référence pour above threshold et simulation ratio

        Retourne : liste de tuples (region_label, valeur) triée
        """

        if metric == "mean":
            scores = self.mean_by_region()

        elif metric == "percentile":
            scores = self.weighted_percentile_by_region(percentile)
        
        elif metric == "median":
            scores = self.median_by_region()

        elif metric == "focality":
            means       = self.mean_by_region()
            percentiles = self.weighted_percentile_by_region(percentile)
            scores = {
                r: percentiles[r] / means[r]
                for r in percentiles
                if r in means and means[r] > 0
            }

        elif metric == "above_threshold":
            scores = self.fraction_above_threshold(threshold_pct=threshold_pct, ref_metric = reference_metric, percentile=percentile, by_volume=by_volume)

        elif metric == "below_threshold":
            scores = self.fraction_below_threshold(threshold_pct, reference_metric, percentile, by_volume)
        
        elif metric == "stimulation_ratio":
            scores = self.stimulation_ratio(reference_metric, percentile)

        sorted_regions = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)

        return sorted_regions if n == "all" else sorted_regions[:n]

    def plot_region_distribution(self, kind="violin", n_regions=None, use_names=None,
                                  ascending=False, max_points_per_region=20000,
                                  percentile=95, figsize=(14, 10), color="lightsteelblue",
                                  show_global_mean=True, show_global_percentile=True,
                                  show_markers=True, show_outliers=False, random_state=0):
        """
        Affiche, pour CE protocole, la distribution du champ magnE par
        tétraèdre au sein de chaque région (box ou violin plot seaborn),
        pondérée par le volume des tétraèdres, avec les repères de moyenne
        et P(percentile) pondérés superposés (même style que
        CorrelationAnalysis.plot_region_distribution).

        Pondération : seaborn box/violin n'acceptent pas de poids par
        point — pour respecter quand même la pondération par volume sans
        dupliquer chaque tétraèdre proportionnellement (ce qui coûterait
        cher avec des millions d'éléments), un ré-échantillonnage pondéré
        À TAILLE FIXE est fait par région : max_points_per_region tirages
        AVEC remise, tirés avec probabilité proportionnelle au volume. Le
        coût total est donc borné à n_regions × max_points_per_region,
        indépendamment du nombre réel de tétraèdres dans le maillage.

        C'est une approximation Monte-Carlo (un peu de bruit d'échantillonnage
        sur la forme de la boîte/violon) — contrairement à la version
        précédente (bxp/gaussian_kde) qui était exacte. Pour compenser, les
        marqueurs de moyenne/P(percentile) affichés sont calculés
        directement sur les données pondérées complètes (via
        np.average/_weighted_percentile), pas sur le ré-échantillon : ils
        restent exacts même si la forme de la boîte/violon est approximative.
        Augmenter max_points_per_region réduit le bruit au prix du temps de
        calcul/mémoire.

        kind        : "box" ou "violin" (seaborn)
        n_regions   : None/"all" → toutes les régions
                      int → garde les n régions avec le plus de tétraèdres
        use_names   : None (défaut) → auto, comme histogram_metric() : noms
                      anatomiques si moins de 50 régions affichées, sinon
                      numéros d'atlas (pour rester lisible)
                      True/False → force noms / numéros, quel que soit n_regions
        ascending   : True (défaut) → régions triées par moyenne pondérée croissante
                      False → décroissante
        max_points_per_region : taille du ré-échantillon pondéré par région
        percentile  : percentile affiché en marqueur/ligne (défaut 95)
        show_global_mean : superpose la moyenne globale pondérée du protocole
                      (ligne noire pointillée, même couleur que les losanges
                      de moyenne par région)
        show_global_percentile : superpose le P(percentile) global pondéré du
                      protocole (ligne rouge pointillée, même couleur que les
                      triangles de P(percentile) par région) — désactivé par
                      défaut
        show_markers : superpose la moyenne et le P(percentile) pondérés de
                      chaque région (losange noir / triangle rouge)
        show_outliers : (kind="box" uniquement) affiche les points aberrants
                      (fliers) au-delà des moustaches. False les masque —
                      utile visuellement avec beaucoup de régions/points.
                      Sans effet sur kind="violin".

        Retourne la liste des region_id affichées, dans l'ordre du plot.
        """
        if kind not in ("box", "violin"):
            raise ValueError(f"kind '{kind}' invalide. Choisir 'box' ou 'violin'")

        labels = self.region_labels
        E = self.magnE
        vols = self.vols

        regions = [r for r in np.unique(labels) if r != 0]

        if n_regions not in (None, "all"):
            counts = {r: int(np.sum(labels == r)) for r in regions}
            regions = sorted(regions, key=lambda r: counts[r], reverse=True)[:n_regions]

        # stats exactes (pondérées), calculées sur les données complètes —
        # utilisées à la fois pour l'ordre et pour les marqueurs
        region_means = {r: np.average(E[labels == r], weights=vols[labels == r]) for r in regions}
        region_pct   = {r: self._weighted_percentile(E[labels == r], vols[labels == r], percentile)
                         for r in regions}

        regions = sorted(regions, key=lambda r: region_means[r], reverse=not ascending)

        # auto naming, same rule as histogram_metric(): names when the
        # displayed set is small enough to stay readable, numbers otherwise
        effective_use_names = (len(regions) < 50) if use_names is None else use_names

        rng = np.random.default_rng(random_state)
        records = []

        for r in regions:
            mask = labels == r
            vals = E[mask]
            w = vols[mask]
            p = w / w.sum()
            idx = rng.choice(len(vals), size=max_points_per_region, replace=True, p=p)
            records.extend({"region": r, "value": v} for v in vals[idx])

        df = pd.DataFrame(records)

        labels_display = (
            self.get_labels_for_regions(regions, use_names=True) if effective_use_names
            else [str(r) for r in regions]
        )

        fig, ax = plt.subplots(figsize=figsize)

        if kind == "box":
            sns.boxplot(data=df, x="region", y="value", order=regions, color=color,
                        showfliers=show_outliers, ax=ax)
        else:
            sns.violinplot(data=df, x="region", y="value", order=regions, color=color,
                            inner="quartile", ax=ax)

        if show_markers:
            for i, r in enumerate(regions):
                ax.scatter(i, region_means[r], color="black", marker="D", s=50,
                           zorder=5, label="Weighted mean" if i == 0 else None)
                ax.scatter(i, region_pct[r], color="crimson", marker="^", s=60,
                           zorder=5, label=f"Weighted P{percentile}" if i == 0 else None)

        if show_global_mean:
            global_mean = self.global_reference(metric="mean")
            ax.axhline(global_mean, color="black", linestyle="--",
                       linewidth=1.5, label=f"Global mean ({global_mean:.4f})")

        if show_global_percentile:
            global_pct = self.global_reference(metric="percentile", percentile=percentile)
            ax.axhline(global_pct, color="crimson", linestyle="--",
                       linewidth=1.5, label=f"Global P{percentile} ({global_pct:.4f})")

        if show_markers or show_global_mean or show_global_percentile:
            ax.legend()

        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(labels_display, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Region")
        ax.set_ylabel("magnE per tetrahedron (volume-weighted)")
        order_label = "ascending" if ascending else "descending"
        ax.set_title(f"{kind.capitalize()} plot — magnE per region ({self.study_id}, {order_label})")

        plt.tight_layout()
        plt.show()

        return regions