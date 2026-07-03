import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd
from pathlib import Path
from analyse_TMS import ProtocoleAnalysis

# ─── CHEMINS ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "results_tms_MA"

# ─── CLASSE ──────────────────────────────────────────────────────────────────

class CorrelationAnalysis:
    """
    Analyse de corrélation entre plusieurs protocoles TMS.
    Charge les protocoles via ProtocoleAnalysis et compare leurs rankings.
    """
    def __init__(self, study_ids=None, metric="median", percentile=95,
                 n_regions="all"):
        """
        study_ids  : liste de study_id à comparer
                     None → prend tous les dossiers dans RESULTS_DIR
        metric     : métrique pour le ranking
                     'mean'       → moyenne pondérée par volume
                     'median'     → médiane
                     'percentile' → P(percentile)
        percentile : utilisé si metric='percentile' (défaut 95)
        n_regions  : "all" → toutes les régions de l'atlas
                     int   → garde les n régions avec le score moyen
                             le plus élevé sur tous les protocoles
        """
        if study_ids is None:
            study_ids = [
                d.name for d in RESULTS_DIR.iterdir()
                if d.is_dir()
            ]
            print(f"  {len(study_ids)} protocoles trouvés automatiquement")

        self.study_ids  = study_ids
        self.metric     = metric
        self.percentile = percentile
        self.n_regions  = n_regions

        # {study_id: vecteur numpy de valeurs alignées sur self.region_ids}
        self.rankings   = {}
        # liste ordonnée des region_ids communs à tous les protocoles
        self.region_ids = None

        self._load_all()

    def _load_all(self):
        """
        Charge tous les protocoles et calcule leurs rankings.
        """
        all_scores = {}

        for i, study_id in enumerate(self.study_ids):
            print(f"  Chargement -{i}/{len(self.study_ids)}- {study_id}")
            try:
                p = ProtocoleAnalysis(study_id=study_id)

                if self.metric == "mean":
                    scores = p.mean_by_region()
                elif self.metric == "median":
                    scores = p.median_by_region()
                elif self.metric == "percentile":
                    scores = p.top_percentile_volume(self.percentile)
                else:
                    raise ValueError(f"metric '{self.metric}' invalide. "
                                     f"Choisir : 'mean', 'median', 'percentile'")

                all_scores[study_id] = scores

            except Exception as e:
                print(f"  [SKIP] {study_id} — erreur : {e}")

        # mise à jour de study_ids pour exclure les protocoles en erreur
        self.study_ids = list(all_scores.keys())

        # union de toutes les régions présentes dans tous les protocoles
        all_region_ids = set()
        for scores in all_scores.values():
            all_region_ids.update(scores.keys())
        self.region_ids = sorted(all_region_ids)

        # si n_regions est un int — garder les n régions avec le score
        # moyen le plus élevé sur tous les protocoles comme référence commune
        if self.n_regions != "all":
            avg_scores = {
                r: np.mean([all_scores[sid].get(r, 0) for sid in self.study_ids])
                for r in self.region_ids
            }
            self.region_ids = sorted(avg_scores,
                                     key=avg_scores.get,
                                     reverse=True)[:self.n_regions]

        # construire les vecteurs alignés sur les mêmes region_ids
        for study_id, scores in all_scores.items():
            self.rankings[study_id] = np.array([
                scores.get(r, 0) for r in self.region_ids
            ])

        print(f"\n  {len(self.study_ids)} protocoles chargés")
        print(f"  {len(self.region_ids)} régions utilisées")

    def spearman_matrix(self, study_ids=None, n_regions=None):
        """
        Calcule la matrice de corrélation de Spearman.
        
        study_ids : liste de study_id à inclure — None → tous
        n_regions : int → garde les n régions avec le score moyen
                    le plus élevé — None → toutes
        """
        # filtrage des study_ids
        ids = study_ids if study_ids is not None else self.study_ids
        ids = list(set(ids) & set(self.study_ids))
        if study_ids is not None and list(set(ids) ^ set(study_ids)):
            print(f"Les protocoles suivantes n'existe pas dans le dossier: {list(set(ids) ^ set(study_ids))}")

        # filtrage des régions
        if n_regions is not None:
            avg_scores = {
                r: np.mean([self.rankings[sid][i]
                            for sid in ids
                            if sid in self.rankings])
                for i, r in enumerate(self.region_ids)
            }
            region_idx = [
                self.region_ids.index(r)
                for r in sorted(avg_scores, key=avg_scores.get, reverse=True)[:n_regions]
            ]
        else:
            region_idx = list(range(len(self.region_ids)))

        # calcul de la matrice
        n      = len(ids)
        matrix = np.zeros((n, n))

        for i, sid_i in enumerate(ids):
            for j, sid_j in enumerate(ids):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    vec_i        = self.rankings[sid_i][region_idx]
                    vec_j        = self.rankings[sid_j][region_idx]
                    corr, _      = spearmanr(vec_i, vec_j)
                    matrix[i, j] = corr
                    matrix[j, i] = corr

        return pd.DataFrame(matrix, index=ids, columns=ids)


    def plot_heatmap(self, study_ids=None, n_regions=None, figsize=(14, 12)):
        """
        Affiche la heatmap de corrélation de Spearman.

        study_ids : liste de study_id à inclure — None → tous
        n_regions : int → garde les n régions les plus stimulées — None → toutes
        """
        corr_matrix = self.spearman_matrix(study_ids=study_ids, n_regions=n_regions)

        n_protocols = len(corr_matrix)
        title_suffix = f"{n_regions} regions" if n_regions else "all regions"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix,
                    ax=ax,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=0.5,
                    annot_kws={"size": 8})

        ax.set_title(f"Spearman Correlation — {self.metric} | "
                    f"{n_protocols} protocols | {title_suffix}", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()