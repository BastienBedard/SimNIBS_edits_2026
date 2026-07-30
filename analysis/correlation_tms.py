import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    Charge les protocoles via ProtocoleAnalysis et compare leurs scores par région.
    """

    def __init__(self, study_ids=None, HO_atlas=False):
        """
        study_ids : liste de study_id à charger
                    None → prend tous les dossiers dans RESULTS_DIR
        HO_atlas  : passé tel quel à ProtocoleAnalysis
        """
        if study_ids is None:
            study_ids = [
                d.name for d in RESULTS_DIR.iterdir()
                if d.is_dir()
            ]
            print(f"  {len(study_ids)} protocoles trouvés automatiquement")

        self.HO_atlas  = HO_atlas
        self.study_ids = study_ids

        # {study_id: objet ProtocoleAnalysis}
        self.protocols = {}

        self._load_all()

    def _load_all(self):
        """
        Charge tous les protocoles (I/O). Ne fait aucun calcul de score —
        ça se fait à la demande dans compute_scores(), pour permettre de
        tester plusieurs métriques sans recharger les données.
        """
        for i, study_id in enumerate(self.study_ids):
            print(f"  Chargement -{i}/{len(self.study_ids)}- {study_id}")
            try:
                self.protocols[study_id] = ProtocoleAnalysis(
                    study_id=study_id, HO_atlas=self.HO_atlas
                )
            except Exception as e:
                print(f"  [SKIP] {study_id} — erreur : {e}")

        # mise à jour de study_ids pour exclure les protocoles en erreur
        self.study_ids = list(self.protocols.keys())

        print(f"\n  {len(self.study_ids)} protocoles chargés")

    def compute_scores(self, metric="median", percentile=95, n_regions="all"):
        """
        Calcule les scores par région pour tous les protocoles chargés.

        metric     : métrique utilisée pour le ranking
                     'mean'       → moyenne pondérée par volume
                     'median'     → médiane
                     'percentile' → P(percentile)
        percentile : utilisé si metric='percentile' (défaut 95)
        n_regions  : "all" → toutes les régions
                     int   → garde les n régions avec le score moyen
                             le plus élevé sur tous les protocoles

        Retourne un DataFrame (protocoles en lignes, régions en colonnes).
        Les régions absentes pour un protocole donné sont NaN.
        """
        all_scores = {}

        for study_id, p in self.protocols.items():
            if metric == "mean":
                scores = p.mean_by_region()
            elif metric == "median":
                scores = p.median_by_region()
            elif metric == "percentile":
                scores = p.weighted_percentile_by_region(percentile)
            else:
                raise ValueError(f"metric '{metric}' invalide. "
                                 f"Choisir : 'mean', 'median', 'percentile'")
            all_scores[study_id] = scores

        # protocoles en lignes, régions en colonnes — alignement automatique
        df = pd.DataFrame(all_scores).T
        print(f"  {df.shape[1]} régions utilisées")

        if n_regions != "all":
            top_regions = df.mean(axis=0).sort_values(ascending=False).index[:n_regions]
            df = df[top_regions]

        return df

    def spearman_matrix(self, metric="median", percentile=95, study_ids=None, n_regions=None):
        """
        Calcule la matrice de corrélation de Spearman entre protocoles.

        metric, percentile : voir compute_scores()
        study_ids : liste de study_id à inclure — None → tous
        n_regions : int → garde les n régions avec le score moyen
                    le plus élevé — None → toutes
        """
        df = self.compute_scores(
            metric=metric, percentile=percentile,
            n_regions=n_regions if n_regions is not None else "all"
        )

        if study_ids is not None:
            missing = set(study_ids) - set(df.index)
            if missing:
                print(f"Les protocoles suivants n'existent pas dans le dossier: {sorted(missing)}")
            df = df.loc[df.index.intersection(study_ids)]

        return df.T.corr(method="spearman")

    def plot_heatmap(self, metric="median", percentile=95, study_ids=None, n_regions=None, figsize=(14, 12)):
        """
        Affiche la heatmap de corrélation de Spearman.
        """
        corr_matrix = self.spearman_matrix(
            metric=metric, percentile=percentile,
            study_ids=study_ids, n_regions=n_regions
        )

        n_protocols  = len(corr_matrix)
        title_suffix = f"{n_regions} regions" if n_regions else "all regions"

        g = sns.clustermap(corr_matrix,
                    figsize=figsize,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=0.5,
                    annot_kws={"size": 8})

        g.fig.suptitle(f"Spearman Correlation — {metric} | "
                    f"{n_protocols} protocols | {title_suffix}", fontsize=12, y=1.02)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

    def plot_histogram(self, metric, percentile=95, ref_metric="percentile",
                        threshold_pct=75, study_ids=None, figsize=(10, 8),
                        color="steelblue", title=None, xlabel="Protocole",
                        ylabel=None, ascending=False):
        """
        Méthode commune pour les histogrammes de comparaison entre protocoles.
        Calcule un score par protocole via p.global_reference() et l'affiche
        en bar chart vertical, dans le même style que les histogrammes de
        ranking par région.
        """
        protocols = (
            self.protocols if study_ids is None
            else {sid: self.protocols[sid] for sid in study_ids if sid in self.protocols}
        )

        scores = {
            study_id: p.global_reference(
                metric=metric, percentile=percentile,
                ref_metric=ref_metric, threshold_pct=threshold_pct,
            )
            for study_id, p in protocols.items()
        }

        s = pd.Series(scores).sort_values(ascending=ascending)
        labels, values = s.index.tolist(), s.values

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(labels, values, color=color)

        if len(labels) < 50:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel is not None else "Field value")

        title_sup = f"Volume above {threshold_pct}% of global " if metric == "above_threshold" else ""
        title_sup = f"Volume below {threshold_pct}% of global " if metric == "below_threshold" else title_sup
        title_sup = "Stimulation ratio in " if metric == "stimulation_ratio" else title_sup
        title_sup = "Mean " if metric == "mean" else title_sup
        title_sup = f"P{percentile} " if metric == "percentile" else title_sup

        ax.set_title(title) if title is not None else \
            ax.set_title(f"{title_sup}Field Value — {len(labels)} protocols")

        plt.tight_layout()
        plt.show()


    def plot_focality_histogram(self, percentile=95, ref_metric="percentile",
                                threshold_pct=75, study_ids=None, figsize=(10, 8)):
        """Histogramme des protocoles triés selon leur score de focalité."""
        self.plot_histogram(
            metric="above_threshold", percentile=percentile, ref_metric=ref_metric,
            threshold_pct=threshold_pct, study_ids=study_ids, figsize=figsize,
            ylabel="Field focality score",
        )


    def plot_mean_histogram(self, study_ids=None, figsize=(10, 8)):
        """Histogramme des protocoles comparés sur la moyenne du champ."""
        self.plot_histogram(
            metric="mean", study_ids=study_ids, figsize=figsize,
            ylabel="Mean field value",
        )


    def plot_p95_histogram(self, percentile=95, study_ids=None, figsize=(10, 8)):
        """Histogramme des protocoles comparés sur le p95 du champ."""
        self.plot_histogram(
            metric="percentile", percentile=percentile, study_ids=study_ids,
            figsize=figsize, ylabel=f"P{percentile} field value",
        )