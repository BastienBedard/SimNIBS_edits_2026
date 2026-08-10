import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.stats import spearmanr
from analyse_TMS import ProtocoleAnalysis

# ─── CHEMINS ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent

HEAD_MODELS = {
    "ernie": {
        "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_ernie",
        "results_dir":  PROJECT_ROOT / "results" / "results_tms_MA_2",
    },
    "coils_setups": {
            "subject_path": PROJECT_ROOT / "data" / "ernie" / "m2m_ernie",
            "results_dir":  PROJECT_ROOT / "results" / "results_tms_coils_setups",
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

# ─── CLASSE ──────────────────────────────────────────────────────────────────

class CorrelationAnalysis:
    """
    Analyse de corrélation entre plusieurs protocoles TMS.
    Charge les protocoles via ProtocoleAnalysis et compare leurs scores par région.
    """

    def __init__(self, study_ids=None, HO_atlas=False, head_model="ernie"):
        """
        study_ids : liste de study_id à charger
                    None → prend tous les dossiers dans RESULTS_DIR
        HO_atlas  : passé tel quel à ProtocoleAnalysis
        head_model   : clé de HEAD_MODELS ("ernie", "smoker_m", "smoker_f").
        """
        self.head_model = head_model
        
        if study_ids is None:
            study_ids = [
                d.name for d in HEAD_MODELS[head_model]["results_dir"].iterdir()
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
                    study_id=study_id, HO_atlas=self.HO_atlas,
                    head_model=self.head_model
                )
            except Exception as e:
                print(f"  [SKIP] {study_id} — erreur : {e}")

        # mise à jour de study_ids pour exclure les protocoles en erreur
        self.study_ids = list(self.protocols.keys())

        print(f"\n  {len(self.study_ids)} protocoles chargés")

    def compute_scores(self, metric="median", percentile=95, n_regions="all",
                        reference_metric="mean"):
        """
        Calcule les scores par région pour tous les protocoles chargés.

        metric     : métrique utilisée pour le ranking
                     'mean'              → moyenne pondérée par volume
                     'median'            → médiane
                     'percentile'        → P(percentile)
                     'stimulation_ratio' → ratio région/référence globale
                                           (déjà normalisé, cf. reference_metric)
        percentile : utilisé si metric='percentile', ou si metric='stimulation_ratio'
                     et reference_metric='percentile'
        reference_metric : métrique de référence utilisée uniquement si
                     metric='stimulation_ratio' — passée à p.stimulation_ratio()
                     ('mean', 'median', ou 'percentile')
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
            elif metric == "stimulation_ratio":
                scores = p.stimulation_ratio(metric=reference_metric, percentile=percentile)
            else:
                raise ValueError(f"metric '{metric}' invalide. "
                                 f"Choisir : 'mean', 'median', 'percentile', 'stimulation_ratio'")
            all_scores[study_id] = scores

        # protocoles en lignes, régions en colonnes — alignement automatique
        df = pd.DataFrame(all_scores).T
        print(f"  {df.shape[1]} régions utilisées")

        if n_regions != "all":
            top_regions = df.mean(axis=0).sort_values(ascending=False).index[:n_regions]
            df = df[top_regions]

        return df

    
    def ranking_matrix(self, metric="median", percentile=95, n_regions="all",
                        reference_metric="mean"):
        """
        Construit une matrice où chaque ligne correspond à un protocole et
        chaque colonne à une position de rang (Rank 1, Rank 2, ...).

        Les valeurs sont les numéros de région (labels de l'atlas), triés par
        score décroissant : Rank 1 = région avec le score le plus élevé pour
        ce protocole.

        metric, percentile, n_regions, reference_metric : voir compute_scores()

        Retourne un DataFrame (protocoles en lignes, rangs en colonnes).
        Si une région est NaN pour un protocole (absente), elle est ignorée
        pour ce protocole et son ranking sera plus court que les autres —
        les colonnes de rang élevé seront alors NaN pour cette ligne.
        """
        df = self.compute_scores(metric=metric, percentile=percentile,
                                  n_regions=n_regions, reference_metric=reference_metric)

        ranked = {
            study_id: row.dropna().sort_values(ascending=False).index.tolist()
            for study_id, row in df.iterrows()
        }

        rank_df = pd.DataFrame(ranked).T  # aligne automatiquement, NaN si manquant
        rank_df.columns = [f"Rank {i + 1}" for i in range(rank_df.shape[1])]

        return rank_df

    def plot_ranking_matrix(self, metric="median", percentile=95, n_top=20,
                             show_names=False, figsize=None, cmap=None,
                             annot_fontsize=8, color_by="region",
                             reference_metric="mean"):
        """
        Affiche la ranking_matrix sous forme de heatmap : chaque ligne est un
        protocole, chaque colonne une position de rang, et chaque cellule est
        annotée avec la région classée à ce rang pour ce protocole.

        metric, percentile, reference_metric : voir compute_scores(). Si
                      metric='stimulation_ratio', reference_metric contrôle
                      la métrique de référence utilisée (défaut 'mean').
        n_top       : nombre de positions de rang à afficher (par défaut 20,
                      pour rester lisible). None ou "all" → toutes.
        show_names  : False (défaut) → les cellules affichent le numéro de
                      région (label de l'atlas) suivi de sa valeur, ex.
                      "101 : 2.03".
                      True  → les cellules affichent le nom anatomique de la
                      région à la place. Les noms viennent directement de
                      ATLAS_LABELS d'un des ProtocoleAnalysis déjà chargés
                      (pas besoin de recharger le JSON séparément).
        color_by    : "region" (défaut) → couleur catégorielle stable par
                      région (même région = même couleur partout dans la
                      heatmap). Une palette est générée dynamiquement avec
                      autant de couleurs distinctes que de régions uniques
                      affichées, pour éviter que plusieurs régions ne
                      partagent la même couleur (limite du "tab20" à 20
                      couleurs).
                      "value"  → couleur continue basée sur la valeur du
                      score de la région (colorbar affichée).
        cmap        : colormap seaborn/matplotlib. Si None (défaut), une
                      palette générée dynamiquement est utilisée pour
                      color_by="region" (voir ci-dessus) et "viridis" pour
                      color_by="value". Ignoré pour color_by="region" si une
                      valeur est passée explicitement, sauf si elle supporte
                      nativement autant de couleurs que de régions uniques.

        Retourne le DataFrame de ranking (non tronqué par n_top, valeurs
        toujours en numéros de région quel que soit show_names).
        """
        if color_by not in ("region", "value"):
            raise ValueError(f"color_by '{color_by}' invalide. Choisir 'region' ou 'value'")

        rank_df = self.ranking_matrix(metric=metric, percentile=percentile,
                                       n_regions="all", reference_metric=reference_metric)
        score_df = self.compute_scores(metric=metric, percentile=percentile,
                                        n_regions="all", reference_metric=reference_metric)

        n_top = None if n_top in (None, "all") else n_top
        plot_df = rank_df if n_top is None else rank_df.iloc[:, :n_top]

        # valeur du score associée à chaque cellule (même région, même rang,
        # même protocole que plot_df), utilisée pour l'annotation et/ou la
        # couleur selon color_by
        value_df = pd.DataFrame(
            {
                col: [
                    score_df.loc[study_id, region] if pd.notna(region) else np.nan
                    for study_id, region in zip(plot_df.index, plot_df[col])
                ]
                for col in plot_df.columns
            },
            index=plot_df.index,
        )

        labels_map = next(iter(self.protocols.values())).ATLAS_LABELS if self.protocols else {}

        heatmap_vmin = heatmap_vmax = None

        if color_by == "region":
            # couleur catégorielle stable par région : la même région garde
            # la même couleur quel que soit son rang ou son protocole
            codes_flat, uniques = pd.factorize(plot_df.values.ravel())
            color_matrix = pd.DataFrame(
                codes_flat.reshape(plot_df.shape),
                index=plot_df.index, columns=plot_df.columns,
            )

            n_unique = len(uniques)

            # "tab20" (ou toute palette matplotlib fixe) ne fournit que
            # 20 couleurs distinctes : au-delà, les codes bouclent et
            # plusieurs régions finissent par partager une couleur. On
            # génère donc une palette avec exactement n_unique couleurs
            # (husl répartit les teintes uniformément sur le cercle
            # chromatique, ce qui reste discriminable même pour un grand N).
            if cmap is None:
                palette = sns.color_palette("husl", n_unique)
                cmap = ListedColormap(palette)

            # borne la normalisation des couleurs pour que chaque code
            # entier [0, n_unique - 1] tombe exactement sur une couleur de
            # la palette, plutôt que d'être interpolé/re-normalisé par
            # seaborn sur la plage par défaut du cmap
            heatmap_vmin, heatmap_vmax = -0.5, n_unique - 0.5
        else:
            color_matrix = value_df

        if show_names:
            annot_df = plot_df.map(
                lambda c: labels_map.get(int(c), "unknown") if pd.notna(c) else ""
            )
        else:
            annot_df = pd.DataFrame(
                {
                    col: [
                        f"{int(region)} : {value:.2f}" if pd.notna(region) else ""
                        for region, value in zip(plot_df[col], value_df[col])
                    ]
                    for col in plot_df.columns
                },
                index=plot_df.index,
            )

        if figsize is None:
            width_factor = 1.4 if show_names else 0.9
            figsize = (max(8, plot_df.shape[1] * width_factor), max(4, plot_df.shape[0] * 0.5))

        if cmap is None:
            cmap = "viridis" if color_by == "value" else "tab20"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            color_matrix,
            annot=annot_df,
            fmt="",
            cmap=cmap,
            cbar=(color_by == "value"),
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": annot_fontsize},
            ax=ax,
        )

        metric_label = f"{metric} (ref={reference_metric})" if metric == "stimulation_ratio" else metric
        title_suffix = f" (top {n_top} rangs)" if n_top else ""
        ax.set_title(f"Region ranking per protocol — {metric_label}{title_suffix}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Protocol")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()

        # légende utile seulement si les numéros (pas les noms) sont affichés
        # et que la couleur encode la région (sinon la colorbar suffit)
        if not show_names and color_by == "region" and self.protocols:
            print("Region legend:")
            for code in uniques:
                if pd.isna(code):
                    continue
                print(f"  {int(code)}: {labels_map.get(int(code), 'unknown')}")

        return rank_df

    def compute_stimulation_ratios(self, metric="mean", percentile=95, study_ids=None):
        """
        Calcule le stimulation_ratio (région / référence globale) pour chaque
        protocole et chaque région.

        Contrairement à compute_scores() qui retourne des valeurs brutes de
        champ, cette méthode retourne des valeurs déjà normalisées — nécessaire
        pour comparer des protocoles avec des dI/dt différents.

        metric     : métrique de référence passée à stimulation_ratio()
                    'mean', 'median', ou 'percentile'
        percentile : utilisé si metric='percentile'
        study_ids  : liste de study_id à inclure — None → tous

        Retourne un DataFrame (protocoles en lignes, régions en colonnes).
        """
        protocols = (
            self.protocols if study_ids is None
            else {sid: self.protocols[sid] for sid in study_ids if sid in self.protocols}
        )

        all_ratios = {
            study_id: p.stimulation_ratio(metric=metric, percentile=percentile)
            for study_id, p in protocols.items()
        }

        df = pd.DataFrame(all_ratios).T
        print(f"  {df.shape[1]} régions utilisées")

        return df

    def _pairwise_ratio_error(self, ratio_df):
        """
        Pour chaque paire de protocoles, calcule le RMSE et le MAE des
        différences de stimulation_ratio, région par région, sur les régions
        présentes (non-NaN) dans les deux protocoles.

        ratio_df : DataFrame de stimulation_ratio (protocoles en lignes,
                régions en colonnes), typiquement compute_stimulation_ratios()

        Retourne trois DataFrames alignés (protocoles x protocoles) :
        rmse_matrix, mae_matrix, n_matrix (nombre de régions comparées par paire).
        """
        protocols = ratio_df.index
        rmse_matrix = pd.DataFrame(index=protocols, columns=protocols, dtype=float)
        mae_matrix  = pd.DataFrame(index=protocols, columns=protocols, dtype=float)
        n_matrix    = pd.DataFrame(index=protocols, columns=protocols, dtype=float)

        for i in protocols:
            for j in protocols:
                common = ratio_df.loc[i].notna() & ratio_df.loc[j].notna()
                n_common = int(common.sum())
                n_matrix.loc[i, j] = n_common

                if n_common == 0:
                    rmse_matrix.loc[i, j] = np.nan
                    mae_matrix.loc[i, j] = np.nan
                    continue

                diff = ratio_df.loc[i, common] - ratio_df.loc[j, common]

                rmse_matrix.loc[i, j] = np.sqrt(np.mean(diff ** 2))
                mae_matrix.loc[i, j]  = np.mean(diff.abs())

        return rmse_matrix, mae_matrix, n_matrix

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

    def _topn_union_stats(self, df, n_regions):
        """
        Pour chaque paire de protocoles, calcule rho de Spearman et la distance
        moyenne de rang sur l'UNION de leurs top-n régions respectives (et non
        sur un top-n global commun à tous les protocoles).

        Concrètement : pour les protocoles A et B, on prend les n régions les
        mieux classées de A, les n régions les mieux classées de B, on fait
        l'union des deux ensembles (donc entre n et 2n régions selon leur
        recouvrement), puis on calcule rho et la distance de rang sur ces
        régions communes (en excluant celles qui seraient NaN pour l'un des
        deux protocoles).

        df : DataFrame scores (protocoles en lignes, régions en colonnes),
            typiquement compute_scores(n_regions="all")
        n_regions : taille du top-n par protocole, avant union

        Retourne trois DataFrames alignés (protocoles x protocoles) :
        rho_matrix, dist_matrix, n_matrix (nombre de régions comparées par paire).
        """
        protocols = df.index
        rho_matrix  = pd.DataFrame(index=protocols, columns=protocols, dtype=float)
        dist_matrix = pd.DataFrame(index=protocols, columns=protocols, dtype=float)
        n_matrix    = pd.DataFrame(index=protocols, columns=protocols, dtype=float)

        top_regions = {
            study_id: set(df.loc[study_id].dropna().sort_values(ascending=False).index[:n_regions])
            for study_id in protocols
        }

        for i in protocols:
            for j in protocols:
                union_regions = top_regions[i] | top_regions[j]
                common = [
                    r for r in union_regions
                    if pd.notna(df.loc[i, r]) and pd.notna(df.loc[j, r])
                ]

                n_matrix.loc[i, j] = len(common)

                if len(common) < 2:
                    rho_matrix.loc[i, j] = np.nan
                    dist_matrix.loc[i, j] = np.nan
                    continue

                vals_i = df.loc[i, common]
                vals_j = df.loc[j, common]

                rho, _ = spearmanr(vals_i, vals_j)
                rho_matrix.loc[i, j] = rho

                rank_i = vals_i.rank(ascending=False)
                rank_j = vals_j.rank(ascending=False)
                dist_matrix.loc[i, j] = (rank_i - rank_j).abs().mean()

        return rho_matrix, dist_matrix, n_matrix

    def rank_distance_matrix(self, metric="median", percentile=95, study_ids=None):
        """
        Calcule la distance moyenne de rang (Spearman footrule moyenné) entre
        protocoles.

        Pour chaque protocole, les régions sont classées par score (rang 1 =
        score le plus élevé). Pour chaque paire de protocoles, la distance est
        la moyenne de |rang_A(région) - rang_B(région)| sur les régions
        présentes (non-NaN) dans les deux protocoles.

        Calculé sur les 139 régions (pas de troncature n_regions ici).

        metric, percentile : voir compute_scores()
        study_ids : liste de study_id à inclure — None → tous

        Retourne un DataFrame (protocoles en lignes/colonnes) de distances
        moyennes de rang.
        """
        df = self.compute_scores(metric=metric, percentile=percentile, n_regions="all")

        if study_ids is not None:
            missing = set(study_ids) - set(df.index)
            if missing:
                print(f"Les protocoles suivants n'existent pas dans le dossier: {sorted(missing)}")
            df = df.loc[df.index.intersection(study_ids)]

        # rang 1 = score le plus élevé, par protocole (ligne)
        rank_df = df.rank(axis=1, ascending=False, method="average")

        protocols = rank_df.index
        dist_matrix = pd.DataFrame(index=protocols, columns=protocols, dtype=float)

        for i in protocols:
            for j in protocols:
                common = rank_df.loc[i].notna() & rank_df.loc[j].notna()
                if not common.any():
                    dist_matrix.loc[i, j] = np.nan
                    continue
                dist_matrix.loc[i, j] = (
                    rank_df.loc[i, common] - rank_df.loc[j, common]
                ).abs().mean()

        return dist_matrix

    def plot_error_heatmap(self, metric="mean", percentile=95, study_ids=None,
                        figsize=(14, 12), cmap="YlOrRd", mae_tiny_threshold=0.01):
        """
        Affiche une heatmap (clustermap) du RMSE, du MAE, et de leur ratio entre
        protocoles, calculés sur le stimulation_ratio par région (donc déjà
        normalisé, contrairement à plot_heatmap() qui compare des rangs sur des
        valeurs brutes de champ).

        Chaque cellule est annotée "RMSE : MAE : ratio". Le clustering et la
        couleur restent basés sur RMSE (magnitude) — le ratio est fourni comme
        contexte additionnel, pas comme signal principal, car il devient instable
        (bruit) quand RMSE et MAE sont tous deux proches de zéro.

        Interprétation :
        RMSE >= MAE toujours (identité mathématique).
        - ratio proche de 1        → différences réparties uniformément
                                        entre régions.
        - ratio nettement > 1      → une ou quelques régions dominent
                                        l'erreur, à inspecter individuellement.
        - RMSE < MAE               → ne devrait jamais arriver, signale un bug.

        metric, percentile   : passés à compute_stimulation_ratios()
        study_ids            : liste de study_id à inclure — None → tous
        mae_tiny_threshold    : si MAE < ce seuil, le ratio est affiché avec
                                1 seule décimale plutôt que 2, car il devient
                                peu fiable (bruit numérique) quand les erreurs
                                sont négligeables.
        """
        ratio_df = self.compute_stimulation_ratios(
            metric=metric, percentile=percentile, study_ids=study_ids
        )

        rmse_matrix, mae_matrix, n_matrix = self._pairwise_ratio_error(ratio_df)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_matrix = rmse_matrix / mae_matrix

        def _format_cell(r, m, ratio):
            if pd.isna(r) or pd.isna(m):
                return ""
            if pd.isna(ratio):
                ratio_str = "–"
            elif m < mae_tiny_threshold:
                ratio_str = f"{ratio:.1f}"
            else:
                ratio_str = f"{ratio:.2f}"
            return f"{r:.2f} : {m:.2f}\nratio {ratio_str}"

        annot_df = pd.DataFrame(
            {
                col: [
                    _format_cell(rmse_matrix.loc[idx, col], mae_matrix.loc[idx, col],
                                ratio_matrix.loc[idx, col])
                    for idx in rmse_matrix.index
                ]
                for col in rmse_matrix.columns
            },
            index=rmse_matrix.index,
        )

        n_protocols = len(rmse_matrix)
        metric_label = f"{metric} (P{percentile})" if metric == "percentile" else metric

        # RMSE matrix is already a distance matrix (symmetric, zero diagonal) —
        # build linkage directly from it instead of letting clustermap treat
        # rows as feature vectors and re-derive a second distance from them.
        condensed = squareform(rmse_matrix.values, checks=False)
        row_linkage = linkage(condensed, method="average")

        g = sns.clustermap(
            rmse_matrix,
            row_linkage=row_linkage,
            col_linkage=row_linkage,
            figsize=figsize,
            annot=annot_df,
            fmt="",
            cmap=cmap,
            vmin=0,
            linewidths=0.5,
            annot_kws={"size": 8},
            dendrogram_ratio=(0.01, 0.15),
            cbar_pos=(0.35, 0.92, 0.3, 0.03),
            cbar_kws={"orientation": "horizontal"},
        )
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)

        g.figure.suptitle(
            f"RMSE : MAE : ratio of stimulation ratio — {metric_label} | "
            f"{n_protocols} protocols",
            fontsize=12, y=1.02,
        )
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

        return rmse_matrix, mae_matrix, ratio_matrix

    def plot_heatmap(self, metric="mean", percentile=95, study_ids=None, n_regions=None, figsize=(14, 12)):
        """
        Affiche la heatmap (clustermap) de corrélation de Spearman, colorée et
        clusterisée par rho. Chaque cellule est annotée avec
        "rho : distance moyenne de rang". Si n_regions est un int, le nombre
        de régions comparées (N) est ajouté sur une ligne séparée, car il
        varie alors selon la paire de protocoles (voir _topn_union_stats).

        n_regions : None (défaut) → comparaison sur toutes les régions
                    communes aux deux protocoles (comportement global).
                    int → comparaison, pour chaque PAIRE de protocoles, sur
                    l'union de leurs top-n régions respectives.
        """
        df = self.compute_scores(metric=metric, percentile=percentile, n_regions="all")

        if study_ids is not None:
            missing = set(study_ids) - set(df.index)
            if missing:
                print(f"Les protocoles suivants n'existent pas dans le dossier: {sorted(missing)}")
            df = df.loc[df.index.intersection(study_ids)]

        if n_regions is None:
            corr_matrix = df.T.corr(method="spearman")
            dist_matrix = self.rank_distance_matrix(metric=metric, percentile=percentile, study_ids=study_ids)
            dist_matrix = dist_matrix.loc[corr_matrix.index, corr_matrix.columns]

            annot_df = corr_matrix.round(2).astype(str) + " : " + dist_matrix.round(1).astype(str)
            title_suffix = "all regions"
        else:
            corr_matrix, dist_matrix, n_matrix = self._topn_union_stats(df, n_regions)

            annot_df = (
                corr_matrix.round(2).astype(str) + " : " + dist_matrix.round(1).astype(str)
                + "\nN=" + n_matrix.astype(int).astype(str)
            )
            title_suffix = f"top {n_regions} union per pair"

        n_protocols = len(corr_matrix)
        metric_label = f"{metric} (P{percentile})" if metric == "percentile" else metric

        g = sns.clustermap(corr_matrix,
                    figsize=figsize,
                    annot=annot_df,
                    fmt="",
                    cmap="coolwarm",
                    vmin=-1, vmax=1,
                    linewidths=0.5,
                    annot_kws={"size": 8},
                    dendrogram_ratio=(0.01, 0.15),
                    cbar_pos=(0.35, 0.92, 0.3, 0.03),
                    cbar_kws={"orientation": "horizontal"})
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)

        g.figure.suptitle(f"Spearman rho : mean rank distance — {metric_label} | "
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