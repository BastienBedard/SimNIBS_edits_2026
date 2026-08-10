import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from analyse_TMS import ProtocoleAnalysis, HEAD_MODELS

# ─── CLASSE ──────────────────────────────────────────────────────────────────

class HeadModelComparison:
    """
    Compare la distribution du champ E d'UN protocole simulé sur plusieurs
    morphologies de tête (head models). Objectif : voir si la morphologie
    change le ranking des régions les plus stimulées.

    Miroir de CorrelationAnalysis (correlation_tms.py), mais l'axe de
    comparaison est le head_model plutôt que le study_id — un seul
    protocole, plusieurs têtes.
    """

    def __init__(self, study_id, head_models=None, HO_atlas=False):
        """
        study_id    : identifiant du protocole à comparer entre les têtes
        head_models : liste de clés de HEAD_MODELS à comparer
                      None → toutes les têtes disponibles dans HEAD_MODELS
        HO_atlas    : passé tel quel à ProtocoleAnalysis
        """
        self.study_id    = study_id
        self.head_models = list(head_models) if head_models is not None else list(HEAD_MODELS)
        self.HO_atlas    = HO_atlas

        # {head_model: objet ProtocoleAnalysis}
        self.protocols = {}

        self._load_all()

    def _load_all(self):
        """
        Charge le même protocole sur chaque tête (I/O). Ne fait aucun calcul
        de score — ça se fait à la demande dans compute_scores(), pour
        permettre de tester plusieurs métriques sans recharger les données.
        """
        for i, head_model in enumerate(self.head_models):
            print(f"  Chargement -{i}/{len(self.head_models)}- {self.study_id} ({head_model})")
            try:
                self.protocols[head_model] = ProtocoleAnalysis(
                    study_id=self.study_id, HO_atlas=self.HO_atlas,
                    head_model=head_model
                )
            except Exception as e:
                print(f"  [SKIP] {head_model} — erreur : {e}")

        # mise à jour de head_models pour exclure les têtes en erreur
        self.head_models = list(self.protocols.keys())

        print(f"\n  {len(self.head_models)} têtes chargées pour {self.study_id}")

    def compute_scores(self, metric="median", percentile=95, n_regions="all"):
        """
        Calcule les scores par région pour chaque tête, pour ce protocole.

        Identique à CorrelationAnalysis.compute_scores, mais les lignes sont
        des head_model plutôt que des study_id.

        metric     : 'mean'       → moyenne pondérée par volume
                     'median'     → médiane
                     'percentile' → P(percentile)
        percentile : utilisé si metric='percentile' (défaut 95)
        n_regions  : "all" → toutes les régions
                     int   → garde les n régions avec le score moyen
                             le plus élevé sur toutes les têtes

        Retourne un DataFrame (têtes en lignes, régions en colonnes).
        """
        all_scores = {}

        for head_model, p in self.protocols.items():
            if metric == "mean":
                scores = p.mean_by_region()
            elif metric == "median":
                scores = p.median_by_region()
            elif metric == "percentile":
                scores = p.weighted_percentile_by_region(percentile)
            else:
                raise ValueError(f"metric '{metric}' invalide. "
                                 f"Choisir : 'mean', 'median', 'percentile'")
            all_scores[head_model] = scores

        # têtes en lignes, régions en colonnes — alignement automatique
        df = pd.DataFrame(all_scores).T
        print(f"  {df.shape[1]} régions utilisées")

        if n_regions != "all":
            top_regions = df.mean(axis=0).sort_values(ascending=False).index[:n_regions]
            df = df[top_regions]

        return df

    def ranking_matrix(self, metric="median", percentile=95, n_regions="all"):
        """
        Construit une matrice où chaque ligne correspond à une tête (head
        model) et chaque colonne à une position de rang (Rank 1, Rank 2, ...),
        pour CE protocole (self.study_id).

        Les valeurs sont les numéros de région (labels de l'atlas), triés par
        score décroissant : Rank 1 = région avec le score le plus élevé pour
        cette tête.

        metric, percentile, n_regions : voir compute_scores()

        Retourne un DataFrame (têtes en lignes, rangs en colonnes).
        Si une région est NaN pour une tête (absente), elle est ignorée pour
        cette tête et son ranking sera plus court que les autres — les
        colonnes de rang élevé seront alors NaN pour cette ligne.
        """
        df = self.compute_scores(metric=metric, percentile=percentile, n_regions=n_regions)

        ranked = {
            head_model: row.dropna().sort_values(ascending=False).index.tolist()
            for head_model, row in df.iterrows()
        }

        rank_df = pd.DataFrame(ranked).T  # aligne automatiquement, NaN si manquant
        rank_df.columns = [f"Rank {i + 1}" for i in range(rank_df.shape[1])]

        return rank_df

    def plot_ranking_matrix(self, metric="median", percentile=95, n_top=20,
                             show_names=False, figsize=None, cmap=None,
                             annot_fontsize=8, color_by="region"):
        """
        Affiche la ranking_matrix sous forme de heatmap : chaque ligne est
        une tête (head model), chaque colonne une position de rang, et
        chaque cellule est annotée avec la région classée à ce rang pour
        cette tête, pour CE protocole (self.study_id).

        Mêmes paramètres et même comportement que
        CorrelationAnalysis.plot_ranking_matrix — voir cette méthode pour le
        détail de show_names / color_by / cmap. Seul l'axe des lignes change :
        têtes au lieu de protocoles.

        Retourne le DataFrame de ranking (non tronqué par n_top, valeurs
        toujours en numéros de région quel que soit show_names).
        """
        if color_by not in ("region", "value"):
            raise ValueError(f"color_by '{color_by}' invalide. Choisir 'region' ou 'value'")

        rank_df = self.ranking_matrix(metric=metric, percentile=percentile, n_regions="all")
        score_df = self.compute_scores(metric=metric, percentile=percentile, n_regions="all")

        n_top = None if n_top in (None, "all") else n_top
        plot_df = rank_df if n_top is None else rank_df.iloc[:, :n_top]

        # valeur du score associée à chaque cellule (même région, même rang,
        # même tête que plot_df), utilisée pour l'annotation et/ou la
        # couleur selon color_by
        value_df = pd.DataFrame(
            {
                col: [
                    score_df.loc[head_model, region] if pd.notna(region) else np.nan
                    for head_model, region in zip(plot_df.index, plot_df[col])
                ]
                for col in plot_df.columns
            },
            index=plot_df.index,
        )

        labels_map = next(iter(self.protocols.values())).ATLAS_LABELS if self.protocols else {}

        if color_by == "region":
            # couleur catégorielle stable par région : la même région garde
            # la même couleur quel que soit son rang ou sa tête
            codes_flat, uniques = pd.factorize(plot_df.values.ravel())
            color_matrix = pd.DataFrame(
                codes_flat.reshape(plot_df.shape),
                index=plot_df.index, columns=plot_df.columns,
            )
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
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": annot_fontsize},
            ax=ax,
        )

        title_suffix = f" (top {n_top} rangs)" if n_top else ""
        ax.set_title(f"Region ranking per head model — {self.study_id} — {metric}{title_suffix}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Head model")
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

    def rank_distance_matrix(self, metric="median", percentile=95, head_models=None):
        """
        Calcule la distance moyenne de rang (Spearman footrule moyenné)
        entre têtes, pour CE protocole (self.study_id).

        Identique à CorrelationAnalysis.rank_distance_matrix, mais les
        lignes/colonnes sont des head_model plutôt que des study_id.

        Pour chaque tête, les régions sont classées par score (rang 1 =
        score le plus élevé). Pour chaque paire de têtes, la distance est la
        moyenne de |rang_A(région) - rang_B(région)| sur les régions
        présentes (non-NaN) dans les deux têtes.

        Calculé sur les 139 régions (pas de troncature n_regions ici).

        metric, percentile : voir compute_scores()
        head_models : liste de head_model à inclure — None → toutes les
                      têtes chargées

        Retourne un DataFrame (têtes en lignes/colonnes) de distances
        moyennes de rang.
        """
        df = self.compute_scores(metric=metric, percentile=percentile, n_regions="all")

        if head_models is not None:
            missing = set(head_models) - set(df.index)
            if missing:
                print(f"Les têtes suivantes n'existent pas parmi celles chargées : {sorted(missing)}")
            df = df.loc[df.index.intersection(head_models)]

        # rang 1 = score le plus élevé, par tête (ligne)
        rank_df = df.rank(axis=1, ascending=False, method="average")

        heads = rank_df.index
        dist_matrix = pd.DataFrame(index=heads, columns=heads, dtype=float)

        for i in heads:
            for j in heads:
                common = rank_df.loc[i].notna() & rank_df.loc[j].notna()
                if not common.any():
                    dist_matrix.loc[i, j] = np.nan
                    continue
                dist_matrix.loc[i, j] = (
                    rank_df.loc[i, common] - rank_df.loc[j, common]
                ).abs().mean()

        return dist_matrix

    def _topn_union_stats(self, df, n_regions):
        """
        Pour chaque paire de têtes, calcule rho de Spearman et la distance
        moyenne de rang sur l'UNION de leurs top-n régions respectives (et
        non sur un top-n global commun à toutes les têtes).

        Identique à CorrelationAnalysis._topn_union_stats, mais opère sur
        des head_model plutôt que des study_id.

        df : DataFrame scores (têtes en lignes, régions en colonnes),
            typiquement compute_scores(n_regions="all")
        n_regions : taille du top-n par tête, avant union

        Retourne trois DataFrames alignés (têtes x têtes) :
        rho_matrix, dist_matrix, n_matrix (nombre de régions comparées par paire).
        """
        heads = df.index
        rho_matrix  = pd.DataFrame(index=heads, columns=heads, dtype=float)
        dist_matrix = pd.DataFrame(index=heads, columns=heads, dtype=float)
        n_matrix    = pd.DataFrame(index=heads, columns=heads, dtype=float)

        top_regions = {
            head_model: set(df.loc[head_model].dropna().sort_values(ascending=False).index[:n_regions])
            for head_model in heads
        }

        for i in heads:
            for j in heads:
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

    def plot_heatmap(self, metric="mean", percentile=95, head_models=None,
                      n_regions=None, figsize=(8, 6)):
        """
        Affiche la heatmap (clustermap) de corrélation de Spearman entre
        têtes, pour CE protocole (self.study_id) — colorée et clusterisée
        par rho. Chaque cellule est annotée avec
        "rho : distance moyenne de rang". Si n_regions est un int, le
        nombre de régions comparées (N) est ajouté sur une ligne séparée,
        car il varie alors selon la paire de têtes (voir _topn_union_stats).

        Identique à CorrelationAnalysis.plot_heatmap, mais les lignes/
        colonnes sont des head_model plutôt que des study_id — avec
        typiquement 3 têtes, le dendrogramme du clustermap reste trivial,
        mais la structure (rho + distance de rang annotés, même colormap)
        reste cohérente avec le reste du pipeline.

        n_regions : None (défaut) → comparaison sur toutes les régions
                    communes aux deux têtes (comportement global).
                    int → comparaison, pour chaque PAIRE de têtes, sur
                    l'union de leurs top-n régions respectives.
        """
        df = self.compute_scores(metric=metric, percentile=percentile, n_regions="all")

        if head_models is not None:
            missing = set(head_models) - set(df.index)
            if missing:
                print(f"Les têtes suivantes n'existent pas parmi celles chargées : {sorted(missing)}")
            df = df.loc[df.index.intersection(head_models)]

        if n_regions is None:
            corr_matrix = df.T.corr(method="spearman")
            dist_matrix = self.rank_distance_matrix(metric=metric, percentile=percentile, head_models=head_models)
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

        n_heads = len(corr_matrix)
        metric_label = f"{metric} (P{percentile})" if metric == "percentile" else metric

        g = sns.clustermap(corr_matrix,
                    figsize=figsize,
                    annot=annot_df,
                    fmt="",
                    cmap="coolwarm",
                    vmin=-1, vmax=1,
                    linewidths=0.5,
                    annot_kws={"size": 8})

        g.figure.suptitle(f"Spearman rho : mean rank distance — {self.study_id} — {metric_label} | "
                    f"{n_heads} head models | {title_suffix}", fontsize=12, y=1.02)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

    def compute_stimulation_ratios(self, metric="mean", percentile=95, head_models=None):
        """
        Calcule le stimulation_ratio (région / référence globale) pour
        chaque tête, pour ce protocole (self.study_id).

        Contrairement à compute_scores() qui retourne des valeurs brutes de
        champ, cette méthode retourne des valeurs déjà normalisées — la
        morphologie de la tête change la distance scalp-cortex (donc la
        magnitude globale du champ) indépendamment d'où il se concentre ;
        stimulation_ratio() retire cet effet d'échelle par tête avant de
        comparer.

        metric     : métrique de référence passée à stimulation_ratio()
                    'mean', 'median', ou 'percentile'
        percentile : utilisé si metric='percentile'
        head_models : liste de head_model à inclure — None → toutes les
                      têtes chargées

        Retourne un DataFrame (têtes en lignes, régions en colonnes).
        """
        protocols = (
            self.protocols if head_models is None
            else {hm: self.protocols[hm] for hm in head_models if hm in self.protocols}
        )

        all_ratios = {
            head_model: p.stimulation_ratio(metric=metric, percentile=percentile)
            for head_model, p in protocols.items()
        }

        df = pd.DataFrame(all_ratios).T
        print(f"  {df.shape[1]} régions utilisées")

        return df

    def _pairwise_ratio_error(self, ratio_df):
        """
        Pour chaque paire de têtes, calcule le RMSE et le MAE des
        différences de stimulation_ratio, région par région, sur les
        régions présentes (non-NaN) dans les deux têtes.

        Identique à CorrelationAnalysis._pairwise_ratio_error, mais opère
        sur des head_model plutôt que des study_id.

        ratio_df : DataFrame de stimulation_ratio (têtes en lignes, régions
                en colonnes), typiquement compute_stimulation_ratios()

        Retourne trois DataFrames alignés (têtes x têtes) :
        rmse_matrix, mae_matrix, n_matrix (nombre de régions comparées par paire).
        """
        heads = ratio_df.index
        rmse_matrix = pd.DataFrame(index=heads, columns=heads, dtype=float)
        mae_matrix  = pd.DataFrame(index=heads, columns=heads, dtype=float)
        n_matrix    = pd.DataFrame(index=heads, columns=heads, dtype=float)

        for i in heads:
            for j in heads:
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

    def plot_error_heatmap(self, metric="mean", percentile=95, head_models=None,
                        figsize=(8, 6), cmap="YlOrRd", mae_tiny_threshold=0.01):
        """
        Affiche une heatmap (clustermap) du RMSE, du MAE, et de leur ratio
        entre têtes, pour CE protocole (self.study_id), calculés sur le
        stimulation_ratio par région (donc déjà normalisé par tête —
        contrairement à plot_heatmap() qui compare des rangs, ceci compare
        des écarts de magnitude région par région).

        Chaque cellule est annotée "RMSE : MAE : ratio". Le clustering et la
        couleur restent basés sur RMSE (magnitude) — le ratio est fourni
        comme contexte additionnel, pas comme signal principal, car il
        devient instable (bruit) quand RMSE et MAE sont tous deux proches
        de zéro.

        Interprétation (identique à CorrelationAnalysis.plot_error_heatmap) :
        RMSE >= MAE toujours (identité mathématique).
        - ratio proche de 1        → différences réparties uniformément
                                        entre régions.
        - ratio nettement > 1      → une ou quelques régions dominent
                                        l'écart entre têtes, à inspecter
                                        individuellement.
        - RMSE < MAE               → ne devrait jamais arriver, signale un bug.

        Important : RMSE et MAE ne disent RIEN sur le ranking (voir
        plot_heatmap() pour ça). Deux têtes peuvent avoir un ranking
        identique (rho=1, voir plot_heatmap) et un RMSE élevé si le profil
        relatif se décale uniformément d'une tête à l'autre — les deux
        heatmaps répondent à des questions différentes et se lisent ensemble.

        metric, percentile   : passés à compute_stimulation_ratios()
        head_models           : liste de head_model à inclure — None → toutes
                                les têtes chargées
        mae_tiny_threshold    : si MAE < ce seuil, le ratio est affiché avec
                                1 seule décimale plutôt que 2, car il devient
                                peu fiable (bruit numérique) quand les erreurs
                                sont négligeables.
        """
        ratio_df = self.compute_stimulation_ratios(
            metric=metric, percentile=percentile, head_models=head_models
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

        n_heads = len(rmse_matrix)
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
        )
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)

        g.figure.suptitle(
            f"RMSE : MAE : ratio of stimulation ratio — {self.study_id} — {metric_label} | "
            f"{n_heads} head models",
            fontsize=12, y=1.02,
        )
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

        return rmse_matrix, mae_matrix, ratio_matrix