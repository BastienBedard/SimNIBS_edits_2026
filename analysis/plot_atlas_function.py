import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def _build_grids(protocole):
    """Construit les deux UnstructuredGrid (atlas, E-field) à partir de l'objet ProtocoleAnalysis."""
    nodes     = protocole.msh.nodes.node_coord
    elm_nodes = protocole.msh.elm.node_number_list[protocole.tissue_mask] - 1  # SimNIBS 1-indexé
    magnE     = protocole.magnE
    region_labels = protocole.region_labels

    n_elms     = elm_nodes.shape[0]
    cells      = np.hstack([np.full((n_elms, 1), 4, dtype=int), elm_nodes]).ravel()
    cell_types = np.full(n_elms, 10)  # VTK_TETRA

    grid_atlas = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid_E     = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid_E["magnE"] = magnE

    return nodes, grid_atlas, grid_E, region_labels


def _build_colormaps(n):
    """Colormap discrète : -1 (hors tissu, gris foncé), 0 (autre région, gris clair), 1..n (régions rankées)."""
    base_color = plt.cm.tab20.colors
    label_colors = list(base_color)[:min(n, 20)]

    if n > 20:
        colors_sup = plt.cm.terrain(np.linspace(0, 0.95, n - 20))
        rng = np.random.default_rng(seed=42)
        rng.shuffle(colors_sup)
        colors_sup = [tuple(map(float, c)) for c in colors_sup[:, :3]]
        label_colors += colors_sup

    all_colors = [
        (0.2, 0.2, 0.2, 1.0),  # -1 : hors tissu
        (0.4, 0.4, 0.4, 1.0),  # 0  : autre région
    ] + label_colors

    cmap_label = ListedColormap(label_colors, N=n)
    cmap_atlas = ListedColormap(all_colors, N=n + 2)
    return cmap_label, cmap_atlas


def _make_clip_callback(grid, scalars, cmap, clim, plotter, actor_name, subplot,
                         n_colors=None, threshold=None):
    state = {"x": None, "y": None, "z": None}

    def callback(value, axis):
        state[axis] = value
        clipped = grid.copy()
        for ax, val in state.items():
            if val is not None:
                normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax]
                origin = {"x": (val, 0, 0), "y": (0, val, 0), "z": (0, 0, val)}[ax]
                clipped = clipped.clip(normal=normal, origin=origin)

        if threshold is not None:
            clipped = clipped.threshold(value=threshold, scalars=scalars)

        plotter.subplot(*subplot)
        kwargs = dict(scalars=scalars, cmap=cmap, clim=clim,
                      show_scalar_bar=False, name=actor_name)
        if n_colors is not None:
            kwargs["n_colors"] = n_colors
        plotter.add_mesh(clipped, **kwargs)
        plotter.render()

    return callback


def plot_atlas_3d(protocole, ranked, metric="mean", percentile=95,
                   show_no_atlas=False, title=None, window_size=(1600, 800)):
    """
    Affiche l'interface pyvista (atlas top-N à gauche, champ E à droite) pour un
    ranking déjà calculé par ProtocoleAnalysis.rank_regions().

    protocole : instance de ProtocoleAnalysis (fournit le mesh, magnE, region_labels, labels)
    ranked    : liste de tuples (region_id, valeur), typiquement la sortie de rank_regions()
    """
    region_ids = [r[0] for r in ranked]
    n = len(region_ids)
    if n == 0:
        raise ValueError("`ranked` est vide, rien à afficher.")

    labels = protocole.get_labels_for_regions(region_ids)

    nodes, grid_atlas, grid_E, region_labels = _build_grids(protocole)

    region_display = np.full(len(region_labels), 0.0)
    region_display[region_labels == 0] = -1.0
    for rank, label in enumerate(region_ids):
        region_display[region_labels == label] = float(rank + 1)
    grid_atlas["atlas_top"] = region_display

    cmap_label, cmap_atlas = _build_colormaps(n)

    plotter = pv.Plotter(shape=(1, 2), window_size=window_size, notebook=False)

    # ── Vue gauche — Atlas ──────────────────────────────────────────────
    plotter.subplot(0, 0)
    metric_label = f"P{percentile}" if metric in ("percentile", "focality") else metric
    header = title if title is not None else f"Atlas — Top {n} regions {metric_label} | {protocole.study_id}"
    plotter.add_text(header, font_size=10)

    plotter.add_mesh(grid_atlas,
                      scalars="atlas_top",
                      cmap=cmap_atlas,
                      clim=[-1, n],
                      n_colors=cmap_atlas.N,
                      show_scalar_bar=False,
                      name="atlas_mesh")

    legend_entries = [["other", [0.4, 0.4, 0.4]]]
    if show_no_atlas:
        legend_entries.append(["no atlas", [0.2, 0.2, 0.2]])
    for rank, (label_id, name) in enumerate(zip(region_ids, labels)):
        color = list(cmap_label(rank))[:3]
        legend_entries.append([f"{label_id} - {name}", color])

    plotter.add_legend(legend_entries, size=(0.35, 0.7), loc="upper left",
                        font_family="courier", face="rectangle")

    cb_atlas = _make_clip_callback(
        grid_atlas, "atlas_top", cmap_atlas, [-1, n], plotter, "atlas_mesh", (0, 0),
        n_colors=cmap_atlas.N, threshold=-0.5 if not show_no_atlas else None)

    plotter.add_slider_widget(callback=lambda v: cb_atlas(v, "x"),
                               rng=[nodes[:, 0].min(), nodes[:, 0].max()],
                               value=nodes[:, 0].max(), title="X",
                               pointa=(0.02, 0.21), pointb=(0.95, 0.21))
    plotter.add_slider_widget(callback=lambda v: cb_atlas(v, "y"),
                               rng=[nodes[:, 1].min(), nodes[:, 1].max()],
                               value=nodes[:, 1].max(), title="Y",
                               pointa=(0.02, 0.14), pointb=(0.95, 0.14))
    plotter.add_slider_widget(callback=lambda v: cb_atlas(v, "z"),
                               rng=[nodes[:, 2].min(), nodes[:, 2].max()],
                               value=nodes[:, 2].max(), title="Z",
                               pointa=(0.02, 0.05), pointb=(0.95, 0.05))

    # ── Vue droite — Champ E ────────────────────────────────────────────
    plotter.subplot(0, 1)
    magnE = grid_E["magnE"]
    plotter.add_text(f"E-field magnE | {protocole.study_id}", font_size=10)
    plotter.add_mesh(grid_E, scalars="magnE", cmap="turbo",
                      clim=[magnE.min(), magnE.max()],
                      show_scalar_bar=False, name="E_mesh")
    plotter.add_scalar_bar(title="E-field (V/m per A/µs)", n_labels=5,
                            label_font_size=10, title_font_size=12,
                            position_x=0.02, position_y=0.02,
                            width=0.95, height=0.15, vertical=False)

    cb_E = _make_clip_callback(grid_E, "magnE", "turbo",
                                [magnE.min(), magnE.max()], plotter, "E_mesh", (0, 1))
    plotter.add_slider_widget(callback=lambda v: cb_E(v, "x"),
                               rng=[nodes[:, 0].min(), nodes[:, 0].max()],
                               value=nodes[:, 0].max(), title="X",
                               pointa=(0.02, 0.31), pointb=(0.95, 0.31))
    plotter.add_slider_widget(callback=lambda v: cb_E(v, "y"),
                               rng=[nodes[:, 1].min(), nodes[:, 1].max()],
                               value=nodes[:, 1].max(), title="Y",
                               pointa=(0.02, 0.24), pointb=(0.95, 0.24))
    plotter.add_slider_widget(callback=lambda v: cb_E(v, "z"),
                               rng=[nodes[:, 2].min(), nodes[:, 2].max()],
                               value=nodes[:, 2].max(), title="Z",
                               pointa=(0.02, 0.17), pointb=(0.95, 0.17))

    plotter.show()

    return plotter