import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from pathlib import Path
import json

PROJECT_ROOT  = Path(__file__).parent.parent
study_id      = "addicott2024"
n             = 20
percentile    = 95
metric = "mean"
show_no_atlas = False

ATLAS_LABELS_PATH = PROJECT_ROOT / "utils" / "atlas_labels.json"
with open(ATLAS_LABELS_PATH, "r") as f:
    ATLAS_LABELS = {int(k): v for k, v in json.load(f).items()}

# ── Chargement des données ────────────────────────────────────────────────────
data          = np.load(PROJECT_ROOT / "results" / "results_tms_MA" / study_id / "viz_data.npz")
nodes         = data["nodes"]
elm_nodes     = data["elm_nodes"]
magnE         = data["magnE"]
region_labels = data["region_labels"]

# ── Construction des deux grilles indépendantes ───────────────────────────────
n_elms     = elm_nodes.shape[0]
cells      = np.hstack([np.full((n_elms, 1), 4, dtype=int), elm_nodes]).ravel()
cell_types = np.full(n_elms, 10)

grid_atlas = pv.UnstructuredGrid(cells, cell_types, nodes)
grid_E     = pv.UnstructuredGrid(cells, cell_types, nodes)
grid_E["magnE"] = magnE

# ── Top N régions par focalité (P95/mean) ────────────────────────────────────
unique_labels = np.unique(region_labels[region_labels > 0])

p95_by_region  = {
    label: np.percentile(magnE[region_labels == label], percentile)
    for label in unique_labels
}
mean_by_region = {
    label: np.average(magnE[region_labels == label],
                      weights=np.ones(np.sum(region_labels == label)))
    for label in unique_labels
}
focality_by_region = {
    label: p95_by_region[label] / mean_by_region[label]
    for label in unique_labels
    if mean_by_region[label] > 0
}

top_labels = sorted(focality_by_region, key=focality_by_region.get, reverse=True)[:n]

region_display = np.full(len(region_labels), 0.0)
region_display[region_labels == 0] = -1.0
for rank, label in enumerate(top_labels):
    region_display[region_labels == label] = float(rank + 1)
grid_atlas["atlas_top"] = region_display

# ── Colormap discrète atlas ───────────────────────────────────────────────────
base_colors = list(plt.cm.tab20.colors)
all_colors  = [
    (0.2, 0.2, 0.2, 1.0),
    (0.3, 0.3, 0.3, 1.0),
] + base_colors[:n]
cmap_atlas = ListedColormap(all_colors)

# ── Fonction de clip ──────────────────────────────────────────────────────────
def make_clip_callback(grid, scalars, cmap, clim, plotter, actor_name, subplot, threshold=None):
    state = {"x": None, "y": None, "z": None}

    def callback(value, axis):
        state[axis] = value
        clipped = grid.copy()
        for ax, val in state.items():
            if val is not None:
                normal = {"x": (1,0,0), "y": (0,1,0), "z": (0,0,1)}[ax]
                origin = {"x": (val,0,0), "y": (0,val,0), "z": (0,0,val)}[ax]
                clipped = clipped.clip(normal=normal, origin=origin)

        if threshold is not None:
            clipped = clipped.threshold(value=threshold, scalars=scalars)

        plotter.subplot(*subplot)
        plotter.add_mesh(clipped,
                         scalars=scalars,
                         cmap=cmap,
                         clim=clim,
                         show_scalar_bar=False,
                         name=actor_name)
        plotter.render()

    return callback



# ── Plotter ───────────────────────────────────────────────────────────────────
plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

# ── Vue gauche — Atlas ────────────────────────────────────────────────────────
plotter.subplot(0, 0)
plotter.add_text(f"Atlas AAL2 — Top {n} regions P{percentile} | {study_id}", font_size=10)

if show_no_atlas:
    grid_atlas_display = grid_atlas
else:
    # filtre seulement les éléments à -1 (hors tissu) 
    # garde les 0 (GM/WM hors top N) en gris
    grid_atlas_display = grid_atlas.threshold(value=-0.5, scalars="atlas_top")

plotter.add_mesh(grid_atlas_display,
                 scalars="atlas_top",
                 cmap=cmap_atlas,
                 clim=[-1, n],
                 show_scalar_bar=False,
                 name="atlas_mesh")

legend_entries = [["other",   [0.3, 0.3, 0.3]],
                  ["no atlas", [0.2, 0.2, 0.2]]]
for rank, label in enumerate(top_labels):
    color = list(cmap_atlas(float(rank + 2) / (n + 2)))[:3]
    name  = ATLAS_LABELS.get(label, f"Unknown_{label}")
    legend_entries.append([f"{label} - {name}", color])
    # legend_entries.append([str(label), color])

plotter.add_legend(legend_entries,
                   size=(0.25, 0.7),
                   loc="upper left",
                   font_family="courier",
                   face="rectangle")

cb_atlas = make_clip_callback(grid_atlas, "atlas_top", cmap_atlas,
                               [-1, n], plotter, "atlas_mesh", (0, 0),
                               threshold=-0.5 if not show_no_atlas else None)

plotter.add_slider_widget(
    callback=lambda v: cb_atlas(v, "x"),
    rng=[nodes[:, 0].min(), nodes[:, 0].max()],
    value=nodes[:, 0].max(), title="X",
    pointa=(0.02, 0.21), pointb=(0.95, 0.21)
)
plotter.add_slider_widget(
    callback=lambda v: cb_atlas(v, "y"),
    rng=[nodes[:, 1].min(), nodes[:, 1].max()],
    value=nodes[:, 1].max(), title="Y",
    pointa=(0.02, 0.14), pointb=(0.95, 0.14)
)
plotter.add_slider_widget(
    callback=lambda v: cb_atlas(v, "z"),
    rng=[nodes[:, 2].min(), nodes[:, 2].max()],
    value=nodes[:, 2].max(), title="Z",
    pointa=(0.02, 0.05), pointb=(0.95, 0.05)
)

# ── Vue droite — Champ E ──────────────────────────────────────────────────────
plotter.subplot(0, 1)
plotter.add_text(f"E-field magnE | {study_id}", font_size=10)
plotter.add_mesh(grid_E,
                 scalars="magnE",
                 cmap="turbo",
                 clim=[magnE.min(), magnE.max()],
                 show_scalar_bar=False,
                 name="E_mesh")
plotter.add_scalar_bar(title="E-field (V/m per A/µs)",
                       n_labels=5,
                       label_font_size=10,
                       title_font_size=12,
                       position_x=0.02,
                       position_y=0.02,
                       width=0.95,
                       height=0.15,
                       vertical=False)

cb_E = make_clip_callback(grid_E, "magnE", "turbo",
                           [magnE.min(), magnE.max()], plotter, "E_mesh", (0, 1))
plotter.add_slider_widget(
    callback=lambda v: cb_E(v, "x"),
    rng=[nodes[:, 0].min(), nodes[:, 0].max()],
    value=nodes[:, 0].max(), title="X",
    pointa=(0.02, 0.31), pointb=(0.95, 0.31)
)
plotter.add_slider_widget(
    callback=lambda v: cb_E(v, "y"),
    rng=[nodes[:, 1].min(), nodes[:, 1].max()],
    value=nodes[:, 1].max(), title="Y",
    pointa=(0.02, 0.24), pointb=(0.95, 0.24)
)
plotter.add_slider_widget(
    callback=lambda v: cb_E(v, "z"),
    rng=[nodes[:, 2].min(), nodes[:, 2].max()],
    value=nodes[:, 2].max(), title="Z",
    pointa=(0.02, 0.17), pointb=(0.95, 0.17)
)

# score_by_region = focality_by_region if metric == "focality" else p95_by_region
# print(f"\n{'Rank':<6} {'Region':<35} {'Score':>15}")
# print("-" * 58)
# for rank, label in enumerate(top_labels):
#     name = ATLAS_LABELS.get(label, f"Unknown_{label}")
#     print(f"{rank+1:<6} {name + ' - ' + str(label):<35} {score_by_region[label]:.4f}")

plotter.show()