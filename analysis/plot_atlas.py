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
HO_atlas      = False
atlas_name = "Harvard-Oxford+" if HO_atlas else "HO_115"

if not HO_atlas:
    ATLAS_LABELS_PATH = PROJECT_ROOT / "utils" / f"atlas_labels_{atlas_name}.json"
    with open(ATLAS_LABELS_PATH, "r") as f:
        ATLAS_LABELS = {int(k): v for k, v in json.load(f).items()}
# atlas_name = "Harvard-Oxford_combined"

# ── Chargement des données ────────────────────────────────────────────────────
file_name = f"viz_data_{atlas_name}.npz"
data          = np.load(PROJECT_ROOT / "results" / "results_tms_MA" / study_id / file_name)
nodes         = data["nodes"]
elm_nodes     = data["elm_nodes"]
magnE         = data["magnE"]
region_labels = data["region_labels"]
if HO_atlas:
    ATLAS_LABELS  = {int(k): v for k, v in enumerate(data["atlas_labels"])}


# ── Top N régions par focalité (P95/mean) ────────────────────────────────────
unique_labels = np.unique(region_labels[region_labels > 0])

if n > len(unique_labels):
    raise ValueError(f"La valeur de n est trop grande, il n'y a que {len(unique_labels)} régions")

p95_by_region  = {
    label: np.percentile(magnE[region_labels == label], percentile)
    for label in unique_labels
}
mean_by_region = {
    label: np.average(magnE[region_labels == label],
                      weights=np.ones(np.sum(region_labels == label)))
    for label in unique_labels
}

metrics = {"percentile"  : p95_by_region,
            "mean" : mean_by_region,
            "focality" : {
                    label: p95_by_region[label] / mean_by_region[label]
                    for label in unique_labels
                    if mean_by_region[label] > 0
                }}

top_labels = sorted(metrics[metric], key=metrics[metric].get, reverse=True)[:n]

# ── Construction des deux grilles indépendantes ───────────────────────────────
n_elms     = elm_nodes.shape[0]
cells      = np.hstack([np.full((n_elms, 1), 4, dtype=int), elm_nodes]).ravel()
cell_types = np.full(n_elms, 10)

grid_atlas = pv.UnstructuredGrid(cells, cell_types, nodes)
grid_E     = pv.UnstructuredGrid(cells, cell_types, nodes)
grid_E["magnE"] = magnE

region_display = np.full(len(region_labels), 0.0)
region_display[region_labels == 0] = -1.0
for rank, label in enumerate(top_labels):
    region_display[region_labels == label] = float(rank + 1)
grid_atlas["atlas_top"] = region_display










# ── Colormap discrète atlas ───────────────────────────────────────────────────

base_color = plt.cm.tab20.colors
label_colors = list(base_color)[:n]

all_colors  = [
    (0.2, 0.2, 0.2, 1.0),
    (0.4, 0.4, 0.4, 1.0),
] + label_colors

if n > 20:
    colors_sup = plt.cm.terrain(np.linspace(0, 0.95, n-20))
    # Mélange aléatoire de l'ordre
    rng = np.random.default_rng(seed=42)
    rng.shuffle(colors_sup)
    colors_sup = [tuple(map(float, c)) for c in colors_sup[:, :3]] # convertire en list de tuple sans transparence
    all_colors += list(colors_sup)
    label_colors += list(colors_sup)



cmap_label = ListedColormap(label_colors, N=n)
cmap_atlas = ListedColormap(all_colors, N=n+2)

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
                         n_colors=cmap.N if hasattr(cmap, "N") else 256,
                         show_scalar_bar=False,
                         name=actor_name)
        plotter.render()

    return callback



# ── Plotter ───────────────────────────────────────────────────────────────────
plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

# ── Vue gauche — Atlas ────────────────────────────────────────────────────────
plotter.subplot(0, 0)
plotter.add_text(f"Atlas {atlas_name} — Top {n} regions P{percentile} | {study_id}", font_size=10)

plotter.add_mesh(grid_atlas,
                 scalars="atlas_top",
                 cmap=cmap_atlas,
                 clim=[-1, n],
                 n_colors=cmap_atlas.N,  
                 show_scalar_bar=False,
                 name="atlas_mesh")

if show_no_atlas:
    legend_entries = [["other",   [0.4, 0.4, 0.4]],
                    ["no atlas", [0.2, 0.2, 0.2]]]
else:
    legend_entries = [["other",   [0.4, 0.4, 0.4]]]
    
for rank, label in enumerate(top_labels):
    color = list(cmap_label(rank))[:3]
    name  = ATLAS_LABELS.get(label, f"Unknown_{label}")
    legend_entries.append([f"{label} - {name}", color])
    # legend_entries.append([str(label), color])

plotter.add_legend(legend_entries,
                   size=(0.35, 0.7),
                   loc="upper left",
                   font_family="courier",
                   face="rectangle")

cb_atlas = make_clip_callback(grid_atlas, "atlas_top", cmap_atlas,
                               [-1, n], plotter, "atlas_mesh", (0, 0),threshold=-0.5 if not show_no_atlas else None)

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