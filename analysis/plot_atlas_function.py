import numpy as np
import pyvista as pv

from plot_atlas_utils import (
    SLIDER_KWARGS,
    ATLAS_ACTOR_NAME,
    E_MESH_ACTOR_NAME,
    IDLE_OPACITY,
    _build_grids,
    _build_colormaps,
    _legend_entries_all,
    _setup_legend,
    _apply_highlight,
    _make_clip_callback,
    _make_e_field_callback,
    _enable_region_interaction,
    _setup_panel_sync,
)


def plot_atlas_3d(protocole, ranked, metric="mean", percentile=95,
                   show_no_atlas=False, title=None, window_size=(1600, 800),
                   show_legend=True, enable_interaction=True):
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

    # État partagé entre les deux panneaux, utilisé par la synchronisation
    # (case à cocher "Sync panels" mise en place plus bas).
    sync_state = {"linked": False}
    selection_state = {"selected_rank": None, "clipped_grid": grid_atlas}
    atlas_clip_state = {"x": None, "y": None, "z": None}
    e_clip_state = {"x": None, "y": None, "z": None}
    atlas_sliders = {}
    e_sliders = {}

    def make_atlas_axis_cb(axis):
        def _cb(value):
            atlas_clip_state[axis] = value
            cb_atlas(value, axis)
            if sync_state["linked"]:
                e_clip_state[axis] = value
                cb_E(value, axis)
                e_sliders[axis].GetRepresentation().SetValue(value)
        return _cb

    def make_e_axis_cb(axis):
        def _cb(value):
            e_clip_state[axis] = value
            cb_E(value, axis)
            if sync_state["linked"]:
                atlas_clip_state[axis] = value
                cb_atlas(value, axis)
                atlas_sliders[axis].GetRepresentation().SetValue(value)
        return _cb

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
                      opacity=IDLE_OPACITY,
                      show_scalar_bar=False,
                      pickable=True,
                      name=ATLAS_ACTOR_NAME)

    if show_legend:
        entries = _legend_entries_all(region_ids, labels, cmap_label, show_no_atlas)
        _setup_legend(plotter, entries, subplot=(0, 0))

    def _on_atlas_clipped(clipped):
        selection_state["clipped_grid"] = clipped
        if selection_state["selected_rank"] is not None:
            _apply_highlight(plotter, clipped, selection_state["selected_rank"],
                              cmap_label, (0, 0))

    cb_atlas = _make_clip_callback(
        grid_atlas, "atlas_top", cmap_atlas, [-1, n], plotter, ATLAS_ACTOR_NAME, (0, 0),
        n_colors=cmap_atlas.N, threshold=-0.5 if not show_no_atlas else None,
        opacity=IDLE_OPACITY, on_clipped=_on_atlas_clipped)

    atlas_sliders["x"] = plotter.add_slider_widget(callback=make_atlas_axis_cb("x"),
                               rng=[nodes[:, 0].min(), nodes[:, 0].max()],
                               value=nodes[:, 0].max(), title="X",
                               pointa=(0.02, 0.18), pointb=(0.98, 0.18), **SLIDER_KWARGS)
    atlas_sliders["y"] = plotter.add_slider_widget(callback=make_atlas_axis_cb("y"),
                               rng=[nodes[:, 1].min(), nodes[:, 1].max()],
                               value=nodes[:, 1].max(), title="Y",
                               pointa=(0.02, 0.10), pointb=(0.98, 0.10), **SLIDER_KWARGS)
    atlas_sliders["z"] = plotter.add_slider_widget(callback=make_atlas_axis_cb("z"),
                               rng=[nodes[:, 2].min(), nodes[:, 2].max()],
                               value=nodes[:, 2].max(), title="Z",
                               pointa=(0.02, 0.02), pointb=(0.98, 0.02), **SLIDER_KWARGS)
    atlas_clip_state.update(x=nodes[:, 0].max(), y=nodes[:, 1].max(), z=nodes[:, 2].max())

    # ── Vue droite — Champ E ────────────────────────────────────────────
    plotter.subplot(0, 1)
    magnE = grid_E["magnE"]
    plotter.add_text(f"E-field magnE | {protocole.study_id}", font_size=10)
    plotter.add_mesh(grid_E, scalars="magnE", cmap="turbo",
                      clim=[magnE.min(), magnE.max()],
                      show_scalar_bar=False, pickable=False, name=E_MESH_ACTOR_NAME)
    plotter.add_scalar_bar(title="E-field (V/m per A/µs)", n_labels=5,
                            label_font_size=10, title_font_size=12,
                            position_x=0.05, position_y=0.86,
                            width=0.9, height=0.08, vertical=False)

    cb_E = _make_e_field_callback(grid_E, [magnE.min(), magnE.max()], plotter, (0, 1))
    e_sliders["x"] = plotter.add_slider_widget(callback=make_e_axis_cb("x"),
                               rng=[nodes[:, 0].min(), nodes[:, 0].max()],
                               value=nodes[:, 0].max(), title="X",
                               pointa=(0.02, 0.18), pointb=(0.98, 0.18), **SLIDER_KWARGS)
    e_sliders["y"] = plotter.add_slider_widget(callback=make_e_axis_cb("y"),
                               rng=[nodes[:, 1].min(), nodes[:, 1].max()],
                               value=nodes[:, 1].max(), title="Y",
                               pointa=(0.02, 0.10), pointb=(0.98, 0.10), **SLIDER_KWARGS)
    e_sliders["z"] = plotter.add_slider_widget(callback=make_e_axis_cb("z"),
                               rng=[nodes[:, 2].min(), nodes[:, 2].max()],
                               value=nodes[:, 2].max(), title="Z",
                               pointa=(0.02, 0.02), pointb=(0.98, 0.02), **SLIDER_KWARGS)
    e_clip_state.update(x=nodes[:, 0].max(), y=nodes[:, 1].max(), z=nodes[:, 2].max())

    # Contexte partagé pour le dim du panneau champ E, déclenché soit par un
    # clic sur une région (si sync déjà actif), soit par l'activation du
    # bouton "Sync panels" (si une région est déjà sélectionnée).
    e_sync_ctx = {
        "region_ids": region_ids,
        "sync_state": sync_state,
        "selection_state": selection_state,
        "set_dim": cb_E.set_dim,
    }

    if enable_interaction:
        _enable_region_interaction(plotter, (0, 0), region_ids, labels, cmap_label,
                                    selection_state=selection_state, e_sync_ctx=e_sync_ctx)

    _setup_panel_sync(plotter, (0, 0), (0, 1), cb_atlas, cb_E,
                       atlas_sliders, e_sliders, atlas_clip_state, e_clip_state,
                       sync_state, button_subplot=(0, 1), e_sync_ctx=e_sync_ctx)

    plotter.show()
    return plotter