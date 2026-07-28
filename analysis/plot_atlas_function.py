import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from vtkmodules.vtkRenderingCore import vtkCellPicker

SLIDER_KWARGS = dict(style="modern", tube_width=0.004, slider_width=0.012,
                      title_height=0.015)

MAX_LEGEND_NAME_LEN = 26
LEGEND_WINDOW = 15          # nombre d'entrées visibles à la fois dans la légende
LEGEND_ACTOR_NAME = "atlas_legend"

ATLAS_ACTOR_NAME = "atlas_mesh"
IDLE_OPACITY = 1.0
DIM_OPACITY = 0.4  # opacité du reste du cerveau quand une région est sélectionnée


def _build_grids(protocole):
    nodes     = protocole.msh.nodes.node_coord
    elm_nodes = protocole.msh.elm.node_number_list[protocole.tissue_mask] - 1
    magnE     = protocole.magnE
    region_labels = protocole.region_labels

    n_elms     = elm_nodes.shape[0]
    cells      = np.hstack([np.full((n_elms, 1), 4, dtype=int), elm_nodes]).ravel()
    cell_types = np.full(n_elms, 10)

    grid_atlas = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid_E     = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid_E["magnE"] = magnE
    grid_E["region_labels"] = region_labels  # used to split the E panel for the dim-on-select feature

    return nodes, grid_atlas, grid_E, region_labels


def _build_colormaps(n):
    base_color = plt.cm.tab20.colors
    label_colors = list(base_color)[:min(n, 20)]

    if n > 20:
        colors_sup = plt.cm.terrain(np.linspace(0, 0.95, n - 20))
        rng = np.random.default_rng(seed=42)
        rng.shuffle(colors_sup)
        colors_sup = [tuple(map(float, c)) for c in colors_sup[:, :3]]
        label_colors += colors_sup

    all_colors = [
        (0.2, 0.2, 0.2, 1.0),
        (0.4, 0.4, 0.4, 1.0),
    ] + label_colors

    cmap_label = ListedColormap(label_colors, N=n)
    cmap_atlas = ListedColormap(all_colors, N=n + 2)
    return cmap_label, cmap_atlas


def _truncate(name, max_len=MAX_LEGEND_NAME_LEN):
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def _legend_entries_all(region_ids, labels, cmap_label, show_no_atlas):
    """Format : rang - nom - #id (le rang correspond à l'ordre du ranking)."""
    entries = [["other", [0.4, 0.4, 0.4]]]
    if show_no_atlas:
        entries.append(["no atlas", [0.2, 0.2, 0.2]])
    for rank, (label_id, name) in enumerate(zip(region_ids, labels)):
        color = list(cmap_label(rank))[:3]
        entries.append([f"{rank + 1} - {_truncate(name)}", color])
    return entries


def _make_clip_callback(grid, scalars, cmap, clim, plotter, actor_name, subplot,
                         n_colors=None, threshold=None, opacity=1.0, on_clipped=None):
    """
    `opacity` may be a fixed float, or the name of a data array on `grid`
    for PyVista's built-in array-based opacity mapping. Since `refresh()`
    re-copies `grid` on every call, mutating a referenced array in place
    (grid[name] = new_values) and calling `callback.refresh()` picks up
    the change while preserving whatever clip planes are currently active.

    `on_clipped`, if provided, is called with the freshly clipped/thresholded
    grid every time `_render()` runs (slider move, refresh, or initial call),
    so callers can keep derived actors (e.g. a region highlight) in sync
    with whatever geometry is currently on screen.
    """
    state = {"x": None, "y": None, "z": None}

    def _render():
        clipped = grid.copy()
        for ax, val in state.items():
            if val is not None:
                normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax]
                origin = {"x": (val, 0, 0), "y": (0, val, 0), "z": (0, 0, val)}[ax]
                clipped = clipped.clip(normal=normal, origin=origin)

        if threshold is not None:
            clipped = clipped.threshold(value=threshold, scalars=scalars)

        plotter.subplot(*subplot)
        kwargs = dict(scalars=scalars, cmap=cmap, clim=clim, opacity=opacity,
                      show_scalar_bar=False, name=actor_name)
        if n_colors is not None:
            kwargs["n_colors"] = n_colors
        plotter.add_mesh(clipped, **kwargs)

        if on_clipped is not None:
            on_clipped(clipped)

        plotter.render()

    def callback(value, axis):
        state[axis] = value
        _render()

    callback.refresh = _render
    return callback


def _set_atlas_opacity(plotter, value):
    actor = plotter.actors.get(ATLAS_ACTOR_NAME)
    if actor is not None:
        actor.GetProperty().SetOpacity(value)


def _make_e_field_callback(grid_E, clim, plotter, subplot,
                            main_actor_name="E_mesh", highlight_actor_name="E_mesh_selected",
                            dim_opacity=0.1):
    """
    Clip callback for the E-field panel. Also supports dimming everything
    outside a selected atlas region via `set_dim(region_id)`, used when
    "Sync panels" is on and a region is selected on the atlas panel: the
    selected region stays at full opacity while the rest of the brain drops
    to `dim_opacity`. Rendered as two actors, since opacity in VTK/PyVista is
    a per-actor property and isn't reliably driven by a per-cell data array
    across repeated add_mesh calls on the same actor name.
    """
    clip_state = {"x": None, "y": None, "z": None}
    dim_state = {"region_id": None}

    def _render():
        clipped = grid_E.copy()
        for ax, val in clip_state.items():
            if val is not None:
                normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax]
                origin = {"x": (val, 0, 0), "y": (0, val, 0), "z": (0, 0, val)}[ax]
                clipped = clipped.clip(normal=normal, origin=origin)

        plotter.subplot(*subplot)
        region_id = dim_state["region_id"]

        if region_id is None:
            try:
                plotter.remove_actor(highlight_actor_name, render=False)
            except (KeyError, ValueError):
                pass
            plotter.add_mesh(clipped, scalars="magnE", cmap="turbo", clim=clim,
                              opacity=1.0, show_scalar_bar=False, pickable=False,
                              name=main_actor_name)
        else:
            mask = clipped.cell_data["region_labels"] == region_id
            rest = clipped.extract_cells(np.where(~mask)[0])
            selected = clipped.extract_cells(np.where(mask)[0])
            plotter.add_mesh(rest, scalars="magnE", cmap="turbo", clim=clim,
                              opacity=dim_opacity, show_scalar_bar=False, pickable=False,
                              name=main_actor_name)
            plotter.add_mesh(selected, scalars="magnE", cmap="turbo", clim=clim,
                              opacity=1.0, show_scalar_bar=False, pickable=False,
                              name=highlight_actor_name)

        plotter.render()

    def callback(value, axis):
        clip_state[axis] = value
        _render()

    def set_dim(region_id):
        dim_state["region_id"] = region_id
        _render()

    callback.refresh = _render
    callback.set_dim = set_dim
    return callback


def _add_legend_window(plotter, entries, start, window=LEGEND_WINDOW):
    subset = entries[start:start + window]
    legend_height = min(0.85, 0.10 + 0.035 * len(subset))
    plotter.add_legend(subset, size=(0.32, legend_height), loc="upper left",
                        font_family="courier", face="rectangle", name=LEGEND_ACTOR_NAME)


def _setup_legend(plotter, entries, subplot):
    """
    Affiche la légende, avec une case à cocher permettant de l'afficher/masquer.
    Si trop d'entrées, ajoute aussi un slider de défilement (masqué avec la
    légende lorsque la case est décochée).
    """
    plotter.subplot(*subplot)

    has_scroll = len(entries) > LEGEND_WINDOW
    state = {"start": 0, "visible": True}

    _add_legend_window(plotter, entries, start=0)

    scroll_widget = None
    if has_scroll:
        max_start = len(entries) - LEGEND_WINDOW

        def on_scroll(value):
            plotter.subplot(*subplot)
            state["start"] = int(round(value))
            _add_legend_window(plotter, entries, start=state["start"])
            plotter.render()

        scroll_widget = plotter.add_slider_widget(callback=on_scroll, rng=[0, max_start], value=0,
                                   title="Scroll régions", fmt="%.0f",
                                   pointa=(0.02, 0.26), pointb=(0.32, 0.26),
                                   **SLIDER_KWARGS)

    def on_toggle_legend(flag):
        plotter.subplot(*subplot)
        state["visible"] = flag
        if flag:
            _add_legend_window(plotter, entries, start=state["start"])
        else:
            try:
                plotter.remove_actor(LEGEND_ACTOR_NAME, render=False)
            except (KeyError, ValueError):
                pass
        if scroll_widget is not None:
            if flag:
                scroll_widget.EnabledOn()
            else:
                scroll_widget.EnabledOff()
        plotter.render()

    # La case à cocher utilise des coordonnées pixel relatives à la fenêtre
    # entière (et non normalisées au sous-graphique comme les sliders), donc
    # on la positionne dans le coin supérieur gauche via la taille réelle de
    # la fenêtre plutôt qu'une valeur normalisée.
    win_w, win_h = plotter.window_size
    button_x, button_y, button_size = 10, win_h - 50, 20
    plotter.add_checkbox_button_widget(on_toggle_legend, value=True,
                                        position=(button_x, button_y),
                                        size=button_size,
                                        color_on="lightgreen", color_off="grey",
                                        background_color="white")
    plotter.add_text("Legend", name="legend_toggle_label", font_size=9,
                      position=(button_x + button_size + 8, button_y + 2),
                      color="white", shadow=True)


def _pick_rank_at(picker, x, y, renderer):
    """Retourne (rank_val, dataset) pour la cellule sous (x, y), ou (None, None) si rien."""
    picker.Pick(x, y, 0, renderer)
    cell_id = picker.GetCellId()
    if cell_id == -1:
        return None, None
    dataset = pv.wrap(picker.GetDataSet())
    if dataset is None or "atlas_top" not in dataset.cell_data:
        return None, None
    rank_val = int(dataset.cell_data["atlas_top"][cell_id])
    return rank_val, dataset


def _apply_highlight(plotter, clipped_grid, rank_val, cmap_label, atlas_subplot):
    """
    (Re)construit l'acteur `highlight_mesh` à partir de la géométrie
    actuellement affichée (`clipped_grid`), qu'elle soit découpée ou non.
    Appelée à la fois lors d'un clic et à chaque déplacement des sliders
    de découpe, afin que la surbrillance reste toujours cohérente avec ce
    qui est réellement visible à l'écran (et ne reste pas figée sur la
    forme qu'avait la région au moment de la sélection).
    """
    plotter.subplot(*atlas_subplot)

    if rank_val is None or "atlas_top" not in clipped_grid.cell_data:
        try:
            plotter.remove_actor("highlight_mesh", render=False)
        except (KeyError, ValueError):
            pass
        _set_atlas_opacity(plotter, IDLE_OPACITY)
        return

    mask = clipped_grid.cell_data["atlas_top"] == rank_val
    if not np.any(mask):
        # La région sélectionnée est actuellement entièrement découpée :
        # on retire l'ancienne surbrillance plutôt que de laisser un
        # résidu obsolète affiché.
        try:
            plotter.remove_actor("highlight_mesh", render=False)
        except (KeyError, ValueError):
            pass
        return

    highlighted = clipped_grid.extract_cells(np.where(mask)[0])
    color = list(cmap_label(rank_val - 1))[:3]
    _set_atlas_opacity(plotter, DIM_OPACITY)
    plotter.add_mesh(highlighted, color=color, opacity=1.0,
                      show_edges=True, edge_color="white", line_width=1,
                      name="highlight_mesh", pickable=False, reset_camera=False)


def _enable_region_interaction(plotter, atlas_subplot, region_ids, labels, cmap_label,
                                selection_state=None, e_sync_ctx=None):
    """
    Clic uniquement : sélectionne la région de façon persistante (surbrillance,
    opacité réduite du reste du cerveau, et nom de la région affiché en haut à
    droite) jusqu'à un clic sur la même région (désélection) ou une autre
    région (change la sélection). Un clic dans le vide ne fait rien, pour ne
    pas perdre la sélection en tournant la caméra.

    `selection_state` (dict with key "selected_rank") is shared with
    `_setup_panel_sync` so toggling the sync checkbox can react to whatever
    is currently selected. `e_sync_ctx`, if provided, drives the E-field
    panel dim (region full opacity / rest at 0.1) whenever the sync checkbox
    is enabled — see `_make_e_field_callback.set_dim`.
    """
    click_picker = vtkCellPicker()
    click_picker.SetTolerance(0.0005)

    label_to_name = {rid: name for rid, name in zip(region_ids, labels)}
    n_cols = plotter.shape[1]
    flat_index = atlas_subplot[0] * n_cols + atlas_subplot[1]
    atlas_renderer = plotter.renderers[flat_index]
    if selection_state is None:
        selection_state = {"selected_rank": None, "clipped_grid": None}
    selection_state.setdefault("clipped_grid", None)
    state = {"is_dragging": False, "press_pos": None}

    def _sync_e_panel(rank_val):
        if e_sync_ctx is None or not e_sync_ctx["sync_state"]["linked"]:
            return
        region_id = None if rank_val is None else region_ids[rank_val - 1]
        e_sync_ctx["set_dim"](region_id)

    def _restrict_picker(picker_obj, actor_name):
        """Limit `picker_obj` to a single named actor. Re-fetches the actor by
        name every time in case add_mesh swapped it for a new object (e.g.
        after a clip-slider move).

        plotter.actors is scoped to whichever subplot is currently active,
        which can drift to (0, 1) whenever an E-field slider is touched, so
        we force the atlas subplot active first or the lookup silently
        returns nothing."""
        plotter.subplot(*atlas_subplot)
        picker_obj.InitializePickList()
        actor = plotter.actors.get(actor_name)
        if actor is not None:
            picker_obj.AddPickList(actor)
        picker_obj.PickFromListOn()
        return actor is not None

    def _region_text(rank_val):
        region_id = region_ids[rank_val - 1]
        name = label_to_name.get(region_id, f"Region {region_id}")
        return f"{rank_val} - {name}"

    def _update_selected_text(rank_val):
        if rank_val is None or rank_val <= 0:
            try:
                plotter.remove_actor("selected_label", render=False)
            except (KeyError, ValueError):
                pass
        else:
            plotter.add_text(_region_text(rank_val), name="selected_label",
                              font_size=12, position="upper_right", color="white",
                              shadow=True)

    def _apply_selection(rank_val, dataset):
        plotter.subplot(*atlas_subplot)
        if dataset is not None:
            # `dataset` est la géométrie (découpée ou non) sur laquelle le
            # clic a eu lieu : on la mémorise comme référence courante pour
            # que les futurs déplacements de sliders repartent d'un état
            # cohérent, même si aucun nouveau slider n'a encore bougé.
            selection_state["clipped_grid"] = dataset
        _apply_highlight(plotter, selection_state["clipped_grid"], rank_val,
                          cmap_label, atlas_subplot)
        selection_state["selected_rank"] = rank_val
        _update_selected_text(rank_val)
        _sync_e_panel(rank_val)
        plotter.render()

    def on_press(caller, event):
        state["is_dragging"] = True
        state["press_pos"] = plotter.iren.interactor.GetEventPosition()

    def on_release(caller, event):
        state["is_dragging"] = False
        release_pos = plotter.iren.interactor.GetEventPosition()
        press_pos = state["press_pos"]
        state["press_pos"] = None
        if press_pos is None:
            return
        dx = release_pos[0] - press_pos[0]
        dy = release_pos[1] - press_pos[1]
        if dx * dx + dy * dy > 9:  # >3px de mouvement = rotation caméra, pas un clic
            return

        x, y = release_pos
        renderer = plotter.iren.get_poked_renderer(x, y)
        if renderer is not atlas_renderer:
            return
        _restrict_picker(click_picker, ATLAS_ACTOR_NAME)
        rank_val, dataset = _pick_rank_at(click_picker, x, y, renderer)
        if rank_val is None or rank_val <= 0:
            return  # clic dans le vide ou sur "other" : sélection inchangée

        if rank_val == selection_state["selected_rank"]:
            _apply_selection(None, None)   # même région : désélection
        else:
            _apply_selection(rank_val, dataset)

    plotter.iren.interactor.AddObserver("LeftButtonPressEvent", on_press)
    plotter.iren.interactor.AddObserver("EndInteractionEvent", on_release)


def _setup_panel_sync(plotter, atlas_subplot, e_subplot, cb_atlas, cb_E,
                       atlas_sliders, e_sliders, atlas_clip_state, e_clip_state,
                       sync_state, button_subplot, button_pos, e_sync_ctx=None):
    """
    Ajoute une case à cocher permettant de synchroniser les deux panneaux
    (atlas et champ E) : caméra (zoom, rotation, pan) et sliders de
    découpe (X/Y/Z). Tant que la case est cochée, toute manipulation d'un
    panneau (rotation/zoom ou déplacement d'un slider) est reproduite sur
    l'autre. Au moment de l'activation, le panneau champ E est aligné sur
    l'état courant du panneau atlas.

    `e_sync_ctx`, if provided, also drives the region-dim on the E panel:
    enabling sync while a region is selected dims the E panel immediately
    (rest at opacity 0.1, selected region at full opacity); disabling sync
    restores the E panel to full opacity right away.
    """
    n_cols = plotter.shape[1]
    atlas_flat = atlas_subplot[0] * n_cols + atlas_subplot[1]
    e_flat = e_subplot[0] * n_cols + e_subplot[1]
    atlas_renderer = plotter.renderers[atlas_flat]
    e_renderer = plotter.renderers[e_flat]

    def on_toggle_sync(flag):
        sync_state["linked"] = flag
        if flag:
            # Le panneau champ E adopte l'état courant du panneau atlas :
            # mêmes plans de découpe...
            for axis, val in atlas_clip_state.items():
                if val is None:
                    continue
                e_clip_state[axis] = val
                cb_E(val, axis)
                e_sliders[axis].GetRepresentation().SetValue(val)
            # ...et la même caméra (objet partagé : tout mouvement sur l'un
            # des deux panneaux se répercute désormais instantanément sur l'autre).
            e_renderer.camera = atlas_renderer.camera
        else:
            # Décorrélation : le panneau champ E garde sa vue actuelle
            # (copie figée de la caméra partagée) au lieu d'en changer.
            frozen_camera = pv.Camera()
            frozen_camera.DeepCopy(atlas_renderer.camera)
            e_renderer.camera = frozen_camera

        if e_sync_ctx is not None:
            rank_val = e_sync_ctx["selection_state"]["selected_rank"]
            region_id = None
            if flag and rank_val is not None:
                region_id = e_sync_ctx["region_ids"][rank_val - 1]
            e_sync_ctx["set_dim"](region_id)

        plotter.render()

    plotter.subplot(*button_subplot)
    button_x, button_y, button_size = button_pos
    plotter.add_checkbox_button_widget(on_toggle_sync, value=False,
                                        position=(button_x, button_y),
                                        size=button_size,
                                        color_on="lightskyblue", color_off="grey",
                                        background_color="white")
    plotter.add_text("Sync panels", name="sync_toggle_label", font_size=9,
                      position=(button_x + button_size + 8, button_y + 2),
                      color="white", shadow=True)


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
                      show_scalar_bar=False, pickable=False, name="E_mesh")
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

    win_w, win_h = plotter.window_size
    _setup_panel_sync(plotter, (0, 0), (0, 1), cb_atlas, cb_E,
                       atlas_sliders, e_sliders, atlas_clip_state, e_clip_state,
                       sync_state, button_subplot=(0, 1),
                       button_pos=(20 , win_h - 55, 20), e_sync_ctx=e_sync_ctx)

    plotter.show()
    return plotter