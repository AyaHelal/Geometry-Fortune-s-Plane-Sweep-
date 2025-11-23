from copy import copy
from decimal import Decimal
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
from matplotlib import patches
import random
from matplotlib import colors as mcolors

from foronoi import Coordinate
from foronoi.algorithm import Algorithm
from foronoi.events import CircleEvent
import matplotlib.pyplot as plt


class Colors:
    SWEEP_LINE = "#966fd6"
    VERTICES = "#34495e"
    BEACH_LINE = "#f1c40f"
    EDGE = "#636e72"
    ARC = "#95a5a6"
    INCIDENT_POINT_POINTER = "#ecf0f1"
    INVALID_CIRCLE = "#ecf0f1"  # red
    VALID_CIRCLE = "#2980b9"  # blue
    CELL_POINTS = "#bdc3c7"  # blue
    TRIANGLE = "#e67e22"  # orange
    BOUNDING_BOX = "#000000"  # black
    TEXT = "#00cec9"  # green
    HELPER = "#ff0000"
    HIGH_LIGHT = "#00ff00"
    EDGE_DIRECTION = "#fdcb6e"
    FIRST_EDGE = "#2ecc71"


class Presets:
    # A minimalistic preset that is useful during construction
    construction = dict(polygon=True, events=True, beach_line=True, sweep_line=True)

    # A minimalistic preset that is useful during clipping
    clipping = dict(polygon=True)

    # A minimalistic preset that is useful to show the final result
    final = dict()


def build_palette(n, seed=42):
    """Return list of n hex colors. Deterministic by seed."""
    random.seed(seed)
    palette = []
    for i in range(n):
        # avoid very dark near-black and very light near-white to keep contrast
        r = random.randint(30, 225)
        g = random.randint(30, 225)
        b = random.randint(30, 225)
        palette.append(mcolors.to_hex((r / 255, g / 255, b / 255)))
    return palette


class Visualizer:
    """
    Visualizer
    """

    def __init__(self, voronoi, canvas_offset=1, figsize=(8, 8), palette_seed=42):
        """
        Visualizer which can also color each Voronoi cell.

        Parameters
        ----------
        voronoi: Voronoi
            The voronoi object
        canvas_offset: float
            The space around the bounding object
        figsize: float, float
            Width, height in inches
        palette_seed: int
            Seed used to generate a deterministic palette for site colors
        """
        self.voronoi = voronoi
        self.min_x, self.max_x, self.min_y, self.max_y = self._canvas_size(voronoi.bounding_poly, canvas_offset)
        plt.close("all")  # Prevents previous created plots from showing up
        fig, ax = plt.subplots(figsize=figsize)
        self.canvas = ax

        # prepare a palette mapped to sites (built later on first use)
        self._palette_seed = palette_seed
        self._site_color_map = None  # {site_obj: color_hex}

    def _set_limits(self):
        self.canvas.set_ylim(self.min_y, self.max_y)
        self.canvas.set_xlim(self.min_x, self.max_x)
        return self

    def get_canvas(self):
        """
        Retrieve the current matplotlib Figure.
        """
        self._set_limits()
        return self.canvas.figure

    def show(self, block=True, **kwargs):
        """
        Display all open figures (wrapper around matplotlib.pyplot.show).

        Parameters
        ----------
        block : bool, optional
            If True, block and run the GUI main loop until all windows are closed.
            If False, ensure that all windows are displayed and return immediately.

        kwargs are forwarded to matplotlib.pyplot.show.
        """
        self._set_limits()
        plt.show(block=block, **kwargs)
        return self

    def _ensure_palette(self):
        """Build a deterministic palette mapping each site to a color."""
        if self._site_color_map is not None:
            return
        sites = list(self.voronoi.sites)
        n = len(sites)
        palette = build_palette(n, seed=self._palette_seed)
        self._site_color_map = {}
        for i, s in enumerate(sites):
            # map by site identity
            self._site_color_map[s] = palette[i]

    def plot_all(self, polygon=False, edges=True, vertices=True, sites=True, cells=False,
                 outgoing_edges=False, border_to_site=False, scale=1,
                 edge_labels=False, site_labels=False, triangles=False, arcs=False, sweep_line=False, events=False,
                 arc_labels=False, beach_line=False):
        """
        Convenience method that calls other methods to display parts of the diagram.
        Added `cells` option to fill and color each Voronoi cell.
        """
        self.plot_sweep_line() if sweep_line else False
        self.plot_polygon() if polygon else False
        self.plot_edges(show_labels=edge_labels) if edges else False
        self.plot_border_to_site() if border_to_site else False
        self.plot_vertices() if vertices else False
        # draw cells before sites so sites are visible on top
        self.plot_cells() if cells else False
        self.plot_sites(show_labels=site_labels) if sites else False
        self.plot_outgoing_edges(scale=scale) if outgoing_edges else False
        self.plot_event(triangles) if events else False
        self.plot_arcs(plot_arcs=arcs, show_labels=arc_labels) if beach_line else False
        self._set_limits()
        return self

    def plot_polygon(self):
        """
        Display the polygon outline.
        """
        if hasattr(self.voronoi.bounding_poly, 'radius'):
            # Draw bounding box as circle
            self.canvas.add_patch(
                patches.Circle((self.voronoi.bounding_poly.xd, self.voronoi.bounding_poly.xd),
                               self.voronoi.bounding_poly.radius,
                               fill=False,
                               edgecolor=Colors.BOUNDING_BOX)
            )
        else:
            # Draw bounding box polygon
            self.canvas.add_patch(
                patches.Polygon(self.voronoi.bounding_poly.get_coordinates(), fill=False, edgecolor=Colors.BOUNDING_BOX)
            )

        return self

    def plot_vertices(self, vertices=None, **kwargs):
        vertices = vertices or self.voronoi.vertices
        xs = [vertex.xd for vertex in vertices]
        ys = [vertex.yd for vertex in vertices]
        self.canvas.scatter(xs, ys, s=50, color=Colors.VERTICES, zorder=10, **kwargs)
        return self

    def plot_outgoing_edges(self, vertices=None, scale=0.5, **kwargs):
        vertices = vertices or self.voronoi.vertices
        scale = Decimal(str(scale))
        for vertex in vertices:
            for edge in vertex.connected_edges:
                start, end = self._origins(edge, None)
                if start is None or end is None:
                    continue
                x_diff = end.xd - start.xd
                y_diff = end.yd - start.yd
                length = Decimal.sqrt(x_diff ** 2 + y_diff ** 2)
                if length == 0:
                    continue
                direction = (x_diff / length, y_diff / length)
                new_end = Coordinate(start.xd + direction[0] * scale, start.yd + direction[1] * scale)
                props = dict(arrowstyle="->", color=Colors.EDGE_DIRECTION, linewidth=3, **kwargs)
                self.canvas.annotate(text='', xy=(new_end.xd, new_end.yd), xytext=(start.xd, start.yd),
                                     arrowprops=props)
        return self

    def plot_sites(self, points=None, show_labels=True, color=None, zorder=15):
        """
        Display the cell points (a.k.a. sites). If palette exists, use site-specific color.
        """
        points = points or list(self.voronoi.sites)
        # ensure palette built
        self._ensure_palette()
        xs = [point.xd for point in points]
        ys = [point.yd for point in points]
        # determine colors list matching points order
        colors = []
        for p in points:
            if color is not None:
                colors.append(color)
            else:
                colors.append(self._site_color_map.get(p, Colors.CELL_POINTS))
        # scatter with per-point colors
        self.canvas.scatter(xs, ys, s=80, color=colors, zorder=zorder, edgecolors="k")
        if show_labels:
            for point in points:
                self.canvas.text(point.xd, point.yd, s=f"P{point.name if point.name is not None else ''}", zorder=20)
        return self

    def plot_edges(self, edges=None, sweep_line=None, show_labels=True, color=Colors.EDGE, **kwargs):
        edges = edges or self.voronoi.edges
        sweep_line = sweep_line or self.voronoi.sweep_line
        for edge in edges:
            self._plot_edge(edge, sweep_line, show_labels, color)
            self._plot_edge(edge.twin, sweep_line, print_name=False, color=color)
        return self

    def plot_border_to_site(self, edges=None, sweep_line=None):
        edges = edges or self.voronoi.edges
        sweep_line = sweep_line or self.voronoi.sweep_line
        for edge in edges:
            self._draw_line_from_edge_midpoint_to_incident_point(edge, sweep_line)
            self._draw_line_from_edge_midpoint_to_incident_point(edge.twin, sweep_line)
        return self

    def plot_arcs(self, arcs=None, sweep_line=None, plot_arcs=False, show_labels=True):
        arcs = arcs or self.voronoi.arcs
        sweep_line = sweep_line or self.voronoi.sweep_line
        min_x, max_x, min_y, max_y = self.min_x, self.max_x, self.min_y, self.max_y
        sweep_line = max_y if sweep_line is None else sweep_line
        x = np.linspace(float(min_x), float(max_x), 1000)
        plot_lines = []
        for arc in arcs:
            plot_line = arc.get_plot(x, sweep_line)
            if plot_line is None:
                if plot_arcs:
                    self.canvas.axvline(x=arc.origin.xd, color=Colors.SWEEP_LINE)
            else:
                if plot_arcs:
                    self.canvas.plot(x, plot_line, linestyle="--", color=Colors.ARC)
                plot_lines.append(plot_line)
        if len(plot_lines) > 0:
            bottom = np.min(plot_lines, axis=0)
            self.canvas.plot(x, bottom, color=Colors.BEACH_LINE)
            if show_labels:
                self._plot_arc_labels(x, plot_lines, bottom, sweep_line, arcs)
        return self

    def _plot_arc_labels(self, x, plot_lines, bottom, sweep_line, arcs):
        indices = np.nanargmin(plot_lines, axis=0)
        unique_indices = np.unique(indices)
        for index in unique_indices:
            x_mean = np.nanmedian(x[(indices == index) & (bottom < self.max_y)])
            y = arcs[index].get_plot(x_mean, sweep_line)
            self.canvas.text(x_mean, y, s=f"{arcs[index].origin.name}", size=12, color=Colors.VALID_CIRCLE, zorder=15)
        return self

    def plot_sweep_line(self, sweep_line=None):
        sweep_line = sweep_line or self.voronoi.sweep_line
        min_x, max_x, min_y, max_y = self.min_x, self.max_x, self.min_y, self.max_y
        self.canvas.plot([min_x, max_x], [sweep_line, sweep_line], color=Colors.SWEEP_LINE)
        return self

    def plot_event(self, event=None, triangles=False):
        event = event or self.voronoi.event
        if isinstance(event, CircleEvent):
            self._plot_circle(event, show_triangle=triangles)
        return self

    def _plot_circle(self, evt, show_triangle=False):
        x, y = evt.center.xd, evt.center.yd
        radius = evt.radius
        color = Colors.VALID_CIRCLE if evt.is_valid else Colors.INVALID_CIRCLE
        circle = plt.Circle((x, y), radius, fill=False, color=color, linewidth=2)
        self.canvas.add_artist(circle)
        if show_triangle:
            triangle = plt.Polygon(evt._get_triangle(), fill=False, color=Colors.TRIANGLE, linewidth=1)
            self.canvas.add_artist(triangle)
        points = evt.point_triple
        self.plot_sites(points, color=Colors.VALID_CIRCLE, show_labels=False, zorder=15)
        return self

    def _plot_edge(self, edge, sweep_line=None, print_name=True, color=Colors.EDGE, **kwargs):
        start, end = self._origins(edge, sweep_line)
        if not (start and end):
            return self
        self.canvas.plot([start.xd, end.xd], [start.yd, end.yd], color)
        if print_name:
            self.canvas.annotate(text=str(edge),
                                 xy=((end.xd + start.xd) / 2, (end.yd + start.yd) / 2),
                                 **kwargs)
        return self

    def _draw_line_from_edge_midpoint_to_incident_point(self, edge, sweep_line=None):
        start, end = self._origins(edge, sweep_line)
        is_first_edge = edge.incident_point is not None and edge.incident_point.first_edge == edge
        incident_point = edge.incident_point
        if start and end and incident_point:
            self.canvas.plot(
                [(start.xd + end.xd) / 2, incident_point.xd],
                [(start.yd + end.yd) / 2, incident_point.yd],
                color=Colors.FIRST_EDGE if is_first_edge else Colors.INCIDENT_POINT_POINTER,
                linestyle="--"
            )
        return self.canvas

    def _origins(self, edge, sweep_line=None):
        # Get axis limits
        max_y = self.max_y
        # Get start and end of edges
        start = edge.get_origin(sweep_line, max_y)
        end = edge.twin.get_origin(sweep_line, max_y)
        return start, end

    def plot_cells(self, alpha=0.95, resolution=400, chunk_size=None):
        """
        Fully float-based Voronoi cell raster coloring.
        Eliminates ALL Decimal operations to avoid errors.
        """

        import numpy as np
        from matplotlib import colors as mcolors

        # -------- 1) Ensure palette --------
        self._ensure_palette()

        # -------- 2) Float bounding box --------
        min_x = float(self.min_x)
        max_x = float(self.max_x)
        min_y = float(self.min_y)
        max_y = float(self.max_y)

        width = max_x - min_x
        height = max_y - min_y

        if width <= 0 or height <= 0:
            return self

        # -------- 3) Raster resolution --------
        if width >= height:
            W = int(resolution)
            H = max(1, int(round(resolution * (height / width))))
        else:
            H = int(resolution)
            W = max(1, int(round(resolution * (width / height))))

        # -------- 4) Build pixel grid --------
        xs = np.linspace(min_x, max_x, W, dtype=float)
        ys = np.linspace(min_y, max_y, H, dtype=float)
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)   # (H*W, 2), float

        # -------- 5) Convert Voronoi sites to pure float --------
        sites = list(self.voronoi.sites)
        coords = np.array([[float(s.xd), float(s.yd)] for s in sites], dtype=float)  # (N,2)

        N = coords.shape[0]
        P = pts.shape[0]

        # -------- 6) Find nearest site (chunking optional) --------
        if chunk_size is None:
            d2 = np.sum((pts[:, None, :] - coords[None, :, :])**2, axis=2)  # float
            nearest = np.argmin(d2, axis=1)
        else:
            nearest = np.empty(P, dtype=np.int32)
            for start in range(0, P, chunk_size):
                end = min(start + chunk_size, P)
                sub = pts[start:end]
                d2_sub = np.sum((sub[:, None, :] - coords[None, :, :])**2, axis=2)
                nearest[start:end] = np.argmin(d2_sub, axis=1)

        # -------- 7) Build RGB map for each site --------
        site_colors = []
        for s in sites:
            hex_color = self._site_color_map.get(s)
            rgb = mcolors.to_rgb(hex_color)
            rgb255 = tuple(int(round(255 * v)) for v in rgb)
            site_colors.append(rgb255)

        site_colors = np.array(site_colors, dtype=np.uint8)

        # -------- 8) Generate final image --------
        img = site_colors[nearest].reshape((H, W, 3))

        extent = (min_x, max_x, min_y, max_y)

        self.canvas.imshow(
            img,
            origin='lower',
            extent=extent,
            alpha=alpha,
            interpolation='bilinear',
            zorder=0
        )

        return self



    @staticmethod
    def _canvas_size(bounding_polygon, offset):
        max_y = bounding_polygon.max_y + offset
        max_x = bounding_polygon.max_x + offset
        min_x = bounding_polygon.min_x - offset
        min_y = bounding_polygon.min_y - offset
        return min_x, max_x, min_y, max_y
