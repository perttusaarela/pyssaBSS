import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .spatial import points_in_polygon


class PolygonDrawer:
    """
    Polygon drawer

    This can be used to construct polygonal partitions of data points
    on a given map. The partition can then be saved a JSON file and
    used for the SPSSA methods.

    Usage
    -----
    1. Initialize:
        drawer = PolygonDrawer(ax, points)
    2. Draw polygons:
        plt.show()
    3. Save polygons:
        drawer.save("polygons.json")

    Parameters
    ----------
    ax : an ax object from pyplot
        this is the map on which we draw. The drawer assumes that this is 
        already constructed (e.g. from png and a bounding box)
    points : ndarray of shape (n, 2)
        the coordinates of the points we want to partition.
        The points should lie inside the bounding box of the ax figure.
    """

    SNAP_RADIUS_PX = 10  # pixels to snap-close polygon
    MOVE_THRESHOLD = 2   # pixels of mouse movement before recomputing

    def __init__(self, ax, points, filename="polygons.json"):
        self.ax = ax
        self.fig = ax.figure
        self.points = np.asarray(points)
        self.filename = filename

        # Precompute point bounding box for fast rejection
        self.pt_xmin, self.pt_ymin = self.points.min(axis=0)
        self.pt_xmax, self.pt_ymax = self.points.max(axis=0)

        self.current = []
        self.polygons = []
        self.polygon_colors = []
        self.finished_lines = []

        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_idx = 0

        # Mouse state for throttling
        self._last_mouse = (None, None)
        self._last_count = 0

        # Plot elements
        self.line, = ax.plot([], [], "ro-", zorder=5)
        self.preview_line, = ax.plot([], [], "r-",  linewidth=1, zorder=4)
        self.closing_line, = ax.plot([], [], "r--", linewidth=1, zorder=4)
        self.snap_marker, = ax.plot([], [], "go", markersize=10, zorder=6)

        self.hud = self.ax.text(
            0.02, 0.98, "",
            transform=self.ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )
        self.help_text = self.ax.text(
            0.98, 0.98,
            "click=add  right-click/n=close\n"
            "backspace=undo vertex  ctrl+z/d=undo poly\n"
            "w=save  q=quit  h=toggle help",
            transform=self.ax.transAxes,
            va="top", ha="right", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            visible=True
        )

        for event, handler in [
            ("button_press_event", self.on_click),
            ("key_press_event",    self.on_key),
            ("scroll_event",       self.on_scroll),
            ("motion_notify_event",self.on_move),
        ]:
            self.fig.canvas.mpl_connect(event, handler)

    # ------------------------------------------------------------------ #
    #  Events                                                              #
    # ------------------------------------------------------------------ #

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            # Snap-to-close if near first vertex
            if self._near_first_vertex(event.xdata, event.ydata) and len(self.current) >= 3:
                self._close_polygon()
            else:
                self.current.append((event.xdata, event.ydata))
                self._update_current()
        elif event.button == 3 and len(self.current) >= 3:
            self._close_polygon()

    def on_key(self, event):
        if event.key == "n":
            if len(self.current) >= 3:
                self._close_polygon()
            else:
                self.current = []
                self._update_current()
        elif event.key in ("d", "ctrl+z"):
            self._delete_last_polygon()
        elif event.key == "backspace":
            self._delete_last_vertex()
        elif event.key == "w":
            self.save(self.filename)
            print(f"Saved to {self.filename}")
        elif event.key == "h":
            self.help_text.set_visible(not self.help_text.get_visible())
            self.fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(self.fig)

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        xdata, ydata = event.xdata, event.ydata
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        self.ax.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
        self.ax.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
        self.fig.canvas.draw_idle()

    def on_move(self, event):
        if event.inaxes != self.ax or not self.current:
            self.preview_line.set_data([], [])
            self.closing_line.set_data([], [])
            self.snap_marker.set_data([], [])
            self.hud.set_text("")
            self.fig.canvas.draw_idle()
            return

        mx, my = event.xdata, event.ydata

        # Snap indicator
        if self._near_first_vertex(mx, my) and len(self.current) >= 3:
            self.snap_marker.set_data([self.current[0][0]], [self.current[0][1]])
        else:
            self.snap_marker.set_data([], [])

        # Preview edge from last vertex to mouse
        x_last, y_last = self.current[-1]
        self.preview_line.set_data([x_last, mx], [y_last, my])

        if len(self.current) >= 2:
            x_first, y_first = self.current[0]
            self.closing_line.set_data([mx, x_first], [my, y_first])

            # Throttle expensive count by movement threshold
            lx, ly = self._last_mouse
            if lx is None or self._pixel_dist(lx, ly, mx, my) > self.MOVE_THRESHOLD:
                temp_poly = np.array(self.current + [(mx, my)])
                self._last_count = self._count_inside(temp_poly)
                self._last_mouse = (mx, my)

            self.hud.set_text(
                f"Vertices: {len(self.current)}  |  Points inside: {self._last_count}"
            )
        else:
            self.closing_line.set_data([], [])
            self.hud.set_text(f"Vertices: {len(self.current)}")

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _count_inside(self, poly):
        """Count points inside polygon with bounding-box pre-filter."""
        px, py = poly[:, 0], poly[:, 1]
        bb_mask = (
            (self.points[:, 0] >= px.min()) & (self.points[:, 0] <= px.max()) &
            (self.points[:, 1] >= py.min()) & (self.points[:, 1] <= py.max())
        )
        candidates = self.points[bb_mask]
        if candidates.size == 0:
            return 0
        return points_in_polygon(candidates, poly).sum()

    def _near_first_vertex(self, mx, my):
        """Check if mouse is within SNAP_RADIUS_PX pixels of the first vertex."""
        if not self.current:
            return False
        x0, y0 = self.current[0]
        # Convert data coords to display coords for pixel comparison
        disp = self.ax.transData.transform([(x0, y0), (mx, my)])
        dx, dy = disp[1] - disp[0]
        return (dx**2 + dy**2) ** 0.5 < self.SNAP_RADIUS_PX

    def _pixel_dist(self, x0, y0, x1, y1):
        """Mouse movement in pixels."""
        disp = self.ax.transData.transform([(x0, y0), (x1, y1)])
        dx, dy = disp[1] - disp[0]
        return (dx**2 + dy**2) ** 0.5

    def _close_polygon(self):
        poly = np.array(self.current + [self.current[0]])
        self.polygons.append(poly)
        color = self.colors[self.color_idx % len(self.colors)]
        self.color_idx += 1
        self.polygon_colors.append(color)
        line, = self.ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
        self.finished_lines.append(line)
        self.current = []
        self._last_mouse = (None, None)
        self.preview_line.set_data([], [])
        self.closing_line.set_data([], [])
        self.snap_marker.set_data([], [])
        self.hud.set_text("")
        self._update_current()

    def _update_current(self):
        if self.current:
            xs, ys = zip(*self.current)
        else:
            xs, ys = [], []
        self.line.set_data(xs, ys)
        if not self.current:
            self.preview_line.set_data([], [])
            self.closing_line.set_data([], [])
            self.hud.set_text("")
        self.fig.canvas.draw_idle()

    def _delete_last_vertex(self):
        if self.current:
            self.current.pop()
            self._last_mouse = (None, None)
            self._update_current()

    def _delete_last_polygon(self):
        if self.polygons:
            self.polygons.pop()
            self.polygon_colors.pop()
            self.finished_lines.pop().remove()
            self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def save(self, filename):
        data = [
            {"vertices": poly.tolist(), "color": color}
            for poly, color in zip(self.polygons, self.polygon_colors)
        ]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def get_polygons(self):
        return self.polygons

