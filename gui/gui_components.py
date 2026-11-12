import tkinter as tk
from tkinter import ttk
from core.log import get_logger

logger = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use('TkAgg') 
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.dates as mdates
    import datetime as dt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TrafficLightWidget(tk.Canvas):
    """
    Vertical 3-bulb signal head with halo glow on the active aspect.
    API unchanged: TrafficLightWidget(parent).set_state(state, is_manually_red).
    """
    WIDTH, HEIGHT = 40, 96

    def __init__(self, parent, **kwargs):
        kwargs.setdefault('width', self.WIDTH)
        kwargs.setdefault('height', self.HEIGHT)
        from gui.theme import BG_DARK
        kwargs.setdefault('bg', BG_DARK)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)

        # Housing box (matte dark casing, slightly rounded look via inset)
        self.housing = self.create_rectangle(3, 3, 37, 93, fill="#1E1E22", outline="#3A3A3C", width=2)

        # Halo rings (drawn under bulbs for glow) + bulbs
        self.red_halo = self.create_oval(7, 7, 33, 33, fill="", outline="", width=2)
        self.red_bulb = self.create_oval(10, 10, 30, 30, fill="#4D0F0F", outline="#2C2C2E", width=1)
        self.yellow_halo = self.create_oval(7, 35, 33, 61, fill="", outline="", width=2)
        self.yellow_bulb = self.create_oval(10, 38, 30, 58, fill="#3D2E02", outline="#2C2C2E", width=1)
        self.green_halo = self.create_oval(7, 63, 33, 89, fill="", outline="", width=2)
        self.green_bulb = self.create_oval(10, 66, 30, 86, fill="#08260D", outline="#2C2C2E", width=1)
        
    def set_state(self, state, is_manually_red=False):
        """Update active/dim color values based on state and manual override status."""
        state = str(state or "RED").upper()
        cache_key = (state, bool(is_manually_red))
        if cache_key == getattr(self, "_last_state", None):
            return
        self._last_state = cache_key
        if is_manually_red:
            self.itemconfig(self.housing, outline="#FF453A", width=2)
        else:
            self.itemconfig(self.housing, outline="#3A3A3C", width=2)

        # Reset halos
        self.itemconfig(self.red_halo, outline="")
        self.itemconfig(self.yellow_halo, outline="")
        self.itemconfig(self.green_halo, outline="")
        if state in ['RED', 'ALL_RED']:
            self.itemconfig(self.red_halo, outline="#FF453A")
            self.itemconfig(self.red_bulb, fill="#FF453A", outline="#FF9F0A", width=2)
            self.itemconfig(self.yellow_bulb, fill="#3D2E02", outline="#2C2C2E", width=1)
            self.itemconfig(self.green_bulb, fill="#08260D", outline="#2C2C2E", width=1)
        elif state == 'YELLOW':
            self.itemconfig(self.red_bulb, fill="#4D0F0F", outline="#2C2C2E", width=1)
            self.itemconfig(self.yellow_halo, outline="#FFD60A")
            self.itemconfig(self.yellow_bulb, fill="#FFD60A", outline="#FFE600", width=2)
            self.itemconfig(self.green_bulb, fill="#08260D", outline="#2C2C2E", width=1)
        elif state == 'GREEN':
            self.itemconfig(self.red_bulb, fill="#4D0F0F", outline="#2C2C2E", width=1)
            self.itemconfig(self.yellow_bulb, fill="#3D2E02", outline="#2C2C2E", width=1)
            self.itemconfig(self.green_halo, outline="#30D158")
            self.itemconfig(self.green_bulb, fill="#30D158", outline="#A6F4C5", width=2)
        else:
            # Fallback dim colors
            self.itemconfig(self.red_bulb, fill="#4D0F0F", outline="#2C2C2E", width=1)
            self.itemconfig(self.yellow_bulb, fill="#3D2E02", outline="#2C2C2E", width=1)
            self.itemconfig(self.green_bulb, fill="#08260D", outline="#2C2C2E", width=1)


class TrendPlotWidget:
    """
    A custom wrapper for Matplotlib Figure and Canvas in Tkinter.
    Encapsulates figure creation, dark styling, filled curve area, and live update loop.
    """
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.fig = None
        self.ax = None
        self.canvas = None
        self.canvas_widget = None
        
        if MATPLOTLIB_AVAILABLE:
            try:
                from gui.theme import BG_CARD, BG_INNER, FG_LIGHT, FG_MUTED, BORDER
                self.fig = Figure(figsize=(4.4, 2.4), dpi=90, facecolor=BG_CARD)
                self.ax = self.fig.add_subplot(111)
                self.ax.set_facecolor(BG_INNER)
                self.ax.set_title("Demand — last 60s", fontsize=9, color=FG_LIGHT, pad=6, loc="left")
                self.ax.set_ylabel("Vehicles", fontsize=8, color=FG_LIGHT)
                self.ax.tick_params(axis='both', which='major', labelsize=7, colors=FG_MUTED, pad=2)
                self.ax.grid(True, linestyle='--', alpha=0.3, color=FG_MUTED, linewidth=0.7)
                for spine in self.ax.spines.values():
                    spine.set_color(BORDER)
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                self.fig.autofmt_xdate(rotation=25, ha='right')
                self.fig.tight_layout(pad=1.0)

                self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
                self.canvas_widget = self.canvas.get_tk_widget()
                self.canvas_widget.pack(fill=tk.BOTH, expand=True)
                self.canvas.draw()
            except Exception as e:
                logger.warning("TrendPlot init failed: %s", e)
                
    def update_trend(self, history_deque, history_cutoff_time, current_time, demand_threshold=None):
        if not MATPLOTLIB_AVAILABLE or self.ax is None or self.fig is None or self.canvas is None:
            return
        try:
            plot_data = [(dt.datetime.fromtimestamp(ts), count) for ts, count in list(history_deque) if ts >= history_cutoff_time]
            self.ax.clear()
            self.ax.set_facecolor('#25252B')

            from gui.theme import BG_INNER, FG_LIGHT, FG_MUTED, BORDER, ACCENT, ORANGE
            if plot_data:
                timestamps, counts = zip(*plot_data)
                # Area fill under the trendline
                self.ax.fill_between(timestamps, counts, color=ACCENT, alpha=0.18, step="mid")
                self.ax.plot(timestamps, counts, marker='o', linestyle='-', markersize=4,
                             markerfacecolor=ACCENT, markeredgecolor="white", markeredgewidth=0.6,
                             color=ACCENT, linewidth=2.0)

                if demand_threshold is not None:
                    self.ax.axhline(y=demand_threshold, color=ORANGE, linestyle='--',
                                    linewidth=1.2, alpha=0.9)
                    self.ax.text(0.98, 0.92, f"threshold {demand_threshold:g}",
                                 transform=self.ax.transAxes, ha="right", va="top",
                                 fontsize=7, color=ORANGE,
                                 bbox=dict(facecolor=BG_INNER, edgecolor=BORDER, boxstyle="round,pad=0.3"))

                self.ax.set_ylim(bottom=-0.5)
                max_count = max(counts) if counts else 0
                self.ax.set_ylim(top=max(max_count * 1.3, 5))
                last = counts[-1]
                self.ax.text(0.02, 0.92, f"now {last:g}",
                             transform=self.ax.transAxes, ha="left", va="top",
                             fontsize=8, weight="bold", color=FG_LIGHT,
                             bbox=dict(facecolor=BG_INNER, edgecolor=BORDER, boxstyle="round,pad=0.3"))
            else:
                now_dt = dt.datetime.fromtimestamp(current_time)
                then_dt = dt.datetime.fromtimestamp(history_cutoff_time)
                self.ax.plot([then_dt, now_dt], [0, 0], color='grey', alpha=0)
                self.ax.set_ylim(0, 5)
                self.ax.text(0.5, 0.55, 'Waiting for detections…', ha='center', va='center',
                             transform=self.ax.transAxes, fontsize=9, color=FG_MUTED)
                self.ax.text(0.5, 0.38, 'Trend appears once frames arrive', ha='center', va='center',
                             transform=self.ax.transAxes, fontsize=7, color=FG_MUTED)

            self.ax.set_title("Demand — last 60s", fontsize=9, pad=6, color=FG_LIGHT, loc="left")
            self.ax.set_ylabel("Vehicles", fontsize=8, color=FG_LIGHT)
            self.ax.tick_params(axis='x', labelsize=7, colors=FG_MUTED, pad=2)
            self.ax.tick_params(axis='y', labelsize=7, colors=FG_MUTED)
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            self.ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.35, color=FG_MUTED)
            for spine in self.ax.spines.values():
                spine.set_color(BORDER)
            self.fig.tight_layout(pad=1.0)
            self.canvas.draw_idle()
        except Exception as e:
            logger.debug("Trend update skipped: %s", e)
