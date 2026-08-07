import tkinter as tk
from tkinter import ttk
import time
import os
from core import config

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
    A custom vertical 3-bulb traffic light box.
    Encapsulates canvas drawing and color toggling logic.
    """
    def __init__(self, parent, **kwargs):
        # We enforce width=28, height=72 by default
        kwargs.setdefault('width', 28)
        kwargs.setdefault('height', 72)
        kwargs.setdefault('bg', "#121214")
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)
        
        # Housing box (matte black casing)
        self.create_rectangle(2, 2, 26, 70, fill="#2C2C2E", outline="#48484A", width=2)
        
        # Create bulb items (Red, Yellow, Green)
        self.red_bulb = self.create_oval(6, 6, 22, 22, fill="#5C1212", outline="#3A3A3C", width=1)
        self.yellow_bulb = self.create_oval(6, 28, 22, 44, fill="#4D3A03", outline="#3A3A3C", width=1)
        self.green_bulb = self.create_oval(6, 50, 22, 66, fill="#0B3012", outline="#3A3A3C", width=1)
        
    def set_state(self, state):
        """Update active/dim color values based on state."""
        if state in ['RED', 'ALL_RED']:
            self.itemconfig(self.red_bulb, fill="#FF453A", outline="#FF9F0A")
            self.itemconfig(self.yellow_bulb, fill="#4D3A03", outline="#3A3A3C")
            self.itemconfig(self.green_bulb, fill="#0B3012", outline="#3A3A3C")
        elif state == 'YELLOW':
            self.itemconfig(self.red_bulb, fill="#5C1212", outline="#3A3A3C")
            self.itemconfig(self.yellow_bulb, fill="#FFD60A", outline="#FFE600")
            self.itemconfig(self.green_bulb, fill="#0B3012", outline="#3A3A3C")
        elif state == 'GREEN':
            self.itemconfig(self.red_bulb, fill="#5C1212", outline="#3A3A3C")
            self.itemconfig(self.yellow_bulb, fill="#4D3A03", outline="#3A3A3C")
            self.itemconfig(self.green_bulb, fill="#30D158", outline="#66FF66")
        else:
            # Fallback dim colors
            self.itemconfig(self.red_bulb, fill="#5C1212", outline="#3A3A3C")
            self.itemconfig(self.yellow_bulb, fill="#4D3A03", outline="#3A3A3C")
            self.itemconfig(self.green_bulb, fill="#0B3012", outline="#3A3A3C")

class TrendPlotWidget:
    """
    A custom wrapper for Matplotlib Figure and Canvas in Tkinter.
    Encapsulates figure creation, dark styling, and update loop.
    """
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.fig = None
        self.ax = None
        self.canvas = None
        self.canvas_widget = None
        
        if MATPLOTLIB_AVAILABLE:
            try:
                self.fig = Figure(figsize=(4, 2.2), dpi=85, facecolor='#1A1A1E') 
                self.ax = self.fig.add_subplot(111)
                self.ax.set_facecolor('#25252B')
                self.ax.set_title("Recent Count Trend", fontsize=8, color='#E2E2E6')
                self.ax.set_ylabel("Count", fontsize=7, color='#E2E2E6')
                self.ax.tick_params(axis='both', which='major', labelsize=6, colors='#8E8E93')
                self.ax.grid(True, linestyle='--', alpha=0.3, color='#8E8E93')
                for spine in self.ax.spines.values():
                    spine.set_color('#3A3A3C')
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                self.fig.autofmt_xdate(rotation=30, ha='right')
                self.fig.tight_layout(pad=0.8)

                self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
                self.canvas_widget = self.canvas.get_tk_widget()
                self.canvas_widget.pack(fill=tk.BOTH, expand=True)
                self.canvas.draw()
            except Exception as e:
                print(f"[TrendPlotWidget] Error initializing plot: {e}")
                
    def update_trend(self, history_deque, history_cutoff_time, current_time):
        if not MATPLOTLIB_AVAILABLE or self.ax is None:
            return
            
        plot_data = [(dt.datetime.fromtimestamp(ts), count) for ts, count in list(history_deque) if ts >= history_cutoff_time]
        self.ax.clear()
        self.ax.set_facecolor('#25252B')
        
        if plot_data:
            timestamps, counts = zip(*plot_data)
            self.ax.plot(timestamps, counts, marker='.', linestyle='-', markersize=3, color='#0A84FF')
            self.ax.set_ylim(bottom=-0.5) 
            max_count = max(counts) if counts else 0
            self.ax.set_ylim(top=max(max_count * 1.2, 5)) 
        else:
            now_dt = dt.datetime.fromtimestamp(current_time)
            then_dt = dt.datetime.fromtimestamp(history_cutoff_time)
            self.ax.plot([then_dt, now_dt], [0, 0], color='grey', alpha=0) 
            self.ax.set_ylim(0, 5)
            self.ax.text(0.5, 0.5, 'No data in window', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=8, color='#8E8E93')

        self.ax.set_title("Recent Count Trend", fontsize=9, pad=2, color='#E2E2E6')
        self.ax.set_ylabel("Count", fontsize=8, color='#E2E2E6')
        self.ax.tick_params(axis='x', labelsize=7, colors='#8E8E93')
        self.ax.tick_params(axis='y', labelsize=7, colors='#8E8E93')
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, color='#8E8E93') 
        for spine in self.ax.spines.values():
            spine.set_color('#3A3A3C')
        self.fig.tight_layout(pad=0.8) 
        self.canvas.draw()
