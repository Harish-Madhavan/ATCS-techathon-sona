"""Shared ATCS dark-theme design system.

Single source of truth for colors, fonts, spacing and ttk styles so
gui.py / gui_setup.py / gui_components.py stay visually consistent.
All helpers are additive — existing widget APIs are unchanged.
"""
import tkinter as tk
from tkinter import ttk, font as tkfont

BG_DARK = "#121214"
BG_CARD = "#1A1A1E"
BG_INNER = "#25252B"
BG_TRACK = "#2C2C2E"
FG_LIGHT = "#E2E2E6"
FG_MUTED = "#8E8E93"
FG_SUBTLE = "#636366"
ACCENT = "#0A84FF"
ACCENT_DARK = "#005BBF"
GREEN = "#30D158"
GREEN_DIM = "#0B3B1A"
YELLOW = "#FFD60A"
YELLOW_DIM = "#4A3D00"
RED = "#FF453A"
RED_DIM = "#4D0F0F"
ORANGE = "#FF9F0A"
BORDER = "#3A3A3C"

TITLE_FONT = ("Helvetica", 16, "bold")
SECTION_FONT = ("Helvetica", 11, "bold")
BASE_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 9)
SMALL_BOLD = ("Helvetica", 9, "bold")
MONO_FONT = ("Courier New", 10)
COUNT_FONT = ("Helvetica", 10, "bold")

CARD_PAD = 8
CARD_GAP = 8
PILL_PADX = 10
PILL_PADY = 4


def apply_theme(root, style=None):
    """Apply the ATCS dark theme to root + ttk styles. Idempotent."""
    style = style or ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    root.configure(bg=BG_DARK)
    style.configure(".", background=BG_DARK, foreground=FG_LIGHT,
                    fieldbackground=BG_DARK, bordercolor=BG_CARD)
    style.configure("TFrame", background=BG_DARK)
    style.configure("Card.TFrame", background=BG_CARD)
    style.configure("TLabelframe", background=BG_DARK, foreground=FG_LIGHT,
                    bordercolor=BG_INNER, relief="solid")
    style.configure("TLabelframe.Label", background=BG_DARK, foreground=ACCENT,
                    font=SECTION_FONT)
    style.configure("TLabel", background=BG_DARK, foreground=FG_LIGHT)
    style.configure("Muted.TLabel", background=BG_DARK, foreground=FG_MUTED)
    style.configure("Card.TLabel", background=BG_CARD, foreground=FG_LIGHT)
    style.configure("Mono.TLabel", background=BG_DARK, foreground=FG_LIGHT,
                    font=("Courier New", 10, "bold"))
    # Buttons
    style.configure("TButton", background=BG_INNER, foreground=FG_LIGHT,
                    bordercolor=BG_CARD, focuscolor=ACCENT,
                    relief="flat", padding=(10, 6))
    style.map("TButton",
              background=[("active", ACCENT), ("pressed", ACCENT_DARK)],
              foreground=[("active", "white")])
    style.configure("Accent.TButton", background=GREEN, foreground="black")
    style.map("Accent.TButton",
              background=[("active", "#248A3D")],
              foreground=[("active", "white")])
    style.configure("Danger.TButton", background=BG_INNER, foreground=RED)
    style.map("Danger.TButton",
              background=[("active", RED)],
              foreground=[("active", "white")])
    style.configure("DangerActive.TButton", background=RED, foreground="white")
    style.map("DangerActive.TButton",
              background=[("active", "#C62F26")],
              foreground=[("active", "white")])
    # Progress bars
    for name, color in (("green", GREEN), ("yellow", YELLOW),
                        ("red", RED), ("default", ACCENT)):
        style.configure(f"{name}.Horizontal.TProgressbar", background=color,
                        troughcolor=BG_INNER, bordercolor=BG_DARK,
                        lightcolor=color, darkcolor=color)
    try:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        root.option_add("*Font", default_font)
    except Exception:
        pass
    return style


def make_badge(parent, textvariable=None, text="", kind="info"):
    """Pill badge label. kind: info|ok|warn|danger|muted|emergency."""
    palettes = {
        "info": {"bg": ACCENT, "fg": "white"},
        "ok": {"bg": "#1C3A24", "fg": GREEN},
        "warn": {"bg": "#3A2F00", "fg": YELLOW},
        "danger": {"bg": RED, "fg": "white"},
        "muted": {"bg": BG_TRACK, "fg": FG_MUTED},
        "emergency": {"bg": RED, "fg": "white"},
    }
    pal = palettes.get(kind, palettes["info"])
    lbl = tk.Label(parent, text=text, textvariable=textvariable,
                   font=SMALL_BOLD, bg=pal["bg"], fg=pal["fg"],
                   padx=PILL_PADX, pady=PILL_PADY, bd=0)
    lbl._badge_kind = kind
    return lbl


def set_badge_kind(badge, kind):
    palettes = {
        "info": {"bg": ACCENT, "fg": "white"},
        "ok": {"bg": "#1C3B26", "fg": GREEN},
        "warn": {"bg": "#3A2F00", "fg": YELLOW},
        "danger": {"bg": RED, "fg": "white"},
        "muted": {"bg": BG_TRACK, "fg": FG_MUTED},
        "emergency": {"bg": RED, "fg": "white"},
    }
    pal = palettes.get(kind, palettes["muted"])
    try:
        badge.config(bg=pal["bg"], fg=pal["fg"])
        badge._badge_kind = kind
    except Exception:
        pass


def truncate_middle(text, max_len=36):
    text = str(text or "")
    if len(text) <= max_len:
        return text
    keep = max_len - 3
    return text[: keep // 2 + 1] + "..." + text[-(keep // 2):]
