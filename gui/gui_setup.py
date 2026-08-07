import tkinter as tk
from tkinter import font, filedialog, ttk
import os
from core import config

class SetupDialog:
    def __init__(self, parent):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Adaptive Traffic Control - Welcome & Setup")
        self.dialog.configure(bg="#121214")
        
        # Center the dialog
        dialog_width = 650
        dialog_height = 700
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (dialog_width // 2)
        y = (screen_height // 2) - (dialog_height // 2)
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()  # Make it modal
        
        self.success = False
        self._build_ui()
        
    def _build_ui(self):
        # Setup dialog-specific fonts
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        header_font = font.Font(family="Helvetica", size=11, weight="bold")
        regular_font = font.Font(family="Helvetica", size=9)
        btn_font = font.Font(family="Helvetica", size=10, weight="bold")
        
        # Bind system settings
        self.esp32_enabled_var = tk.BooleanVar(value=config.ESP32_ENABLED)
        self.esp32_port_var = tk.StringVar(value=config.ESP32_PORT)
        self.rl_enabled_var = tk.BooleanVar(value=config.RL_MODEL_ENABLED)
        self.rl_model_path_var = tk.StringVar(value=config.RL_MODEL_PATH)

        # Prepopulate with current config paths
        self.current_paths = {}
        for approach_name, video_path in config.VIDEO_PATHS:
            self.current_paths[approach_name] = tk.StringVar(value=video_path)
            
        header_frame = tk.Frame(self.dialog, bg="#121214")
        header_frame.pack(fill=tk.X, pady=(20, 10), padx=20)
        
        lbl_title = tk.Label(header_frame, text="ADAPTIVE TRAFFIC LIGHT CONTROLLER", font=title_font, fg="#0A84FF", bg="#121214")
        lbl_title.pack(anchor=tk.W)
        
        lbl_subtitle = tk.Label(header_frame, text="Configure video inputs for each approach before starting simulation", font=regular_font, fg="#8E8E93", bg="#121214")
        lbl_subtitle.pack(anchor=tk.W, pady=(2, 0))
        
        sep = tk.Frame(self.dialog, height=1, bg="#2C2C2E")
        sep.pack(fill=tk.X, padx=20, pady=5)
        
        approaches_frame = tk.Frame(self.dialog, bg="#121214")
        approaches_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        directions = ["Northbound", "Eastbound", "Westbound", "Southbound"]
        self.path_labels = {}
        
        for idx, direction in enumerate(directions):
            row_frame = tk.Frame(approaches_frame, bg="#1A1A1E", bd=1, relief="solid", highlightbackground="#2C2C2E", highlightcolor="#2C2C2E")
            row_frame.pack(fill=tk.X, pady=6, ipady=4)
            
            color_accent = "#FF453A" if idx == 0 else ("#FFD60A" if idx == 1 else ("#30D158" if idx == 2 else "#0A84FF"))
            accent_bar = tk.Frame(row_frame, width=4, bg=color_accent)
            accent_bar.pack(side=tk.LEFT, fill=tk.Y)
            
            info_frame = tk.Frame(row_frame, bg="#1A1A1E")
            info_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
            
            lbl_dir = tk.Label(info_frame, text=f"{direction} Approach", font=header_font, fg="#E2E2E6", bg="#1A1A1E")
            lbl_dir.pack(anchor=tk.W)
            
            initial_path = self.current_paths[direction].get()
            initial_filename = os.path.basename(initial_path) if initial_path else "Not Configured"
            
            lbl_filename = tk.Label(info_frame, text=initial_filename, font=regular_font, fg="#8E8E93", bg="#1A1A1E", wraplength=350, justify=tk.LEFT)
            lbl_filename.pack(anchor=tk.W, pady=(2, 0))
            self.path_labels[direction] = lbl_filename
            
            btn_browse = tk.Button(
                row_frame, text="Browse", font=regular_font, bg="#2C2C2E", fg="#E2E2E6",
                activebackground="#0A84FF", activeforeground="white", bd=0, relief="flat",
                padx=8, pady=4, cursor="hand2", command=lambda d=direction: self._browse_path(d)
            )
            btn_browse.pack(side=tk.RIGHT, padx=15)
            
        global_helpers_frame = tk.Frame(self.dialog, bg="#121214")
        global_helpers_frame.pack(fill=tk.X, padx=20, pady=10)
        
        btn_apply_all = tk.Button(
            global_helpers_frame, text="Apply Single Video to All Approaches", font=regular_font,
            bg="#1A1A1E", fg="#0A84FF", activebackground="#0A84FF", activeforeground="white",
            bd=1, relief="solid", highlightbackground="#0A84FF", cursor="hand2",
            padx=10, pady=4, command=self._apply_single_video
        )
        btn_apply_all.pack(side=tk.LEFT)
        
        btn_reset = tk.Button(
            global_helpers_frame, text="Reset to Defaults", font=regular_font,
            bg="#1A1A1E", fg="#8E8E93", activebackground="#2C2C2E", activeforeground="white",
            bd=0, cursor="hand2", padx=10, pady=4, command=self._reset_to_defaults
        )
        btn_reset.pack(side=tk.RIGHT)

        # System Configuration Frame
        settings_frame = tk.LabelFrame(self.dialog, text="System Configuration", font=header_font, bg="#121214", fg="#0A84FF", bd=1, relief="solid", padx=10, pady=8)
        settings_frame.pack(fill=tk.X, padx=20, pady=5)
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure(1, weight=1)

        # Left Column: Hardware (ESP32)
        hw_frame = tk.Frame(settings_frame, bg="#121214")
        hw_frame.grid(row=0, column=0, sticky="nsew", padx=10)

        cb_esp = tk.Checkbutton(hw_frame, text="Enable ESP32 Serial", variable=self.esp32_enabled_var, bg="#121214", fg="#E2E2E6", selectcolor="#1A1A1E", activebackground="#121214", activeforeground="#E2E2E6", font=regular_font)
        cb_esp.pack(anchor=tk.W)

        port_row = tk.Frame(hw_frame, bg="#121214")
        port_row.pack(anchor=tk.W, pady=3)
        lbl_port = tk.Label(port_row, text="Port / MOCK:", fg="#8E8E93", bg="#121214", font=regular_font)
        lbl_port.pack(side=tk.LEFT)
        ent_port = tk.Entry(port_row, textvariable=self.esp32_port_var, width=12, bg="#1A1A1E", fg="#E2E2E6", insertbackground="white", bd=1, relief="solid")
        ent_port.pack(side=tk.LEFT, padx=5)

        # Right Column: AI Control (RL Agent)
        ai_frame = tk.Frame(settings_frame, bg="#121214")
        ai_frame.grid(row=0, column=1, sticky="nsew", padx=10)

        cb_rl = tk.Checkbutton(ai_frame, text="Enable RL Agent Control", variable=self.rl_enabled_var, bg="#121214", fg="#E2E2E6", selectcolor="#1A1A1E", activebackground="#121214", activeforeground="#E2E2E6", font=regular_font)
        cb_rl.pack(anchor=tk.W)

        rl_row = tk.Frame(ai_frame, bg="#121214")
        rl_row.pack(anchor=tk.W, pady=3, fill=tk.X)

        btn_browse_rl = tk.Button(
            rl_row, text="Browse Weights", font=regular_font, bg="#2C2C2E", fg="#E2E2E6",
            activebackground="#0A84FF", activeforeground="white", bd=0, relief="flat",
            padx=8, pady=3, cursor="hand2", command=self._browse_rl_model
        )
        btn_browse_rl.pack(side=tk.LEFT)

        self.lbl_rl_filename = tk.Label(rl_row, text=os.path.basename(self.rl_model_path_var.get()), font=regular_font, fg="#8E8E93", bg="#121214", wraplength=120, justify=tk.LEFT)
        self.lbl_rl_filename.pack(side=tk.LEFT, padx=8)
        
        sep2 = tk.Frame(self.dialog, height=1, bg="#2C2C2E")
        sep2.pack(fill=tk.X, padx=20, pady=10)
        
        bottom_frame = tk.Frame(self.dialog, bg="#121214")
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 20), padx=20)
        
        btn_exit = tk.Button(
            bottom_frame, text="Exit Setup", font=btn_font,
            bg="#2C2C2E", fg="#FF453A", activebackground="#FF453A", activeforeground="white",
            bd=0, relief="flat", width=15, pady=8, cursor="hand2", command=self._on_cancel
        )
        btn_exit.pack(side=tk.LEFT)
        
        btn_start = tk.Button(
            bottom_frame, text="Start Simulation", font=btn_font,
            bg="#30D158", fg="black", activebackground="#248A3D", activeforeground="white",
            bd=0, relief="flat", width=25, pady=8, cursor="hand2", command=self._on_start
        )
        btn_start.pack(side=tk.RIGHT)
        
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.parent.wait_window(self.dialog)

    def _browse_path(self, direction):
        initial_dir = os.path.dirname(self.current_paths[direction].get()) if os.path.exists(self.current_paths[direction].get()) else os.path.dirname(__file__)
        selected_file = filedialog.askopenfilename(
            parent=self.dialog,
            initialdir=initial_dir,
            title=f"Select Video for {direction}",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if selected_file:
            self.current_paths[direction].set(selected_file)
            filename = os.path.basename(selected_file)
            self.path_labels[direction].config(text=filename, fg="#30D158")
            
    def _browse_rl_model(self):
        initial_dir = os.path.dirname(self.rl_model_path_var.get()) if os.path.exists(self.rl_model_path_var.get()) else os.path.dirname(__file__)
        selected_file = filedialog.askopenfilename(
            parent=self.dialog,
            initialdir=initial_dir,
            title="Select RL Model Weights (.pth)",
            filetypes=[("Weights files", "*.pth *.pt"), ("All files", "*.*")]
        )
        if selected_file:
            self.rl_model_path_var.set(selected_file)
            self.lbl_rl_filename.config(text=os.path.basename(selected_file), fg="#30D158")
            
    def _apply_single_video(self):
        selected_file = filedialog.askopenfilename(
            parent=self.dialog,
            title="Select Video to Apply to All Approaches",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if selected_file:
            for direction in self.current_paths:
                self.current_paths[direction].set(selected_file)
                filename = os.path.basename(selected_file)
                self.path_labels[direction].config(text=filename, fg="#30D158")
                
    def _reset_to_defaults(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_vid = os.path.join(project_root, "assets", "vid1.mp4")
        for direction in self.current_paths:
            self.current_paths[direction].set(default_vid)
            self.path_labels[direction].config(text=os.path.basename(default_vid), fg="#E2E2E6")
        
        # Reset system settings
        self.esp32_enabled_var.set(False)
        self.esp32_port_var.set("COM3")
        self.rl_enabled_var.set(True)
        default_rl_path = os.path.join(project_root, "models", "rl_model.pth")
        self.rl_model_path_var.set(default_rl_path)
        self.lbl_rl_filename.config(text=os.path.basename(default_rl_path), fg="#8E8E93")
            
    def _on_start(self):
        new_video_paths = []
        for d in self.current_paths:
            new_video_paths.append((d, self.current_paths[d].get()))
        config.VIDEO_PATHS = new_video_paths
        
        # Commit system settings to config
        config.ESP32_ENABLED = self.esp32_enabled_var.get()
        config.ESP32_PORT = self.esp32_port_var.get()
        config.RL_MODEL_ENABLED = self.rl_enabled_var.get()
        config.RL_MODEL_PATH = self.rl_model_path_var.get()
        
        print(f"[GUI Setup] Video paths updated to: {config.VIDEO_PATHS}")
        print(f"[GUI Setup] ESP32: {config.ESP32_ENABLED} on Port {config.ESP32_PORT}")
        print(f"[GUI Setup] RL Enabled: {config.RL_MODEL_ENABLED} using weights {config.RL_MODEL_PATH}")
        self.success = True
        self.dialog.destroy()
        
    def _on_cancel(self):
        self.success = False
        self.dialog.destroy()
