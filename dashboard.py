import tkinter as tk
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Heatmapper as hmr
import webview
import threading
import os

class DashboardPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.parent = parent
        self.csv_file = "CSV/Station_1.csv"  # Ideally this should come from a config

        # Flag for window
        self.webview_window = None

        # State variables
        self.mode = "yearly"
        self.selected_param = None
        self.selected_year = 2016
        self.hm = hmr.HeatmapByParameter(stations_geojson="stations_final.geojson", csv_folder="CSV")
        # Initialize
        self.load_data()
        self.create_widgets()
        self.plot_yearly_average()

    def load_data(self):
        """Load the CSV data"""
        try:
            self.df = pd.read_csv(self.csv_file)
            self.df["Date"] = pd.to_datetime(self.df["Date"], format="%b %Y", errors='coerce')
            self.df["Year"] = self.df["Date"].dt.year
            self.df["Month"] = self.df["Date"].dt.month
            for col in ["pH", "Ammonia", "Nitrate", "Phosphate"]:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            self.df = pd.DataFrame()

    def create_widgets(self):
        """Create UI widgets for the dashboard"""
        tk.Label(self, text="DASHBOARD", font=("Arial", 25, "bold")).grid(row=0, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

        # Year Dropdown
        self.year_label = tk.Label(self, text="Select Year:", font=("Arial", 12))
        self.year_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.year_var = tk.StringVar(value="2016")
        self.year_dropdown = tk.OptionMenu(self, self.year_var, *[str(y) for y in range(2016, 2026)])
        self.year_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Parameter Selection
        self.param_var = tk.StringVar(value="pH")
        self.param_vars = {param: tk.IntVar() for param in ["pH", "Ammonia", "Nitrate", "Phosphate", "Chlorophyll-a", "Temperature", "DO"]}

        self.param_frame = tk.Frame(self)
        self.param_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.error_message = tk.Label(self, text="", fg="red", font=("Arial", 12))
        self.error_message.grid(row=6, column=0, columnspan=3)

        # Buttons
        tk.Button(self, text="Get Monthly Data", command=self.display_monthly_data, font=("Arial", 12), bg="#1d97bd", fg="white").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        tk.Button(self, text="Get Yearly Data", command=self.display_yearly_data, font=("Arial", 12), bg="#1d97bd", fg="white").grid(row=5, column=1, padx=10, pady=5, sticky="w")

        self.generate_btn = tk.Button(self, text="Generate Heatmap", command=self.generate_and_load_map, font=("Arial", 12), bg="#1d97bd", fg="white")
        self.generate_btn.grid(row=1, column=2, padx=10, pady=10, sticky="w")

        # Initial plot canvas
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=6, padx=30, pady=30)

        self.update_parameter_buttons()

    def generate_and_load_map(self):
        """Generate heatmap HTML and load it in Webview"""
        param = self.param_var.get()

        self.hm.generate_map(parameter=param)
        map_path = os.path.abspath("station_heatmap.html")

        if not self.webview_window or not self.webview_window.loaded:
            self.start_webview(map_path)
        else:
            try:
                self.webview_window.load_url(f'file:///{map_path}')
            except Exception:
                # If window has been closed, recreate
                self.start_webview(map_path)

    def start_webview(self, map_path):
        self.webview_window = webview.create_window(f'Heatmap Viewer ({self.hm.current_parameter})', f'file:///{map_path}', width=900, height=700)
        webview.start()
        self.webview_window = None  # Reset after closing

    def update_parameter_buttons(self):
        """Update parameter selection buttons based on mode"""
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        if self.mode == "monthly":
            for i, param in enumerate(self.param_vars.keys()):
                tk.Radiobutton(self.param_frame, text=param, variable=self.param_var, value=param, font=("Arial", 10)).grid(row=0, column=i, padx=5, pady=5, sticky="w")
        else:
            for i, param in enumerate(self.param_vars.keys()):
                tk.Checkbutton(self.param_frame, text=param, variable=self.param_vars[param], font=("Arial", 10)).grid(row=0, column=i, padx=5, pady=5, sticky="w")

    def plot_yearly_average(self):
        """Initial yearly average graph"""
        self.ax.clear()
        if not self.df.empty:
            yearly_avg = self.df.groupby("Year")[["Ammonia", "Nitrate", "Phosphate"]].mean()
            for param in yearly_avg.columns:
                self.ax.plot(yearly_avg.index, yearly_avg[param], marker='o', label=param)
            self.ax.set_title("Yearly Average of Water Quality Parameters")
            self.ax.set_xlabel("Year")
            self.ax.set_ylabel("Average Value")
            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
        self.canvas.draw()

    def display_monthly_data(self):
        """Display monthly bar chart based on selected parameter"""
        self.mode = "monthly"
        self.update_parameter_buttons()
        self.error_message.config(text="")

        selected_param = self.param_var.get()
        selected_year = int(self.year_var.get())

        filtered_df = self.df[self.df["Year"] == selected_year]
        if filtered_df.empty:
            self.error_message.config(text=f"No data for {selected_year}")
            return

        monthly_avg = filtered_df.groupby("Month")[selected_param].mean()

        self.ax.clear()
        self.ax.bar(range(1, 13), monthly_avg)
        self.ax.set_title(f"Monthly {selected_param} Data for {selected_year}")
        self.ax.set_xlabel("Month")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)
        self.canvas.draw()

    def display_yearly_data(self):
        """Display yearly line chart based on selected parameters"""
        self.mode = "yearly"
        self.update_parameter_buttons()
        self.error_message.config(text="")

        selected_params = [param for param, var in self.param_vars.items() if var.get() == 1]
        if not selected_params:
            self.error_message.config(text="Select at least one parameter.")
            return

        filtered_df = self.df.groupby("Year")[selected_params].mean()

        self.ax.clear()
        for param in selected_params:
            self.ax.plot(filtered_df.index, filtered_df[param], marker='o', label=param)
        self.ax.set_title("Yearly Data for Selected Parameters")
        self.ax.set_xlabel("Year")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")