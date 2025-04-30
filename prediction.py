import customtkinter as ctk
import webview
from heatmapper import Heatmapper as hm

class PredictionPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.csv_folder = "CSV/"
        self.geojson_path = "heatmapper/stations_final.geojson"
        self.heatmap = hm.HeatmapByParameter(self.csv_folder, self.geojson_path)
        self.create_widgets()

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(self)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(left_panel, text="PREDICTION TOOL", font=("Arial", 20, "bold")).grid(row=0, column=0, columnspan=2, pady=(10, 20))

        self.year_var = ctk.StringVar(value="2023")
        self.month_var = ctk.StringVar(value="Jan")
        self.param_var = ctk.StringVar(value="Nitrate")

        ctk.CTkLabel(left_panel, text="Year:").grid(row=1, column=0, padx=5, sticky="e")
        ctk.CTkOptionMenu(left_panel, variable=self.year_var, values=[str(y) for y in range(2016, 2025)]).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(left_panel, text="Month:").grid(row=2, column=0, padx=5, sticky="e")
        ctk.CTkOptionMenu(left_panel, variable=self.month_var, values=[
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(left_panel, text="Parameter:").grid(row=3, column=0, padx=5, sticky="e")
        ctk.CTkOptionMenu(left_panel, variable=self.param_var, values=[
            "Nitrate", "Phosphate", "Dissolved Oxygen", "pH", "Ammonia", "Chlorophyll-a", "Temperature", "DO", "Phytoplankton"
        ]).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        headers = ["Station", "Nitrate", "Phosphate", "DO", "pH Levels"]
        for col, h in enumerate(headers):
            ctk.CTkLabel(left_panel, text=h, font=("Arial", 12, "bold")).grid(row=4, column=col, padx=2, pady=2)

        for row in range(3):
            ctk.CTkLabel(left_panel, text=f"Month {row+1}").grid(row=5 + row, column=0, padx=2, pady=2)
            for col in range(1, len(headers)):
                ctk.CTkLabel(left_panel, text="Predicted Data").grid(row=5 + row, column=col, padx=2, pady=2)

        ctk.CTkLabel(left_panel, text="Threshold: Predicted Data").grid(row=8, column=0, columnspan=3, pady=(15, 5))
        ctk.CTkLabel(left_panel, text="Algal Bloom Chance: e.g. 87%").grid(row=9, column=0, columnspan=3)

        ctk.CTkButton(left_panel, text="Generate Heatmap", command=self.generate_heatmap).grid(row=10, column=0, columnspan=3, pady=15)

        self.map_frame = ctk.CTkFrame(self)
        self.map_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    def generate_heatmap(self):
        year = int(self.year_var.get())
        month = self.month_str_to_number(self.month_var.get())
        param = self.param_var.get()

        output_path = "heatmapper/station_heatmap.html"
        self.heatmap.generate_map(param, year, month, output_path=output_path)

        webview.create_window(f"Station Heatmap - {param}", output_path)
        webview.start()

    def month_str_to_number(self, m):
        months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                  "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                  "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        return months[m]

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
