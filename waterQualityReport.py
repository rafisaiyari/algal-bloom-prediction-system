import customtkinter as ctk
import tkinter as tk  # Still need tk for some widget types not available in CTk
import pandas as pd


class WaterQualityReport(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.cells = {}

        # Station file mapping
        self.station_files = {
            "Station 1": "CSV/Station_1_CWB.csv",
            "Station 2": "CSV/Station_2_EB.csv",
            "Station 4": "CSV/Station_4_CB.csv",
            "Station 5": "CSV/Station_5_NWB.csv",
            "Station 8": "CSV/Station_8_SouthB.csv",
            "Station 15": "CSV/Station_15_SP.csv",
            "Station 16": "CSV/Station_16_SR.csv",
            "Station 17": "CSV/Station_17_Sanctuary.csv",
            "Station 18": "CSV/Station_18_Pagsanjan.csv",
        }
        self.selected_station = ctk.StringVar(value="Station 1")  # Default station

        self.create_widgets()

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")

    def create_widgets(self):
        reportlb = ctk.CTkLabel(self, text="WATER QUALITY REPORT", font=("Arial", 25, "bold"))
        reportlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        dropdownlb = ctk.CTkLabel(self, text="Select Station:", font=("Arial", 15))
        dropdownlb.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

        station_dropdown = ctk.CTkOptionMenu(
                self,
                variable=self.selected_station,  # Bind the selected station variable
                values=list(self.station_files.keys()),  # Populate dropdown with station names
                command=self.on_station_change  # Callback when a station is selected
        )
        station_dropdown.grid(row=1, column=0, padx=120, pady=5, sticky="w")

        self.load_csv_data()
        self.create_legend()

    def load_csv_data(self):
        station_name = self.selected_station.get()
        file_path = self.station_files.get(station_name, "")

        if not file_path:
            return

        df = pd.read_csv(file_path).fillna("Nan")

        # Clear previous data
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame) and "data_frame" in str(widget):
                widget.destroy()

        container = ctk.CTkFrame(self, fg_color="white")
        # Manually set the name after creation in tkinter's way if needed
        container._name = "!data_frame"
        container.grid(row=2, column=0, sticky="nsw", padx=10, pady=10)

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # For the canvas and scrollbar, we still need to use Tkinter since CTk doesn't have direct canvas equivalent
        canvas = tk.Canvas(container, bg="white", height=800, width=1500)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(container, orientation="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        data_frame = ctk.CTkFrame(canvas, fg_color="white")
        canvas.create_window((0, 0), window=data_frame, anchor="nw")

        for col_idx, col_name in enumerate(df.columns):
            header_label = ctk.CTkLabel(
                data_frame, text=col_name,
                font=("Arial", 10, "bold"), fg_color="lightgray",
                padx=10, pady=5, corner_radius=0
            )
            header_label.grid(row=0, column=col_idx, sticky="nsew")

        def get_color(param, value):
            try:
                value = float(value)
                if param == "pH":
                    return "blue" if 6.5 <= value <= 8.5 else "lightblue" if 6.5 <= value <= 9.0 else "green" if 6.0 <= value <= 9.0 else "red"
                elif param == "Nitrate":
                    return "blue" if value < 7 else "lightblue" if 7 <= value <= 15 else "red"
                elif param == "Ammonia":
                    return "blue" if value < 0.06 else "lightblue" if 0.06 <= value <= 0.30 else "red"
                elif param == "Phosphate":
                    return "blue" if value < 0.025 else "lightblue" if 0.025 <= value <= 0.05 else "red"
            except:
                return "white"

        for row_idx, row in df.iterrows():
            for col_idx, col_name in enumerate(df.columns):
                cell_value = row[col_name]
                cell_color = get_color(col_name, cell_value)

                # For the entry widgets with color backgrounds, we'll use Tkinter since CTk's Entry
                # doesn't support the same level of styling with background colors
                cell_entry = tk.Entry(
                    data_frame,
                    font=("Arial", 14),
                    borderwidth=1,
                    justify="center",
                    bg=cell_color,
                    readonlybackground=cell_color,
                )
                cell_entry.insert(0, cell_value)
                cell_entry.grid(row=row_idx + 1, column=col_idx, sticky="nsew")

                self.cells[(row_idx, col_idx)] = cell_entry

        # set all cells to readonly
        for cell_entry in self.cells.values():
            cell_entry.configure(state='readonly')

        for col_idx in range(len(df.columns)):
            data_frame.grid_columnconfigure(col_idx, weight=1)

        data_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_mouse_wheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")

        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

    def create_legend(self):
        legend_data = {
            "pH": [
                {"color": "blue", "label": "Conformed with Classes A and B (6.5-8.5)"},
                {"color": "lightblue", "label": "Conformed with Class C (6.5-9.0)"},
                {"color": "green", "label": "Conformed with Class D (6.0-9.0)"},
                {"color": "red", "label": "Failed Guidelines (< 6 or > 9)"},
                {"color": "gray", "label": "No data"},
            ],
            "Nitrate": [
                {"color": "blue", "label": "Conformed with Classes A, B, C (< 7 mg/L)"},
                {"color": "lightblue", "label": "Conformed with Class D (7-15 mg/L)"},
                {"color": "red", "label": "Failed Guidelines (> 15 mg/L)"},
                {"color": "gray", "label": "No data"},
            ],
            "Ammonia": [
                {"color": "blue", "label": "Conformed with Classes A, B, C (< 0.06 mg/L)"},
                {"color": "lightblue", "label": "Conformed with Class D (0.06-0.30 mg/L)"},
                {"color": "red", "label": "Failed Guidelines (> 0.30 mg/L)"},
                {"color": "gray", "label": "No data"},
            ],
            "Phosphate": [
                {"color": "blue", "label": "Conformed with Classes A, B, C (< 0.025 mg/L)"},
                {"color": "lightblue", "label": "Conformed with Class D (0.025-0.05 mg/L)"},
                {"color": "red", "label": "Failed Guidelines (> 0.05 mg/L)"},
                {"color": "gray", "label": "No data"},
            ],
        }

        legend_container = ctk.CTkFrame(self, fg_color="transparent")
        # Set name if needed for identification later
        legend_container._name = "!legend_container"
        legend_container.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        columns = 2
        row_num = 0
        col_num = 0

        for parameter, items in legend_data.items():
            legend_frame = ctk.CTkFrame(legend_container, fg_color="transparent")
            # Apply padding when grid/pack is used, not in the constructor
            legend_frame.grid(row=row_num, column=col_num, sticky="w", padx=10, pady=5)

            title_label = ctk.CTkLabel(legend_frame, text=f"{parameter} Legend:", font=("Arial", 12, "bold"))
            title_label.pack(side="top", anchor="w", pady=2)

            for item in items:
                row_container = ctk.CTkFrame(legend_frame, fg_color="transparent")
                row_container.pack(side="top", fill="x", padx=5, pady=1)

                # For color boxes, we'll use standard tkinter Labels
                # Create a color box - using tk.Label since CTkLabel doesn't support solid background colors the same way
                color_box = tk.Label(row_container, bg=item["color"], width=4, height=1, relief="solid", borderwidth=1)
                color_box.pack(side="left", padx=5)

                # Label description next to the color box
                label = ctk.CTkLabel(row_container, text=item["label"], anchor="w")
                label.pack(side="left", fill="x", expand=True)

            # Move to the next column, or next row if needed
            col_num += 1
            if col_num >= columns:
                col_num = 0
                row_num += 1

    def on_station_change(self, selected_station):
        print(f"Selected station: {selected_station}")
        self.load_csv_data()