import tkinter as tk
import pandas as pd


class WaterQualRep(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.parent = parent
        self.cells = {}  

        #will add in next patch sirs, which is the dropdown menu
        self.station_files = { 
            "Station 1": "CSV/Station_1_CWB.csv",
            "Station 2": "CSV/Station_2_CWB.csv",
            "Station 3": "CSV/Station_3_CWB.csv",
        }
        self.selected_station = tk.StringVar()
        self.selected_station.set("Station 1")  # Default station
        self.create_widgets()
        self.create_widgets()

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")

    def create_widgets(self):
        reportlb = tk.Label(self, text="WATER QUALITY REPORT", font=("Arial", 25, "bold"))
        reportlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        self.load_csv_data()
        self.create_legend()

    def load_csv_data(self, *args):
        station_name = self.selected_station.get()
        file_path = self.station_files.get(station_name, "")

        if not file_path:
            return

        df = pd.read_csv(file_path)
        df = df.fillna("Nan")

        # Clear previous data
        for widget in self.winfo_children():
            if isinstance(widget, tk.Frame) and widget.winfo_name().startswith("!data_frame"):
                widget.destroy()

        container = tk.Frame(self, bg="white", name="!data_frame")
        container.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, bg="white", height=800, width=1800)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a Frame inside the Canvas to hold all data
        data_frame = tk.Frame(canvas, bg="white")
        canvas.create_window((0, 0), window=data_frame, anchor="nw")

        # Preload all column headers
        for col_idx, col_name in enumerate(df.columns):
            header_label = tk.Label(
                data_frame, text=col_name,
                font=("Arial", 10, "bold"), bg="lightgray",
                padx=10, pady=5, borderwidth=1, relief="solid"
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

                cell_entry = tk.Entry(
                    data_frame, 
                    bg=cell_color, fg="black",
                    font=("Arial", 14), borderwidth=1,
                    justify="center"
                )
                cell_entry.insert(0, cell_value)
                cell_entry.grid(row=row_idx + 1, column=col_idx, sticky="nsew")

                # Store reference to cell for future updates
                self.cells[(row_idx, col_idx)] = cell_entry

        # Adjust column weights for proper resizing
        for col_idx in range(len(df.columns)):
            data_frame.grid_columnconfigure(col_idx, weight=1)

        # Update scroll region once
        data_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Enable smooth mouse scrolling
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

        # Remove existing legend frames
        for widget in self.winfo_children():
            if isinstance(widget, tk.Frame) and widget.winfo_name().startswith("!legend_"):
                widget.destroy()

        row_num = 3  # Start placing legends below the table
        for parameter, items in legend_data.items():
            legend_frame = tk.Frame(self, bg="#F1F1F1", padx=20, pady=10, name=f"!legend_{parameter}")
            legend_frame.grid(row=row_num, column=0, sticky="w")

            title_label = tk.Label(legend_frame, text=f"{parameter} Legend:", font=("Arial", 12, "bold"), bg="#F1F1F1")
            title_label.pack(side="top", anchor="w")

            for item in items:
                color_canvas = tk.Canvas(legend_frame, width=20, height=10, highlightthickness=0, bg="#F1F1F1")
                color_canvas.create_rectangle(2, 2, 18, 8, fill=item["color"], outline="")
                color_canvas.pack(side="left", padx=(0, 5))

                text_label = tk.Label(legend_frame, text=item["label"], bg="#F1F1F1")
                text_label.pack(side="left")

            row_num += 1
