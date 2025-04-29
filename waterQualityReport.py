import customtkinter as ctk
import tkinter as tk  # Still need tk for some widget types not available in CTk
import pandas as pd
import threading
from functools import partial
import queue


class WaterQualityReport(ctk.CTkFrame):
    # Class-level cache for preloaded data
    _data_cache = {
        'full_df': None,
        'station_data': {},
        'initialized': False
    }

    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.cells = {}

        # Initialize queue and threading variables
        self.render_queue = queue.Queue()
        self.render_thread = None
        self.is_rendering = False

        # Update station mapping to match your Excel file
        self.station_names = {
            "Station 1": "Station_1_CWB",
            "Station 2": "Station_2_EastB",
            "Station 4": "Station_4_CentralB",
            "Station 5": "Station_5_NorthernWestBay",
            "Station 8": "Station_8_SouthB",
            "Station 15": "Station_15_SanPedro",
            "Station 16": "Station_16_Sta. Rosa",
            "Station 17": "Station_17_Sanctuary",
            "Station 18": "Station_18_Pagsanjan"
        }

        # Load and preload data first
        if not self._data_cache['initialized']:
            self.preload_data()
        else:
            print("Using cached data")

        # Get unique stations for dropdown and set initial station
        self.unique_stations = sorted(
            self._data_cache['full_df']["Station"].unique().tolist(),
            key=lambda x: int(x.split()[1].split('_')[0]) if x.split()[1].isdigit() else float('inf')
        )
        self.selected_station = ctk.StringVar()
        
        # Create widgets
        self.create_widgets()
        
        # Set initial station and trigger load AFTER widgets are created
        self.selected_station.set("Station 1")
        self.load_csv_data()  # Explicitly load Station 1 data

    def preload_data(self):
        """Load all data only once and cache it"""
        print("Preloading all station data...")
        
        # Define the columns we want to display
        self.display_columns = [
            'Date', 'pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
            'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)',
            'Temperature', 'Chlorophyll-a (ug/L)', 'Station',
            'Phytoplankton'
        ]
        
        try:
            # Load the full dataset
            df = pd.read_excel("CSV/merged_stations.xlsx")
            print("Unique stations in Excel:", df['Station'].unique())
            
            # Convert date column to datetime if it's not already
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            # Filter only the columns we want to display
            df = df[self.display_columns]
            
            # Create a reverse mapping for station names
            reverse_station_map = {v: k for k, v in self.station_names.items()}
            
            # Map the station codes to display names
            df['Station'] = df['Station'].map(reverse_station_map).fillna(df['Station'])
            
            # Store in cache
            self._data_cache['full_df'] = df
            print(f"Loaded full dataset with shape: {df.shape}")

            # Pre-filter data for each display station name
            for display_name in self.station_names.keys():
                station_mask = df["Station"] == display_name
                filtered_data = df[station_mask].copy()
                filtered_data = filtered_data.fillna("Nan")
                
                self._data_cache['station_data'][display_name] = filtered_data
                print(f"Cached {len(filtered_data)} rows for {display_name}")

            self._data_cache['initialized'] = True
            print("Data preloading complete")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")

    def create_widgets(self):
        # Create main content container
        content_container = ctk.CTkFrame(self, fg_color="transparent")
        content_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create report container (left side)
        report_container = ctk.CTkFrame(content_container, fg_color="transparent")
        report_container.grid(row=0, column=0, sticky="nsew", padx=(10, 5))
        
        # Title and dropdown in report container
        reportlb = ctk.CTkLabel(report_container, text="WATER QUALITY REPORT", font=("Arial", 25, "bold"))
        reportlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        dropdownlb = ctk.CTkLabel(report_container, text="Select Station:", font=("Arial", 15))
        dropdownlb.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

        station_dropdown = ctk.CTkOptionMenu(
            report_container,
            variable=self.selected_station,
            values=self.unique_stations,
            command=self.on_station_change,
            width=200  # Fixed width for dropdown
        )
        station_dropdown.grid(row=1, column=0, padx=120, pady=5, sticky="w")

        # Create legend container (right side)
        legend_container = ctk.CTkFrame(content_container, fg_color="transparent")
        legend_container.grid(row=0, column=1, sticky="nsew", padx=(5, 10))

        # Configure grid weights
        content_container.columnconfigure(0, weight=4)  # Report takes 80% of width
        content_container.columnconfigure(1, weight=1)  # Legend takes 20% of width
        content_container.rowconfigure(0, weight=1)

        # Configure report container grid weights
        report_container.columnconfigure(0, weight=1)
        report_container.rowconfigure(2, weight=1)  # Make data grid expandable

        # Save containers as instance variables for later use
        self.report_container = report_container
        self.legend_container = legend_container

        self.load_csv_data()
        self.create_legend()

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")

    def load_csv_data(self):
        station_name = self.selected_station.get()
        print(f"Loading data for station: {station_name}")

        # Get data from cache
        if station_name not in self._data_cache['station_data']:
            print(f"No cached data found for {station_name}")
            return

        df = self._data_cache['station_data'][station_name]
        print(f"Using cached data with {len(df)} rows")

        # Clear previous data
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame) and "data_frame" in str(widget):
                widget.destroy()

        # Create main container
        container = ctk.CTkFrame(self.report_container, fg_color="transparent", corner_radius=15)
        container._name = "!data_frame"
        container.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)

        # Create scrollable frame with adjusted width
        scrollable_frame = ctk.CTkScrollableFrame(
            container,
            fg_color="white",
            width=1200,  # Adjusted width
            height=800,
            corner_radius=0
        )
        scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Style the header cells
        self.render_headers(scrollable_frame, df.columns)

        # Configure column widths for the data grid
        for i in range(len(df.columns)):
            scrollable_frame.grid_columnconfigure(i, weight=1, minsize=100)  # Minimum column width

        # Start the rendering thread
        self.render_data_cells(scrollable_frame, df)

    def render_headers(self, data_frame, columns):
        """Render header cells"""
        for col_idx, col_name in enumerate(columns):
            header_label = ctk.CTkLabel(
                data_frame,
                text=col_name,
                font=("Arial", 12, "bold"),
                fg_color="#E6E6E6",
                corner_radius=8,
                height=35,
                width=120
            )
            header_label.grid(row=0, column=col_idx, sticky="nsew", padx=4, pady=4)

    def render_data_cells(self, data_frame, df):
        """Render data cells in batches using a separate thread"""
        BATCH_SIZE = 50  # Number of rows to render at once

        def create_cell(row_idx, col_idx, value, col_name):
            cell_color = self.get_color(col_name, value)
            
            cell_frame = ctk.CTkFrame(
                data_frame,
                fg_color=cell_color,
                corner_radius=8,
                height=35,
                width=120,
                border_width=0
            )
            cell_frame.grid(row=row_idx + 1, column=col_idx, sticky="nsew", padx=4, pady=4)
            cell_frame.grid_propagate(False)

            cell_label = ctk.CTkLabel(
                cell_frame,
                text=str(value),
                font=("Arial", 12),
                fg_color="transparent",
                anchor="center"
            )
            cell_label.place(relx=0.5, rely=0.5, anchor="center")

        def render_batch():
            """Render cells in batches"""
            try:
                while not self.render_queue.empty() and self.is_rendering:
                    batch = []
                    for _ in range(BATCH_SIZE):
                        if self.render_queue.empty():
                            break
                        batch.append(self.render_queue.get_nowait())

                    for row_idx, col_idx, value, col_name in batch:
                        # Schedule cell creation on the main thread
                        self.after(0, create_cell, row_idx, col_idx, value, col_name)
                    
                    # Give the main thread a chance to update
                    self.after(10)

            finally:
                self.is_rendering = False

        # Clear any existing render queue
        while not self.render_queue.empty():
            self.render_queue.get_nowait()

        # Fill the queue with cell data
        for row_idx, row in df.iterrows():
            for col_idx, col_name in enumerate(df.columns):
                self.render_queue.put((row_idx, col_idx, row[col_name], col_name))

        # Start new rendering thread if not already running
        if not self.is_rendering:
            self.is_rendering = True
            self.render_thread = threading.Thread(target=render_batch, daemon=True)
            self.render_thread.start()

    def create_legend(self):
        legend_data = {
        "pH": [
            {"color": "#A7C7E7", "label": "Conformed with Classes A and B (6.5-8.5)"},  
            {"color": "#C3E6CB", "label": "Conformed with Class C (6.5-9.0)"},          
            {"color": "#F9E79F", "label": "Conformed with Class D (6.0-9.0)"},          
            {"color": "#F5B7B1", "label": "Failed Guidelines (< 6 or > 9)"},            
            {"color": "#D5DBDB", "label": "No data"},                                    
        ],
        "Nitrate": [
            {"color": "#A7C7E7", "label": "Conformed with Classes A, B, C (< 7 mg/L)"},
            {"color": "#C3E6CB", "label": "Conformed with Class D (7-15 mg/L)"},
            {"color": "#F5B7B1", "label": "Failed Guidelines (> 15 mg/L)"},
            {"color": "#D5DBDB", "label": "No data"},
        ],
        "Ammonia": [
            {"color": "#A7C7E7", "label": "Conformed with Classes A, B, C (< 0.06 mg/L)"},
            {"color": "#C3E6CB", "label": "Conformed with Class D (0.06-0.30 mg/L)"},
            {"color": "#F5B7B1", "label": "Failed Guidelines (> 0.30 mg/L)"},
            {"color": "#D5DBDB", "label": "No data"},
        ],
        "Phosphate": [
            {"color": "#A7C7E7", "label": "Conformed with Classes A, B, C (< 0.025 mg/L)"},
            {"color": "#C3E6CB", "label": "Conformed with Class D (0.025-0.05 mg/L)"},
            {"color": "#F5B7B1", "label": "Failed Guidelines (> 0.05 mg/L)"},
            {"color": "#D5DBDB", "label": "No data"},
        ],
        "Chlorophyll-a": [
            {"color": "#98FB98", "label": "Very Low (< 25 ug/L)"},
            {"color": "#90EE90", "label": "Low (25-50 ug/L)"},
            {"color": "#3CB371", "label": "Medium (50-100 ug/L)"},
            {"color": "#228B22", "label": "High (100-150 ug/L)"},
            {"color": "#006400", "label": "Very High (> 150 ug/L)"},
            {"color": "#D5DBDB", "label": "No data"},
        ],
        "Phytoplankton": [
            {"color": "#FFB6C1", "label": "Minor Bloom (50,000-99,999 cells/L)"},
            {"color": "#FF69B4", "label": "Moderate Bloom (100,000-499,999 cells/L)"},
            {"color": "#FF1493", "label": "Massive Bloom (≥ 500,000 cells/L)"},
            {"color": "#FFFFFF", "label": "No Bloom (< 50,000 cells/L)"},
            {"color": "#D5DBDB", "label": "No data"},
        ],
    }
        # Clear any existing legend
        for widget in self.legend_container.winfo_children():
            widget.destroy()

        # Add legend title
        legend_title = ctk.CTkLabel(
            self.legend_container, 
            text="LEGENDS", 
            font=("Arial", 16, "bold")
        )
        legend_title.pack(side="top", pady=(10, 20))

        # Create scrollable frame with adjusted size
        legend_scroll = ctk.CTkScrollableFrame(
            self.legend_container,
            fg_color="transparent",
            width=250,  # Reduced width
            height=800  # Full height
        )
        legend_scroll.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # Add legends in single column
        for parameter, items in legend_data.items():
            param_frame = ctk.CTkFrame(legend_scroll, fg_color="transparent")
            param_frame.pack(side="top", fill="x", pady=(0, 15), padx=5)

            title_label = ctk.CTkLabel(
                param_frame,
                text=f"{parameter} Legend:",
                font=("Arial", 14, "bold")
            )
            title_label.pack(side="top", anchor="w", pady=(0, 5))

            for item in items:
                row_container = ctk.CTkFrame(param_frame, fg_color="transparent")
                row_container.pack(side="top", fill="x", pady=1)

                color_box = ctk.CTkFrame(
                    row_container,
                    fg_color=item["color"],
                    width=30,
                    height=20,
                    corner_radius=3
                )
                color_box.pack(side="left", padx=5)
                color_box.pack_propagate(False)

                label = ctk.CTkLabel(
                    row_container,
                    text=item["label"],
                    anchor="w",
                    font=("Arial", 11),  # Slightly smaller font
                    wraplength=200  # Adjusted wraplength
                )
                label.pack(side="left", fill="x", expand=True, padx=(5, 10))

    def get_color(self, param, value):
        """Returns the appropriate color based on parameter value ranges."""
        try:
            if pd.isna(value) or value == "Nan":
                return "#D5DBDB"  # Light gray for NaN values

            # Non-numeric columns should be white
            if param in ["Date", "Station", "Occurences"]:  # Removed Phytoplankton from here
                return "#FFFFFF"

            # For numeric parameters
            if param in ["pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", 
                        "Inorganic Phosphate (mg/L)", "Dissolved Oxygen (mg/L)",
                        "Temperature", "Chlorophyll-a (ug/L)", "Phytoplankton"]:
                
                # Try to convert to float for comparison
                try:
                    value = float(value)
                except:
                    return "#D5DBDB"  # Light gray for invalid values

                if param == "pH (units)":
                    if 6.5 <= value <= 8.5:
                        return "#A7C7E7"  # Soft blue - Classes A and B
                    elif 6.5 <= value <= 9.0:
                        return "#C3E6CB"  # Soft green - Class C
                    elif 6.0 <= value <= 9.0:
                        return "#F9E79F"  # Soft yellow - Class D
                    else:
                        return "#F5B7B1"  # Soft red - Failed Guidelines
                        
                elif param == "Nitrate (mg/L)":
                    if value < 7:
                        return "#A7C7E7"
                    elif 7 <= value <= 15:
                        return "#C3E6CB"
                    else:
                        return "#F5B7B1"
                        
                elif param == "Ammonia (mg/L)":
                    if value < 0.06:
                        return "#A7C7E7"
                    elif 0.06 <= value <= 0.30:
                        return "#C3E6CB"
                    else:
                        return "#F5B7B1"
                        
                elif param == "Inorganic Phosphate (mg/L)":
                    if value < 0.025:
                        return "#A7C7E7"
                    elif 0.025 <= value <= 0.05:
                        return "#C3E6CB"
                    else:
                        return "#F5B7B1"

                elif param == "Chlorophyll-a (ug/L)":
                    if value < 25:
                        return "#98FB98"  # Very Low
                    elif 25 <= value < 50:
                        return "#90EE90"  # Low
                    elif 50 <= value < 100:
                        return "#3CB371"  # Medium
                    elif 100 <= value < 150:
                        return "#228B22"  # High
                    else:
                        return "#006400"  # Very High

                elif param == "Phytoplankton":
                    try:
                        value = float(value)
                        if value < 50000:
                            return "#FFFFFF"  # No Bloom
                        elif 50000 <= value < 100000:
                            return "#FFB6C1"  # Minor Bloom
                        elif 100000 <= value < 500000:
                            return "#FF69B4"  # Moderate Bloom
                        else:
                            return "#FF1493"  # Massive Bloom
                    except:
                        return "#FFFFFF"  # Default for non-numeric phytoplankton values

            return "#FFFFFF"  # Default white for other columns
            
        except:
            return "#D5DBDB"  # Light gray for any errors

    def on_station_change(self, selected_station):
        """Handle station selection change."""
        print(f"Selected station: {selected_station}")
        self.selected_station.set(selected_station)
        self.load_csv_data()

    def destroy(self):
        """Clean up resources when widget is destroyed"""
        self.is_rendering = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        super().destroy()