import customtkinter as ctk
import tkinter as tk
import pandas as pd
import threading
import queue
import time


class WaterQualityReport(ctk.CTkFrame):
    # Class-level cache for preloaded data
    _data_cache = {
        'full_df': None,
        'station_data': {},
        'initialized': False
    }

    def __init__(self, parent, bg_color=None, show_loading=True):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.show_loading = show_loading

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

        # Load and preload data if not already initialized
        if not self._data_cache['initialized']:
            self.preload_data(silent=not show_loading)
        else:
            print("Using cached data")

        # Get unique stations for dropdown and sort them properly
        self.unique_stations = sorted(
            self._data_cache['full_df']["Station"].unique().tolist(),
            key=lambda x: int(x.split()[1].split('_')[0]) if x.split()[1].isdigit() else float('inf')
        )
        self.selected_station = ctk.StringVar(value=self.unique_stations[0])

        # Create UI elements
        self.create_widgets()

        # Rendering queue and thread
        self.render_queue = queue.Queue()
        self.is_rendering = False
        self.render_thread = None

        # Loading progress tracking
        self.total_cells = 0
        self.processed_cells = 0
        self.loading_frame = None
        self.progress_bar = None
        self.progress_label = None

    def preload_data(self, silent=False):
        """Load all data only once and cache it"""
        if not silent:
            print("Preloading all station data...")

        # Define the columns we want to display
        self.display_columns = [
            'Date', 'pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
            'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)',
            'Temperature', 'Chlorophyll-a (ug/L)', 'Phytoplankton', 'Station'
        ]

        try:
            df = pd.read_excel("train/merged_stations.xlsx")
            if not silent:
                print("Unique stations in Excel:", df['Station'].unique())

            df['Date'] = df['Date'].dt.date

            # Filter only the columns we want to display
            df = df[self.display_columns]

            # Create a reverse mapping for station names
            reverse_station_map = {v: k for k, v in self.station_names.items()}

            # Map the station codes to display names
            df['Station'] = df['Station'].map(reverse_station_map).fillna(df['Station'])

            # Store in cache
            self._data_cache['full_df'] = df
            if not silent:
                print(f"Loaded full dataset with shape: {df.shape}")

            # Pre-filter data for each display station name
            for display_name in self.station_names.keys():
                station_mask = df["Station"] == display_name
                filtered_data = df[station_mask].copy()
                filtered_data = filtered_data.fillna("Nan")

                self._data_cache['station_data'][display_name] = filtered_data
                if not silent:
                    print(f"Cached {len(filtered_data)} rows for {display_name}")

            self._data_cache['initialized'] = True
            if not silent:
                print("Data preloading complete")

        except Exception as e:
            if not silent:
                print(f"Error loading data: {str(e)}")

    def create_widgets(self):
        # Create main content container with more width
        self.content_container = ctk.CTkFrame(self, fg_color="transparent")
        self.content_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Create report container (left side) - make this wider
        self.report_container = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.report_container.grid(row=0, column=0, sticky="nsew", padx=(10, 5))

        # Title and dropdown in report container
        reportlb = ctk.CTkLabel(self.report_container, text="WATER QUALITY REPORT", font=("Arial", 25, "bold"))
        reportlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        dropdownlb = ctk.CTkLabel(self.report_container, text="Select Station:", font=("Arial", 15))
        dropdownlb.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

        station_dropdown = ctk.CTkOptionMenu(
            self.report_container,
            variable=self.selected_station,
            values=self.unique_stations,
            command=self.on_station_change
        )
        station_dropdown.grid(row=1, column=0, padx=120, pady=5, sticky="w")

        # Create legend container (right side)
        self.legend_container = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.legend_container.grid(row=0, column=1, sticky="nsew", padx=(5, 10))

        # Create a container for the data grid - make this wider
        self.data_container = ctk.CTkFrame(self.report_container, fg_color="transparent", corner_radius=15)
        self.data_container.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)

        # Configure weights so data container expands
        self.report_container.rowconfigure(2, weight=1)
        self.report_container.columnconfigure(0, weight=1)

        # Configure grid weights - give more weight to the report side
        self.content_container.columnconfigure(0, weight=4)  # Report takes 80% of width
        self.content_container.columnconfigure(1, weight=1)  # Legend takes 20% of width
        self.content_container.rowconfigure(0, weight=1)  # Row expands vertically

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")
        # Load data when showing the page
        self.load_station_data()
        self.create_legend()

    def create_loading_ui(self):
        """Create loading overlay that covers the entire window"""
        # Create a semi-transparent overlay frame that covers everything
        self.loading_frame = ctk.CTkFrame(
            self,  # Attach to self instead of data_container
            fg_color=("#FFFFFF", "#1E1E1E"),  # Light/Dark mode colors
            corner_radius=0,  # No rounded corners for full coverage
        )
        self.loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)  # Cover entire window

        # Create centered container for loading elements
        loading_container = ctk.CTkFrame(
            self.loading_frame,
            fg_color=("#F0F0F0", "#2D2D2D"),
            corner_radius=15,
            border_width=1,
            border_color=("#E0E0E0", "#404040")
        )
        loading_container.place(relx=0.5, rely=0.5, anchor="center")

        # Add loading icon/title
        title_label = ctk.CTkLabel(
            loading_container,
            text="Loading Water Quality Data",
            font=("Arial", 20, "bold"),
            text_color=("gray10", "gray90")
        )
        title_label.pack(pady=(30, 5))

        # Add status message
        self.status_label = ctk.CTkLabel(
            loading_container,
            text="Preparing data...",
            font=("Arial", 12),
            text_color=("gray40", "gray60")
        )
        self.status_label.pack(pady=(0, 20))

        # Add modern progress bar
        self.progress_bar = ctk.CTkProgressBar(
            loading_container,
            width=300,
            height=15,
            corner_radius=10,
            mode="determinate",
            progress_color=("#2FB344", "#1D6F2B"),
            border_width=1,
            border_color=("gray70", "gray30")
        )
        self.progress_bar.pack(pady=(0, 15))
        self.progress_bar.set(0)

        # Add percentage indicator
        self.progress_label = ctk.CTkLabel(
            loading_container,
            text="0%",
            font=("Arial", 16),
            text_color=("gray10", "gray90")
        )
        self.progress_label.pack(pady=(0, 30))

    def update_progress(self, value, percent_text):
        """Update progress bar and label"""
        if self.progress_bar and self.progress_label:
            self.progress_bar.set(value)
            self.progress_label.configure(text=percent_text)

    def remove_loading_ui(self):
        """Remove the loading overlay"""
        if self.loading_frame:
            self.loading_frame.destroy()
            self.loading_frame = None
            self.progress_bar = None
            self.progress_label = None

    def load_station_data(self):
        """Load and display data for the selected station"""
        # Clear any existing data table
        for widget in self.data_container.winfo_children():
            widget.destroy()

        station_name = self.selected_station.get()
        print(f"Loading data for station: {station_name}")

        # Get data from cache
        if station_name not in self._data_cache['station_data']:
            print(f"No cached data found for {station_name}")
            return

        df = self._data_cache['station_data'][station_name]
        print(f"Using cached data with {len(df)} rows")

        # Calculate temperature changes
        df = self.calculate_temperature_changes(df)
        
        # Add Temp_Change to display columns after Temperature
        temp_index = self.display_columns.index('Temperature')
        self.display_columns.insert(temp_index + 1, 'Temp_Change')

        # Reorder columns to put Station at the end
        columns = list(df.columns)
        if 'Temperature' in columns:
            columns.remove('Temperature')
            columns.append('Temperature')
            df = df[columns]
        if 'Temp_Change' in columns:
            columns.remove('Temp_Change')
            columns.append('Temp_Change')
            df = df[columns]
        if 'Station' in columns:
            columns.remove('Station')
            columns.append('Station')
            df = df[columns]

        # Count the number of columns
        num_columns = len(df.columns)

        # Create scrollable frame for data - make this much wider
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.data_container,
            fg_color="white",
            width=1150,  # Increased width to fit all columns comfortably
            height=500
        )
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.data_container.rowconfigure(0, weight=1)
        self.data_container.columnconfigure(0, weight=1)

        # Use fixed cell width with enough space
        cell_width = 100  # Fixed comfortable width for all cells

        # Prepare grid for headers
        headers = list(df.columns)

        # Create header row
        for col_idx, col_name in enumerate(headers):
            header_frame = ctk.CTkFrame(
                self.scrollable_frame,
                fg_color="#E6E6E6",
                corner_radius=6,  # Match the cell corner radius
                height=30,  # Match the cell height
                width=cell_width
            )
            header_frame.grid(row=0, column=col_idx, sticky="nsew", padx=2, pady=2)
            header_frame.grid_propagate(False)

            header_label = ctk.CTkLabel(
                header_frame,
                text=col_name,
                font=("Arial", 10, "bold"),  # Match the cell font size
                fg_color="transparent",
                text_color="black",
                wraplength=cell_width - 8  # Adjusted wraplength
            )
            header_label.place(relx=0.5, rely=0.5, anchor="center")

        # Calculate total cells for progress tracking
        self.total_cells = len(df) * len(headers)
        self.processed_cells = 0

        # Create loading overlay only for the grid rendering process
        # The loading frame will be shown before starting the cell rendering
        self.create_loading_ui()

        # Start batch rendering of data cells
        self.start_batch_rendering(df, cell_width)

    def start_batch_rendering(self, df, cell_width):
        """Set up batch rendering of data cells"""
        # Reduce cell width to make columns narrower
        cell_width = 90  # Changed from 120 to 90

        # Rest of the existing code...
        self.is_rendering = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=0.1)

        # Fill queue with cell data
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col_name in enumerate(df.columns):
                self.render_queue.put((row_idx, col_idx, row[col_name], col_name, cell_width))

        # Start rendering thread
        self.is_rendering = True
        self.render_thread = threading.Thread(target=self._render_batch, daemon=True)
        self.render_thread.start()

    def _render_batch(self):
        """Render cells in batches to improve performance"""
        BATCH_SIZE = 30  # Adjust based on performance
        try:
            while not self.render_queue.empty() and self.is_rendering:
                batch = []
                for _ in range(BATCH_SIZE):
                    if self.render_queue.empty():
                        break
                    batch.append(self.render_queue.get_nowait())

                # Schedule UI updates on main thread
                self.after(0, self._create_batch, batch)

                # Update progress counters
                self.processed_cells += len(batch)
                progress_value = min(1.0, self.processed_cells / max(1, self.total_cells))
                progress_percent = f"{int(progress_value * 100)}%"

                # Update progress bar on main thread
                self.after(0, self.update_progress, progress_value, progress_percent)

                # Check if we're done
                if self.processed_cells >= self.total_cells or self.render_queue.empty():
                    # Remove loading UI after a short delay to ensure user sees 100%
                    self.after(500, self.remove_loading_ui)

                threading.Event().wait(0.01)

        finally:
            self.is_rendering = False
            # Ensure loading UI is removed if the thread exits abnormally
            self.after(0, self.remove_loading_ui)

    def _create_batch(self, batch):
        """Create a batch of cells on the main thread"""
        for row_idx, col_idx, value, col_name, width in batch:
            cell_color = self.get_color(col_name, value)

            cell_frame = ctk.CTkFrame(
                self.scrollable_frame,
                fg_color=cell_color,
                corner_radius=6,  
                height=30,  
                width=width,
                border_width=0
            )
            cell_frame.grid(row=row_idx + 1, column=col_idx, sticky="nsew", padx=2, pady=1)  # Reduced padding
            cell_frame.grid_propagate(False)

            # Format numeric values to display cleanly
            display_value = self.format_cell_value(value, col_name)

            cell_label = ctk.CTkLabel(
                cell_frame,
                text=display_value,
                font=("Arial", 10),  # Reduced font size from 11 to 10
                fg_color="transparent",
                text_color="black"
            )
            cell_label.place(relx=0.5, rely=0.5, anchor="center")

    def format_cell_value(self, value, col_name):
        """Format cell values to be more concise"""
        if col_name == 'Date':
            # Convert datetime to date string without time
            try:
                if isinstance(value, pd.Timestamp):
                    return value.strftime('%Y-%m-%d')
                elif isinstance(value, datetime.datetime):
                    return value.date().strftime('%Y-%m-%d')
                return str(value).split()[0]  # Fallback: split at first space
            except:
                return str(value)
        
        if value == "Nan" or pd.isna(value):
            return "Nan"

        # Format temperature change values
        if col_name == "Temp_Change":
            try:
                num_value = float(value)
                if num_value > 0:
                    return f"+{num_value:.1f}°C"
                return f"{num_value:.1f}°C"
            except:
                return str(value)

        # Format numeric values to limit decimal places
        if col_name in ["pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)",
                        "Inorganic Phosphate (mg/L)", "Dissolved Oxygen (mg/L)",
                        "Temperature", "Chlorophyll-a (ug/L)"]:
            try:
                num_value = float(value)
                # Use fewer decimal places for small numbers
                if abs(num_value) < 0.1:
                    return f"{num_value:.3f}"
                elif abs(num_value) < 10:
                    return f"{num_value:.2f}"
                else:
                    return f"{num_value:.1f}"
            except:
                pass

        # Format large numbers (like phytoplankton counts)
        if col_name == "Phytoplankton":
            try:
                num_value = int(float(value))
                if num_value >= 1000000:
                    return f"{num_value / 1000000:.1f}M"
                elif num_value >= 1000:
                    return f"{num_value / 1000:.1f}K"
                else:
                    return str(num_value)
            except:
                pass

        if col_name == "Temp_Change":
            try:
                num_value = float(value)
                if num_value > 0:
                    return f"+{num_value:.1f}°C"
                else:
                    return f"{num_value:.1f}°C"
            except:
                return str(value)

        return str(value)

    def create_legend(self):
        """Create the legend section with consistent box sizes"""
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
            "DO": [
            {"color": "#A7C7E7", "label": "Conformed with Classes A, B and C (≥ 5 mg/L)"},
            {"color": "#C3E6CB", "label": "Conformed with Class D (≥ 2 mg/L to < 5 mg/L)"},
            {"color": "#F5B7B1", "label": "Failed Guidelines (< 2 mg/L)"},
            {"color": "#D5DBDB", "label": "No data"},
            ],

            "Temperature": [
                {"color": "#A7C7E7", "label": "Good (≤ 2°C)"},
                {"color": "#C3E6CB", "label": "Moderate (3°C)"},
                {"color": "#F5B7B1", "label": "Bad (4-5°C)"},
                {"color": "#FF6B6B", "label": "Worst (≥ 6°C)"},
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
        legend_title.grid(row=0, column=0, columnspan=2, pady=(10, 15))

        # Create scrollable frame for legend to handle large legends
        legend_scrollable = ctk.CTkFrame(
            self.legend_container,
            fg_color="transparent",
            width=300,
            height=500
        )
        legend_scrollable.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.legend_container.rowconfigure(1, weight=1)
        self.legend_container.columnconfigure(0, weight=1)

        # Create two columns for legends
        left_column = ctk.CTkFrame(legend_scrollable, fg_color="transparent")
        right_column = ctk.CTkFrame(legend_scrollable, fg_color="transparent")

        left_column.grid(row=0, column=0, sticky="nw", padx=5)
        right_column.grid(row=0, column=1, sticky="nw", padx=5)
        legend_scrollable.columnconfigure(0, weight=1)
        legend_scrollable.columnconfigure(1, weight=1)

        # Split legends between two columns
        legend_items = list(legend_data.items())
        mid_point = len(legend_items) // 2

        # Define consistent box size for all legend items
        box_width = 20  # Width in pixels
        box_height = 20  # Height in pixels

        # Fill left column
        for idx, (parameter, items) in enumerate(legend_items[:mid_point]):
            param_frame = ctk.CTkFrame(left_column, fg_color="transparent")
            param_frame.pack(side="top", fill="x", pady=(0, 15))

            title_label = ctk.CTkLabel(
                param_frame,
                text=f"{parameter} Legend:",
                font=("Arial", 14, "bold")
            )
            title_label.pack(side="top", anchor="w", pady=(0, 5))

            for item in items:
                row_container = ctk.CTkFrame(param_frame, fg_color="transparent")
                row_container.pack(side="top", fill="x", padx=5, pady=2)

                # Create a canvas for consistent color box size
                color_canvas = tk.Canvas(
                    row_container,
                    width=box_width,
                    height=box_height,
                    highlightthickness=1,
                    highlightbackground="black"
                )
                color_canvas.pack(side="left", padx=5)

                # Fill the canvas with the color
                color_canvas.create_rectangle(
                    0, 0, box_width, box_height,
                    fill=item["color"],
                    outline="black",
                    width=1
                )

                label = ctk.CTkLabel(
                    row_container,
                    text=item["label"],
                    anchor="w",
                    font=("Arial", 11),
                    wraplength=120,
                    justify="left"
                )
                label.pack(side="left", fill="x", expand=True)

        # Fill right column
        for idx, (parameter, items) in enumerate(legend_items[mid_point:]):
            param_frame = ctk.CTkFrame(right_column, fg_color="transparent")
            param_frame.pack(side="top", fill="x", pady=(0, 15))  # Add pack() here

            title_label = ctk.CTkLabel(  # Create new title_label for each parameter
                param_frame,
                text=f"{parameter} Legend:",
                font=("Arial", 14, "bold")
            )
            title_label.pack(side="top", anchor="w", pady=(0, 5))

            for item in items:
                row_container = ctk.CTkFrame(param_frame, fg_color="transparent")
                row_container.pack(side="top", fill="x", padx=5, pady=2)

                # Create a canvas for consistent color box size
                color_canvas = tk.Canvas(
                    row_container,
                    width=box_width,
                    height=box_height,
                    highlightthickness=1,
                    highlightbackground="black"
                )
                color_canvas.pack(side="left", padx=5)

                # Fill the canvas with the color
                color_canvas.create_rectangle(
                    0, 0, box_width, box_height,
                    fill=item["color"],
                    outline="black",
                    width=1
                )

                label = ctk.CTkLabel(
                    row_container,
                    text=item["label"],
                    anchor="w",
                    font=("Arial", 11),
                    wraplength=120,
                    justify="left"
                )
                label.pack(side="left", fill="x", expand=True)

    def get_color(self, param, value):
        """Returns the appropriate color based on parameter value ranges."""
        try:
            if pd.isna(value) or value == "Nan":
                return "#D5DBDB"  # Light gray for NaN values
                
            # Temperature Change color coding
            if param == "Temp_Change":
                value = float(value)
                if value <= 2:
                    return "#A7C7E7"  # Good (≤ 2°C)
                elif value <= 3:
                    return "#C3E6CB"  # Moderate (3°C)
                elif value <= 5:
                    return "#F5B7B1"  # Bad (4-5°C)
                else:
                    return "#FF6B6B"  # Worst (≥ 6°C)

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
                    if value < 50000:
                        return "#FFFFFF"  # No Bloom
                    elif 50000 <= value < 100000:
                        return "#FFB6C1"  # Minor Bloom
                    elif 100000 <= value < 500000:
                        return "#FF69B4"  # Moderate Bloom
                    else:
                        return "#FF1493"  # Massive Bloom
                    
                elif param == "Dissolved Oxygen (mg/L)":
                    if value >= 5:
                        return "#A7C7E7"  # Conformed with Classes A, B and C
                    elif 2 <= value < 5:
                        return "#C3E6CB"  # Conformed with Class D
                    else:
                        return "#F5B7B1"  # Failed Guidelines

            return "#FFFFFF"  # Default white for other columns

        except:
            return "#D5DBDB"  # Light gray for any errors

    def on_station_change(self, selected_station):
        """Handle station selection change."""
        print(f"Selected station: {selected_station}")
        self.selected_station.set(selected_station)
        self.load_station_data()

    def destroy(self):
        """Clean up resources when widget is destroyed"""
        self.is_rendering = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        super().destroy()

    def calculate_temperature_changes(self, df):
        """Calculate temperature changes from previous month for each row"""
        # Convert date to datetime for sorting
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Calculate temperature change
        df['Temp_Change'] = df['Temperature'].astype(float).diff()
        
        # Replace first row's NaN with 0
        df['Temp_Change'] = df['Temp_Change'].fillna(0)
        
        return df

