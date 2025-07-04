import customtkinter as ctk
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator


class DashboardPage(ctk.CTkFrame):
    # Class variables to cache data between instances
    _data_cache = {
        'full_df': None,
        'station_data': {},
        'initialized': False
    }

    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color="#FFFFFF")  # Force white background
        self.parent = parent

        # Set parent's background color to white as well using proper CustomTkinter attribute
        if hasattr(parent, 'configure'):
            try:
                parent.configure(fg_color="#FFFFFF")  # Use fg_color instead of bg
            except Exception as e:
                print(f"Warning: Could not configure parent: {e}")

        # Define the station names from your dataset
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

        # Path to the single CSV file containing all stations data
        self.csv_file = "train/merged_stations.xlsx"

        # Instance variables to track UI state
        self.monthly_canvas = None
        self.yearly_canvas = None
        self.monthly_df = None
        self.yearly_df = None

        # Shared figure dimensions for both visualizations
        self.fig_width = 8.0
        self.fig_height = 5.0
        self.fig_dpi = 72

        # Store dimensions for synchronization
        self.current_fig_width = self.fig_width
        self.current_fig_height = self.fig_height

        # Make the main frame expand to fill available space
        self.pack_propagate(False)

        # Configure main frame row and column weights to allow expansion
        self.rowconfigure(1, weight=1)
        for i in range(6):
            self.columnconfigure(i, weight=1)

        # Load data if not already loaded
        if not self._data_cache['initialized']:
            self.preload_data()
        else:
            print("Using cached data")

        # Create widget layout
        self.create_widgets()

        # Set visible flag to track when page is showing
        self.is_visible = True

    # ========== DATA MANAGEMENT METHODS ==========

    def preload_data(self):
        """Load all data only once and cache it"""
        print("Preloading all station data...")
        # Load the full dataset once
        self._data_cache['full_df'] = self.load_all_data(self.csv_file)

        # Pre-filter data for each station and cache it
        for station_name, station_code in self.station_names.items():
            filtered_data = self._data_cache['full_df'][self._data_cache['full_df']["Station"] == station_code].copy()
            self._data_cache['station_data'][station_name] = filtered_data
            print(f"Cached {len(filtered_data)} rows for {station_name}")

        self._data_cache['initialized'] = True
        print("Data preloading complete")

    def load_all_data(self, filename):
        """Load the full dataset containing all stations"""
        try:
            # Load CSV file
            df = pd.read_excel(filename)

            # Standardize Date column format
            try:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            except:
                print(f"Warning: Unable to parse dates in {filename}")

            # Create year and month columns
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month

            # Convert numeric columns and handle NaN values
            # Adjust column names to match your new dataset
            numeric_columns = [
                "pH (units)",
                "Ammonia (mg/L)",
                "Nitrate (mg/L)",
                "Inorganic Phosphate (mg/L)",
                "Dissolved Oxygen (mg/L)",
                "Temperature",
                "Chlorophyll-a (ug/L)"
            ]

            # Create shorter names for use in the interface
            self.column_mappings = {
                "pH (units)": "pH",
                "Ammonia (mg/L)": "Ammonia",
                "Nitrate (mg/L)": "Nitrate",
                "Inorganic Phosphate (mg/L)": "Phosphate",
                "Dissolved Oxygen (mg/L)": "DO",
                "Temperature": "Temperature",
                "Chlorophyll-a (ug/L)": "Chlorophyll-a",
                # Add "Phytoplankton" if it exists in your dataset
            }

            # Add shortened column names for easier reference
            for full_name, short_name in self.column_mappings.items():
                if full_name in df.columns:
                    df[short_name] = df[full_name]

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            print(f"Loaded {len(df)} rows from {filename}")

            return df

        except Exception as e:
            print(f"Error loading CSV {filename}: {e}")
            return pd.DataFrame()

    def filter_by_station(self, station_name):
        """Get cached data for the selected station"""
        try:
            # Use cached data if available
            station_display_name = next((name for name, code in self.station_names.items() if code == station_name),
                                        None)

            if station_display_name and station_display_name in self._data_cache['station_data']:
                print(f"Using cached data for {station_display_name}")
                return self._data_cache['station_data'][station_display_name]
            else:
                # Fallback to filtering from full dataset if not in cache
                filtered_df = self._data_cache['full_df'][self._data_cache['full_df']["Station"] == station_name].copy()
                print(f"Filtered to {len(filtered_df)} rows for {station_name} (not from cache)")
                return filtered_df
        except Exception as e:
            print(f"Error retrieving data for {station_name}: {e}")
            return pd.DataFrame()

    # ========== UI SETUP METHODS ==========

    def create_widgets(self):
        # Header - move it directly after the sidebar with no extra padding
        dashboardlb = ctk.CTkLabel(self, text="DASHBOARD", font=("Segoe UI", 25, "bold"), text_color="#2c3e50")
        dashboardlb.grid(row=0, column=1, padx=20, pady=20, sticky="nw", columnspan=6)  # Reduced left padding

        # Configure the main window grid with specific weights
        self.columnconfigure(0, weight=0, minsize=0)  # Sidebar width only, no extra space
        self.columnconfigure(1, weight=1)  # Start content immediately after sidebar
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)
        self.columnconfigure(6, weight=1)

        # Create a frame that will contain both panels
        main_content_frame = ctk.CTkFrame(self, fg_color="#FFFFFF")
        main_content_frame.grid(row=1, column=1, columnspan=6, sticky="nsew", padx=10, pady=10)

        # Configure the main content frame
        main_content_frame.columnconfigure(0, weight=5)
        main_content_frame.columnconfigure(1, weight=5)
        main_content_frame.rowconfigure(0, weight=1)

        # Create left frame for monthly data
        self.monthly_frame = ctk.CTkFrame(main_content_frame, fg_color="#FFFFFF", border_width=1,
                                          border_color="#CCCCCC", corner_radius=4)
        self.monthly_frame.grid(row=0, column=0, sticky="nsw", padx=5, pady=5)

        # Create right frame for yearly data
        self.yearly_frame = ctk.CTkFrame(main_content_frame, fg_color="#FFFFFF", border_width=1,
                                         border_color="#CCCCCC", corner_radius=4)
        self.yearly_frame.grid(row=0, column=1, sticky="nsw", padx=5, pady=5)

        # Ensure both frames expand to fill the available space
        self.monthly_frame.pack_propagate(False)
        self.yearly_frame.pack_propagate(False)

        # Set up the monthly frame
        self.setup_monthly_frame()

        # Set up the yearly frame
        self.setup_yearly_frame()

        # Bind resize event to update canvases
        self.bind("<Configure>", self.on_window_resize)

        # Store current state for comparison
        self.last_width = self.winfo_width()
        self.last_height = self.winfo_height()

    def setup_monthly_frame(self):
        """Set up the monthly data frame (left side)"""
        # Configure grid weights to ensure proper resizing
        for i in range(6):  # Reduced row count after combining controls
            self.monthly_frame.rowconfigure(i, weight=0)  # Don't expand rows
        self.monthly_frame.rowconfigure(4, weight=2)  # Only expand canvas row

        for i in range(4):  # Assuming 4 columns
            self.monthly_frame.columnconfigure(i, weight=1)

        # Title
        monthly_title = ctk.CTkLabel(self.monthly_frame, text="Monthly Data", font=("Segoe UI", 16, "bold"),
                                     text_color="#2c3e50")
        monthly_title.grid(row=0, column=0, padx=10, pady=10, columnspan=4, sticky="nw")

        # Create a single row frame for all controls
        controls_frame = ctk.CTkFrame(self.monthly_frame, fg_color="#FFFFFF")
        controls_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=0, sticky="ew")

        # Configure the columns for the controls frame
        controls_frame.columnconfigure(0, weight=0)  # Station label - fixed width
        controls_frame.columnconfigure(1, weight=0)  # Station dropdown - fixed width
        controls_frame.columnconfigure(2, weight=0)  # Spacer - fixed width
        controls_frame.columnconfigure(3, weight=0)  # Year label - fixed width
        controls_frame.columnconfigure(4, weight=0)  # Year dropdown - fixed width
        controls_frame.columnconfigure(5, weight=1)  # Flexible spacer - expands to fill

        # Station selection label and dropdown in one row
        station_label = ctk.CTkLabel(controls_frame, text="Select Station:", font=("Segoe UI", 12),
                                     text_color="#2c3e50")
        station_label.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")

        self.monthly_station_var = ctk.StringVar(value="Station 1")  # Default station

        # Add callback for auto-update
        self.monthly_station_var.trace_add("write", lambda *args: self.update_monthly_station())

        # Create dropdown with more appropriate width
        self.monthly_station_dropdown = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.monthly_station_var,
            values=list(self.station_names.keys()),
            width=100,  # Reduced width as requested
            fg_color="#1f6aa5",  # Blue button background
            button_color="#1f6aa5",  # Primary blue for button
            button_hover_color="#3680bb",  # Slightly lighter blue for hover
            text_color="#FFFFFF",  # White text for readability on blue button
            dropdown_fg_color="#FFFFFF",  # White background for dropdown menu
            dropdown_hover_color="#e6f0f7",  # Light blue hover for menu items
            dropdown_text_color="#2c3e50"  # Dark text for dropdown items on white
        )
        self.monthly_station_dropdown.grid(row=0, column=1, padx=0, pady=5, sticky="w")

        # Filter data for initial station - use the display name to get cached data
        station_key = self.monthly_station_var.get()
        self.monthly_df = self._data_cache['station_data'].get(station_key, pd.DataFrame())

        # Year selection - add to the same row with spacing
        year_label = ctk.CTkLabel(controls_frame, text="Select Year:", font=("Segoe UI", 12), text_color="#2c3e50")
        year_label.grid(row=0, column=3, padx=(20, 10), pady=5, sticky="w")

        self.monthly_year_var = ctk.StringVar()

        # Get available years from data
        available_years = sorted(self.monthly_df["Year"].dropna().unique()) if not self.monthly_df.empty else []
        default_year = "2016" if 2016 in available_years else str(int(available_years[0])) if len(
            available_years) > 0 else "2016"
        self.monthly_year_var.set(default_year)  # Default year

        # Add callback for auto-update
        self.monthly_year_var.trace_add("write", lambda *args: self.display_monthly_data())

        # Use available years or fallback to range
        years_for_dropdown = [str(int(y)) for y in available_years] if len(available_years) > 0 else [str(y) for y in
                                                                                                      range(2016, 2026)]

        # Year dropdown with appropriate width
        self.monthly_year_dropdown = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.monthly_year_var,
            values=years_for_dropdown,
            width=80,  # Narrower to fit years
            fg_color="#1f6aa5",  # Blue button background
            button_color="#1f6aa5",  # Primary blue for button
            button_hover_color="#3680bb",  # Slightly lighter blue for hover
            text_color="#FFFFFF",  # White text for readability on blue button
            dropdown_fg_color="#FFFFFF",  # White background for dropdown menu
            dropdown_hover_color="#e6f0f7",  # Light blue hover for menu items
            dropdown_text_color="#2c3e50"  # Dark text for dropdown items on white
        )
        self.monthly_year_dropdown.grid(row=0, column=4, padx=0, pady=5, sticky="w")

        # Parameter selection
        parameter_names = list(self.column_mappings.values())
        self.monthly_param_var = ctk.StringVar(value="pH")  # Default selection

        # Add callback for auto-update
        self.monthly_param_var.trace_add("write", lambda *args: self.display_monthly_data())

        param_label = ctk.CTkLabel(self.monthly_frame, text="Select Parameter:", font=("Segoe UI", 12),
                                   text_color="#2c3e50")
        param_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        # Create a more responsive frame for radio buttons with white background
        radio_frame = ctk.CTkFrame(self.monthly_frame, fg_color="#FFFFFF")
        radio_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Configure grid columns to distribute space evenly
        for i in range(len(parameter_names)):
            radio_frame.columnconfigure(i, weight=1)

        # Define spacing based on parameter count
        radio_padding = 5 if len(parameter_names) > 5 else 8

        # Create radio buttons with better spacing and responsive layout
        for i, param in enumerate(parameter_names):
            rb = ctk.CTkRadioButton(radio_frame, text=param, variable=self.monthly_param_var, value=param,
                                    font=("Segoe UI", 10), text_color="#2c3e50",
                                    fg_color="#1f6aa5", hover_color="#1f6aa5", border_color="#c4cfd8")
            # Position radio buttons evenly
            rb.grid(row=0, column=i, padx=radio_padding, pady=5, sticky="w")

        # Create canvas container with white background
        self.monthly_canvas_container = ctk.CTkFrame(self.monthly_frame, fg_color="#FFFFFF")
        self.monthly_canvas_container.grid(row=4, column=0, columnspan=4, padx=10, pady=(5, 5), sticky="nsew")

        # Create the canvas frame where the matplotlib figure will be placed
        self.monthly_canvas_frame = ctk.CTkFrame(self.monthly_canvas_container, fg_color="#FFFFFF")
        self.monthly_canvas_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Error message
        self.monthly_error = ctk.CTkLabel(self.monthly_frame, text="", font=("Segoe UI", 12), text_color="#5d7285")
        self.monthly_error.grid(row=5, column=0, columnspan=4, padx=10, pady=(5, 15), sticky="w")

    def setup_yearly_frame(self):
        """Set up the yearly data frame (right side)"""
        # Configure grid weights to match the monthly frame for alignment
        for i in range(6):  # Reduced row count after combining controls
            self.yearly_frame.rowconfigure(i, weight=0)
        self.yearly_frame.rowconfigure(4, weight=2)  # Only expand canvas row - match monthly frame

        for i in range(4):  # Using 4 columns
            self.yearly_frame.columnconfigure(i, weight=1)

        # Title
        yearly_title = ctk.CTkLabel(self.yearly_frame, text="Yearly Data", font=("Segoe UI", 16, "bold"),
                                    text_color="#2c3e50")
        yearly_title.grid(row=0, column=0, padx=10, pady=10, columnspan=4, sticky="nw")

        # Create a single row frame for all controls
        controls_frame = ctk.CTkFrame(self.yearly_frame, fg_color="#FFFFFF")
        controls_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=0, sticky="ew")

        # Configure the columns for the controls frame
        controls_frame.columnconfigure(0, weight=0)  # Station label - fixed width
        controls_frame.columnconfigure(1, weight=0)  # Station dropdown - fixed width
        controls_frame.columnconfigure(2, weight=0)  # Spacer - fixed width
        controls_frame.columnconfigure(3, weight=0)  # Year Range label - fixed width
        controls_frame.columnconfigure(4, weight=0)  # From label - fixed width
        controls_frame.columnconfigure(5, weight=0)  # From dropdown - fixed width
        controls_frame.columnconfigure(6, weight=0)  # To label - fixed width
        controls_frame.columnconfigure(7, weight=0)  # To dropdown - fixed width
        controls_frame.columnconfigure(8, weight=1)  # Flexible spacer - expands to fill

        # Station selection label and dropdown in one row
        station_label = ctk.CTkLabel(controls_frame, text="Select Station:", font=("Segoe UI", 12),
                                     text_color="#2c3e50")
        station_label.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")

        self.yearly_station_var = ctk.StringVar(value="Station 1")  # Default station

        # Add callback for auto-update
        self.yearly_station_var.trace_add("write", lambda *args: self.update_yearly_station())

        # Create dropdown with more appropriate width
        self.yearly_station_dropdown = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.yearly_station_var,
            values=list(self.station_names.keys()),
            width=100,  # Reduced width as requested
            fg_color="#1f6aa5",  # Blue button background
            button_color="#1f6aa5",  # Primary blue for button
            button_hover_color="#3680bb",  # Slightly lighter blue for hover
            text_color="#FFFFFF",  # White text for readability on blue button
            dropdown_fg_color="#FFFFFF",  # White background for dropdown menu
            dropdown_hover_color="#e6f0f7",  # Light blue hover for menu items
            dropdown_text_color="#2c3e50"  # Dark text for dropdown items on white
        )
        self.yearly_station_dropdown.grid(row=0, column=1, padx=0, pady=5, sticky="w")

        # Filter data for initial station - use the display name to get cached data
        station_key = self.yearly_station_var.get()
        self.yearly_df = self._data_cache['station_data'].get(station_key, pd.DataFrame())

        # Get available years for the selected station
        available_years = sorted(self.yearly_df["Year"].dropna().unique()) if not self.yearly_df.empty else []
        years_for_dropdown = [str(int(y)) for y in available_years] if len(available_years) > 0 else []

        # Year Range label - add to the same row with spacing
        year_range_label = ctk.CTkLabel(controls_frame, text="Year Range:", font=("Segoe UI", 12), text_color="#2c3e50")
        year_range_label.grid(row=0, column=3, padx=(20, 10), pady=5, sticky="w")

        # Start year section - in the same row
        start_year_label = ctk.CTkLabel(controls_frame, text="From:", font=("Segoe UI", 11), text_color="#5d7285")
        start_year_label.grid(row=0, column=4, padx=(0, 5), pady=5, sticky="w")

        self.start_year_var = ctk.StringVar()
        if years_for_dropdown:
            self.start_year_var.set(years_for_dropdown[0])  # Set to earliest year

        self.start_year_dropdown = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.start_year_var,
            values=years_for_dropdown if years_for_dropdown else [""],
            width=70,  # Fixed width for clean alignment
            fg_color="#1f6aa5",  # Blue button background
            button_color="#1f6aa5",  # Primary blue for button
            button_hover_color="#3680bb",  # Slightly lighter blue for hover
            text_color="#FFFFFF",  # White text for readability on blue button
            dropdown_fg_color="#FFFFFF",  # White background for dropdown menu
            dropdown_hover_color="#e6f0f7",  # Light blue hover for menu items
            dropdown_text_color="#2c3e50"  # Dark text for dropdown items on white
        )
        self.start_year_dropdown.grid(row=0, column=5, padx=(0, 15), pady=5, sticky="w")

        # End year section - in the same row
        end_year_label = ctk.CTkLabel(controls_frame, text="To:", font=("Segoe UI", 11), text_color="#5d7285")
        end_year_label.grid(row=0, column=6, padx=(0, 5), pady=5, sticky="w")

        self.end_year_var = ctk.StringVar()
        if years_for_dropdown:
            self.end_year_var.set(years_for_dropdown[-1])  # Set to latest year

        self.end_year_dropdown = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.end_year_var,
            values=years_for_dropdown if years_for_dropdown else [""],
            width=70,  # Fixed width for clean alignment
            fg_color="#1f6aa5",  # Blue button background
            button_color="#1f6aa5",  # Primary blue for button
            button_hover_color="#3680bb",  # Slightly lighter blue for hover
            text_color="#FFFFFF",  # White text for readability on blue button
            dropdown_fg_color="#FFFFFF",  # White background for dropdown menu
            dropdown_hover_color="#e6f0f7",  # Light blue hover for menu items
            dropdown_text_color="#2c3e50"  # Dark text for dropdown items on white
        )
        self.end_year_dropdown.grid(row=0, column=7, padx=0, pady=5, sticky="w")

        # Add callbacks for auto-update when year range changes
        self.start_year_var.trace_add("write", lambda *args: self.display_yearly_data())
        self.end_year_var.trace_add("write", lambda *args: self.display_yearly_data())

        # Parameter selection with checkboxes
        param_label = ctk.CTkLabel(self.yearly_frame, text="Select Parameters:", font=("Segoe UI", 12),
                                   text_color="#2c3e50")
        param_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        parameter_names = list(self.column_mappings.values())

        # Default parameters to select
        default_params = ["Nitrate", "Phosphate", "Ammonia"]

        # Initialize checkboxes with callback function
        self.param_var_cb = {}
        self.param_checkboxes = []

        # Create a more responsive frame for checkboxes
        checkbox_frame = ctk.CTkFrame(self.yearly_frame, fg_color="#FFFFFF")
        checkbox_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Configure grid columns to distribute space evenly
        for i in range(len(parameter_names)):
            checkbox_frame.columnconfigure(i, weight=1)

        # Function to handle checkbox state changes
        def param_checkbox_callback():
            # Automatically update the graph when checkbox state changes
            self.display_yearly_data()

        # Define spacing based on parameter count - match with radio buttons
        checkbox_padding = 5 if len(parameter_names) > 5 else 8

        # Create variables for parameters and checkboxes with responsive layout
        for i, param in enumerate(parameter_names):
            # Set default selected for specific parameters
            is_default = param in default_params
            cb_var = ctk.IntVar(value=1 if is_default else 0)
            cb_var.trace_add("write", lambda *args: param_checkbox_callback())  # Add callback
            self.param_var_cb[param] = cb_var

            cb = ctk.CTkCheckBox(checkbox_frame, text=param, variable=self.param_var_cb[param], font=("Segoe UI", 10),
                                 text_color="#2c3e50", fg_color="#1f6aa5", hover_color="#1f6aa5",
                                 border_color="#c4cfd8")
            # Align with radio buttons in the monthly frame
            cb.grid(row=0, column=i, padx=checkbox_padding, pady=5, sticky="w")

        # Create canvas container with white background
        self.yearly_canvas_container = ctk.CTkFrame(self.yearly_frame, fg_color="#FFFFFF")
        self.yearly_canvas_container.grid(row=4, column=0, columnspan=4, padx=10, pady=(5, 5), sticky="nsew")

        # Create the canvas frame with white background
        self.yearly_canvas_frame = ctk.CTkFrame(self.yearly_canvas_container, fg_color="#FFFFFF")
        self.yearly_canvas_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Error message
        self.yearly_error = ctk.CTkLabel(self.yearly_frame, text="", font=("Segoe UI", 12), text_color="#5d7285")
        self.yearly_error.grid(row=5, column=0, columnspan=4, padx=10, pady=(5, 15), sticky="w")

        # Display initial charts
        self.display_monthly_data()
        self.display_yearly_data()

    # ========== The rest of the methods remain unchanged ==========

    def on_window_resize(self, event):
        """Handle window resize events to update the graphs"""
        # Only respond if size has significantly changed (avoid small fluctuations)
        width_changed = hasattr(self, 'last_width') and abs(self.last_width - event.width) > 10
        height_changed = hasattr(self, 'last_height') and abs(self.last_height - event.height) > 10

        if not hasattr(self, 'last_width') or width_changed or height_changed:
            self.last_width = event.width
            self.last_height = event.height

            # Allow more time for the UI to update before redrawing
            self.after(200, self.update_graphs)

    def update_graphs(self):
        """Update both graphs with current size information"""
        # Only update graphs if the page is visible
        if hasattr(self, 'is_visible') and self.is_visible:
            # Calculate shared dimensions based on available space
            self.calculate_shared_figure_dimensions()

            # Update and redraw graphs with a larger delay between them
            # This prevents potential display issues and ensures proper rendering
            self.display_monthly_data()
            self.after(150, self.display_yearly_data())  # Increased delay

    def calculate_shared_figure_dimensions(self):
        """Calculate shared dimensions for both graphs"""
        # Get the dimensions of both containers
        monthly_width = self.monthly_canvas_container.winfo_width()
        monthly_height = self.monthly_canvas_container.winfo_height()

        yearly_width = self.yearly_canvas_container.winfo_width()
        yearly_height = self.yearly_canvas_container.winfo_height()

        # Use the smaller dimensions to ensure both graphs have the same size
        container_width = min(monthly_width, yearly_width)
        container_height = min(monthly_height, yearly_height)

        # Ensure minimum dimensions and buffer zone to prevent border cut-offs
        if container_width < 50 or container_height < 50:
            # Window probably still initializing, use default size
            self.current_fig_width = self.fig_width
            self.current_fig_height = self.fig_height
        else:
            # Calculate figure size in inches based on container size and DPI
            # Reduce to 85% of container width to add buffer space and prevent cut-offs
            self.current_fig_width = max(5, container_width / self.fig_dpi * 0.95)  # Increased from 0.85
            self.current_fig_height = max(4, container_height / self.fig_dpi * 0.95)  # Increased from 0.85
        print(f"Using shared dimensions: {self.current_fig_width}x{self.current_fig_height} inches")

    def update_monthly_station(self):
        """Update data when monthly station selection changes"""
        station_name = self.monthly_station_var.get()

        try:
            # Get cached data for this station
            self.monthly_df = self._data_cache['station_data'].get(station_name, pd.DataFrame())

            if self.monthly_df.empty:
                self.monthly_error.configure(text=f"No data available for {station_name}")
                return

            # Update year dropdown with available years for this station
            available_years = sorted(self.monthly_df["Year"].dropna().unique())

            if len(available_years) > 0:
                # Update the year dropdown menu
                year_strings = [str(int(y)) for y in available_years]
                self.monthly_year_dropdown.configure(values=year_strings)

                # Set to first available year
                self.monthly_year_var.set(str(int(available_years[0])))
            else:
                self.monthly_error.configure(text=f"No year data available for {station_name}")

        except Exception as e:
            print(f"Error updating monthly station: {e}")
            self.monthly_error.configure(text=f"Error loading station data: {str(e)}")

    def update_yearly_station(self):
        """Update data when yearly station selection changes"""
        station_name = self.yearly_station_var.get()

        try:
            # Get cached data for this station
            self.yearly_df = self._data_cache['station_data'].get(station_name, pd.DataFrame())

            if self.yearly_df.empty:
                self.yearly_error.configure(text=f"No data available for {station_name}")
                return

            # Update year dropdowns with available years for this station
            available_years = sorted(self.yearly_df["Year"].dropna().unique())

            if len(available_years) > 0:
                # Update the year dropdown values
                year_strings = [str(int(y)) for y in available_years]
                self.start_year_dropdown.configure(values=year_strings)
                self.end_year_dropdown.configure(values=year_strings)

                # Set to first and last available years
                self.start_year_var.set(str(int(available_years[0])))
                self.end_year_var.set(str(int(available_years[-1])))
            else:
                self.yearly_error.configure(text=f"No year data available for {station_name}")

        except Exception as e:
            print(f"Error updating yearly station: {e}")
            self.yearly_error.configure(text=f"Error loading station data: {str(e)}")

    # ========== VISUALIZATION METHODS ==========

    def display_monthly_data(self):
        """Display monthly data in the left panel with shared sizing"""
        self.monthly_error.configure(text="")

        try:
            # Use the shared dimensions calculated earlier
            dynamic_width = self.current_fig_width
            dynamic_height = self.current_fig_height

            selected_year = int(self.monthly_year_var.get()) if self.monthly_year_var.get() else None
            selected_param = self.monthly_param_var.get()
            selected_station = self.monthly_station_var.get()

            if not selected_param:
                self.monthly_error.configure(text="No Parameter Selected.")
                return

            if selected_year is None:
                self.monthly_error.configure(text="No Year Selected.")
                return

            filtered_df = self.monthly_df[
                self.monthly_df["Year"] == selected_year] if not self.monthly_df.empty else pd.DataFrame()

            if filtered_df.empty:
                print(f"No data available for the year {selected_year}!")
                self.monthly_error.configure(text=f"No data available for {selected_year}.")
                return

            # Clear previous canvas if it exists
            if self.monthly_canvas is not None:
                self.monthly_canvas.get_tk_widget().pack_forget()
                self.monthly_canvas = None

            # Create bar graph with shared figure size and white background
            # Add more padding to prevent cut-offs by decreasing figure size slightly
            fig = Figure(figsize=(dynamic_width, dynamic_height), dpi=self.fig_dpi, facecolor='white')
            ax = fig.add_subplot(111)

            # Filter out NaN values before plotting
            valid_data = filtered_df.dropna(subset=[selected_param, "Month"])

            if valid_data.empty:
                print(f"No valid data for {selected_param} in {selected_year}")
                self.monthly_error.configure(text=f"No valid data for {selected_param} in {selected_year}")
                return

            # Group by month and plot - this will automatically skip NaN values
            monthly_avg = valid_data.groupby("Month")[selected_param].mean()

            if not monthly_avg.empty:
                month_nums = monthly_avg.index.tolist()
                values = monthly_avg.values

                ax.bar(month_nums, values, label=selected_param, alpha=0.7,
                       color="#1f6aa5")  # Bar chart with primary blue
                ax.set_title(f"{selected_station} - Monthly {selected_param} for {selected_year}")
                ax.set_xlabel("Month")
                ax.set_ylabel(f"{selected_param} Value")
                ax.set_xticks(range(1, 13))  # Show ticks for all months even if no data
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Set fixed y-axis limits based on the parameter type
                # Get parameter range information from all data for this station/parameter
                all_station_data = self.monthly_df.dropna(subset=[selected_param])
                if not all_station_data.empty:
                    param_min = all_station_data[selected_param].min()
                    param_max = all_station_data[selected_param].max()
                    buffer = (param_max - param_min) * 0.1  # Add 10% buffer

                    # Set y-axis with some padding to prevent bars from touching top/bottom
                    ax.set_ylim([max(0, param_min - buffer), param_max + buffer])

                # Ensure consistent x-axis
                ax.set_xlim([0.5, 12.5])  # From before month 1 to after month 12

                # Create additional padding in layout to prevent cut-offs
                fig.tight_layout(pad=1.0)  # Increased padding
                fig.patch.set_visible(True)
                fig.set_facecolor('white')

                # Create the canvas with the figure and keep a reference
                self.monthly_canvas = FigureCanvasTkAgg(fig, master=self.monthly_canvas_frame)
                self.monthly_canvas.draw()

                # Make sure canvas gets a white background too
                canvas_widget = self.monthly_canvas.get_tk_widget()
                canvas_widget.configure(bg='white', highlightthickness=0)
                canvas_widget.pack(fill="both", expand=True, padx=15, pady=10)  # Added padding
            else:
                print("No valid monthly data available.")
                self.monthly_error.configure(text="No valid monthly data available.")
                return

        except Exception as e:
            print(f"Error displaying monthly data: {e}")
            self.monthly_error.configure(text=f"Error: {str(e)}")

    def display_yearly_data(self):
        """Display yearly data in the right panel with shared sizing"""
        self.yearly_error.configure(text="")

        try:
            # Use the shared dimensions calculated earlier
            dynamic_width = self.current_fig_width
            dynamic_height = self.current_fig_height

            # Get selected parameters from checkboxes
            selected_params = [param for param, var in self.param_var_cb.items() if var.get() == 1]
            selected_station = self.yearly_station_var.get()

            if not selected_params:
                self.yearly_error.configure(text="No Parameter Selected.")
                return

            # Get year range selection
            start_year = self.start_year_var.get()
            end_year = self.end_year_var.get()

            if not start_year or not end_year:
                self.yearly_error.configure(text="Please select valid year range.")
                return

            start_year = int(start_year)
            end_year = int(end_year)

            # Validate year range
            if start_year > end_year:
                start_year, end_year = end_year, start_year  # Swap if in wrong order

            # Calculate year span
            year_span = end_year - start_year + 1

            # Clear previous canvas if it exists
            if self.yearly_canvas is not None:
                self.yearly_canvas.get_tk_widget().pack_forget()
                self.yearly_canvas = None

            # Create the plot with shared figure size
            fig = Figure(figsize=(dynamic_width, dynamic_height), dpi=self.fig_dpi, facecolor='white')
            ax = fig.add_subplot(111)

            # Filter data by year range - STRICTLY only include the selected years
            filtered_df = self.yearly_df[(self.yearly_df["Year"] >= start_year) &
                                         (self.yearly_df["Year"] <= end_year)]

            if filtered_df.empty:
                self.yearly_error.configure(text=f"No data available for years {start_year}-{end_year}.")
                return

            # Keep track if we successfully plotted anything
            plotted_any = False

            # Plot each parameter separately - this approach skips NaN values automatically
            colors = ['#1f6aa5', '#ff9800', '#4caf50', '#9c27b0', '#e91e63', '#795548', '#607d8b']

            # Create a dictionary to store min/max values for each parameter
            param_ranges = {}

            for i, param in enumerate(selected_params):
                # Drop NaN values for this parameter and year
                valid_data = filtered_df.dropna(subset=[param, "Year"])

                if valid_data.empty:
                    continue

                # For shorter timespans, use more detailed temporal data
                if year_span <= 3:
                    # Group by year and month for finer detail
                    # Make an explicit copy of the data to avoid SettingWithCopyWarning
                    detailed_data = valid_data.copy()

                    # Use both Year and Month for x-axis
                    if 'Month' in detailed_data.columns and not detailed_data['Month'].isna().all():
                        # Create YearMonth column - now safely on a copy
                        detailed_data.loc[:, 'YearMonth'] = detailed_data['Year'] + (detailed_data['Month'] / 12)

                        # Sort by YearMonth for proper line plotting
                        time_series = detailed_data.sort_values(by='YearMonth')

                        # For very short timespans, plot individual data points with connecting lines
                        color_idx = i % len(colors)
                        ax.plot(time_series['YearMonth'], time_series[param], marker='o',
                                linestyle='-', linewidth=1, markersize=4,
                                label=param, color=colors[color_idx], alpha=0.8)

                        # Store min/max for this parameter
                        param_ranges[param] = (time_series[param].min(), time_series[param].max())

                        plotted_any = True
                    else:
                        # Fallback to yearly averages if no monthly data
                        yearly_avg = valid_data.groupby("Year")[param].mean()

                        if not yearly_avg.empty:
                            color_idx = i % len(colors)
                            ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linestyle='-',
                                    label=param, color=colors[color_idx])

                            # Store min/max for this parameter
                            param_ranges[param] = (yearly_avg.min(), yearly_avg.max())

                            plotted_any = True
                else:
                    # For longer timespans, use yearly averages
                    yearly_avg = valid_data.groupby("Year")[param].mean()

                    if not yearly_avg.empty:
                        color_idx = i % len(colors)
                        ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linestyle='-',
                                label=param, color=colors[color_idx])

                        # Store min/max for this parameter
                        param_ranges[param] = (yearly_avg.min(), yearly_avg.max())

                        plotted_any = True

            if not plotted_any:
                print("No valid data available for the selected parameters!")
                self.yearly_error.configure(text="No valid data available for selected parameters.")
                return

            # Set title and axis labels
            ax.set_title(f"{selected_station} - Yearly Data ({start_year}-{end_year})")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")

            # CRITICAL FIX: Strictly enforce x-axis limits to exactly match the selected year range
            # This ensures we don't see extra years beyond what was selected
            if year_span <= 3 and any('Month' in filtered_df.columns for _ in filtered_df):
                # For short timespan with monthly data

                # Format x-ticks to show year and month
                def format_date(x, pos=None):
                    year = int(x)
                    month = int(round((x - year) * 12))
                    if month == 0:
                        return f"{year}"
                    elif month % 3 == 0:  # Show quarterly labels
                        return f"{month}/Q{month // 3}"
                    return ""

                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_date))

                # Set minor ticks for months
                ax.xaxis.set_minor_locator(MultipleLocator(1 / 12))

                # CRITICAL: Set x-axis limits to precisely match selected years
                # Start from January of start_year and end at December of end_year
                ax.set_xlim([start_year, end_year + (12 / 12) - (1 / 12)])  # End just before January of the next year

                # Set major ticks at year boundaries
                ax.set_xticks(range(start_year, end_year + 1))

                # Add grid lines for clarity
                ax.grid(True, which='major', alpha=0.5)
                ax.grid(True, which='minor', alpha=0.2)
            else:
                # For yearly data or longer timespans

                # IMPORTANT: Set x-axis limits with small buffer but strictly within the selected range
                ax.set_xlim([start_year - 0.2, end_year + 0.2])

                # Set appropriate ticks
                if year_span <= 5:
                    # For short to medium timespans (≤5 years), show all years
                    ax.set_xticks(range(start_year, end_year + 1))
                else:
                    # For longer timespans, calculate appropriate step size
                    step = max(1, year_span // 5)  # Show around 5-6 ticks
                    ticks = list(range(start_year, end_year + 1, step))

                    # Make sure end_year is included
                    if end_year not in ticks:
                        ticks.append(end_year)

                    ax.set_xticks(ticks)

            # Adjust y-axis based on the data range
            if param_ranges:
                # Calculate global min and max across all selected parameters
                global_min = min([min_val for min_val, _ in param_ranges.values()])
                global_max = max([max_val for _, max_val in param_ranges.values()])

                # Add buffer for better visualization
                y_range = global_max - global_min
                buffer = y_range * 0.1  # 10% buffer

                # Set y limits with buffer, ensuring we don't go below zero if all values are positive
                y_min = max(0, global_min - buffer) if global_min >= 0 else global_min - buffer
                y_max = global_max + buffer

                ax.set_ylim([y_min, y_max])

            # Configure grid and legend
            ax.grid(True, alpha=0.3)

            # Place legend in the best position based on available space
            # Use bbox_to_anchor to ensure it's placed properly within the figure
            if len(selected_params) > 3:
                # For many parameters, place below with multiple columns
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(selected_params)))
            else:
                # For fewer parameters, try to place within the plot area
                ax.legend(loc='best')

            # Format x-ticks as integers (no decimal points for years)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

            # Increased padding in the layout to prevent cut-offs
            fig.tight_layout(pad=1.0)
            fig.patch.set_visible(True)
            fig.set_facecolor('white')

            # Create the canvas with the figure and keep a reference
            self.yearly_canvas = FigureCanvasTkAgg(fig, master=self.yearly_canvas_frame)
            self.yearly_canvas.draw()

            # Make sure canvas gets a white background too
            canvas_widget = self.yearly_canvas.get_tk_widget()
            canvas_widget.configure(bg='white', highlightthickness=0)
            canvas_widget.pack(fill="both", expand=True, padx=15, pady=10)  # Added padding

        except Exception as e:
            print(f"Error displaying yearly data: {e}")
            self.yearly_error.configure(text=f"Error: {str(e)}")

    # ========== GENERAL UTILITY METHODS ==========

    def show(self):
        """Show this frame and make sure it expands to fill available space"""
        self.grid(row=0, column=0, sticky="nsew")
        self.is_visible = True

        # Make sure the parent container allows this frame to expand
        if self.parent:
            self.parent.rowconfigure(0, weight=1)
            self.parent.columnconfigure(0, weight=1)

            # Set parent background to white using proper CustomTkinter attribute
            if hasattr(self.parent, 'configure'):
                try:
                    self.parent.configure(fg_color="#FFFFFF")  # Use fg_color instead of bg
                except Exception as e:
                    print(f"Warning: Could not configure parent: {e}")

            # Try to set the top level window's background with proper attributes
            try:
                root = self.winfo_toplevel()
                if hasattr(root, 'configure'):
                    root.configure(fg_color="#FFFFFF")  # CustomTkinter uses fg_color
            except Exception as e:
                print(f"Warning: Could not configure root: {e}")

    def hide(self):
        """Hide this frame and mark it as not visible"""
        self.grid_forget()
        self.is_visible = False