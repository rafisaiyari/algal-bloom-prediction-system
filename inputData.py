import customtkinter as ctk
import pandas as pd
import os
from datetime import datetime
import sys
import threading
import queue
import io
from contextlib import redirect_stdout
import importlib.util

# Import the audit logger - first try direct import
try:
    from audit import get_audit_logger
except ImportError:
    # If direct import fails, try to get the parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from audit import get_audit_logger
    except ImportError:
        # If still not found, create a dummy function to avoid errors
        def get_audit_logger():
            class DummyLogger:
                def log_activity(self, *args, **kwargs):
                    print("Audit logging not available")

                def log_system_event(self, *args, **kwargs):
                    print("Audit logging not available")

            return DummyLogger()

        print("Warning: Audit logger not found. Creating dummy logger.")

try:
    # Try to import the model module
    spec = importlib.util.spec_from_file_location("model_trainer", "model_trainer.py")
    model_trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_trainer)
    print("Successfully imported model_trainer.py")
except Exception as e:
    print(f"Error importing model_trainer.py: {e}")

    # Create placeholder functions to avoid errors
    class DummyModule:
        def main(*args, **kwargs):
            return "Error: Model module not loaded properly"

    model_trainer = DummyModule()

class InputDataPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None, current_username=None, user_type=None):
        # Store user information for audit logging
        self.current_username = current_username or "unknown_user"
        self.user_type = user_type or "regular"

        # Get an instance of the audit logger
        self.audit_logger = get_audit_logger()

        # Original headers list
        self.llda_headers = ["pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)",
                             "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)", "Phytoplankton"]
        # New solar headers
        self.pagasa_headers = ["Solar Mean", "Solar Max", "Solar Min"]
        # Combined headers list
        self.headers = self.llda_headers + self.pagasa_headers

        self.stations = ["I", "II", "IV", "V", "VIII", "XV", "XVI", "XVII", "XVIII"]
        self.entries = {}
        self.station_frames = {}  # To store frames for each station row
        self.station_checkboxes = {}  # To store checkboxes for filtering
        self.station_vars = {}  # To store checkbox variables

        # Variables for parameter filtering
        self.llda_var = ctk.BooleanVar(value=True)  # LLDA parameters selected by default
        self.pagasa_var = ctk.BooleanVar(value=False)  # PAGASA parameters not selected by default
        # Initialize visible headers with LLDA headers since LLDA is checked by default
        self.visible_headers = self.llda_headers.copy()  # Will store currently visible headers

        # Map station codes to their full names
        self.station_names = {
            "I": "Station_1_CWB",
            "II": "Station_2_EastB",
            "IV": "Station_4_CentralB",
            "V": "Station_5_NorthernWestB",
            "VIII": "Station_8_SouthB",
            "XV": "Station_15_SanPedro",
            "XVI": "Station_16_Sta. Rosa",
            "XVII": "Station_17_Sanctuary",
            "XVIII": "Station_18_Pagsanjan",
        }

        # Define valid ranges for each parameter
        self.valid_ranges = {
            "pH (units)": (0, 14),
            "Ammonia (mg/L)": (0, 10),
            "Nitrate (mg/L)": (0, 100),
            "Inorganic Phosphate (mg/L)": (0, 10),
            "Dissolved Oxygen (mg/L)": (0, 20),
            "Temperature": (0, 40),
            "Chlorophyll-a (ug/L)": (0, 300),
            "Phytoplankton": (0, 1000000),
            "Solar Mean": (0, 1500),
            "Solar Max": (0, 2000),
            "Solar Min": (0, 1000)
        }

        # Column positions for headers (will be used for showing/hiding)
        self.header_columns = {}

        # Define the blue color for buttons and widgets
        self.button_color = "#1f6aa5"

        # Store the default border color to use when resetting fields
        self.default_border_color = None
        self.default_text_color = ("gray10", "#DCE4EE")

        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.initialize_file_settings()

        # Define checkbox callbacks before creating widgets
        self.define_callbacks()
        self.create_widgets()

    # Define callbacks for checkboxes
    def define_callbacks(self):
        # Direct callbacks for checkboxes
        def on_llda_checkbox_clicked():
            print(f"LLDA checkbox clicked - value is now: {self.llda_var.get()}")
            self.apply_param_filter()
            self.apply_filter()  # Reapply station filter to ensure consistency

            # Log filter change
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    username=self.current_username,
                    user_type=self.user_type,
                    event_type="filter_change",
                    details=f"LLDA filter toggled to {self.llda_var.get()}"
                )

        def on_pagasa_checkbox_clicked():
            print(f"PAGASA checkbox clicked - value is now: {self.pagasa_var.get()}")
            self.apply_param_filter()
            self.apply_filter()  # Reapply station filter to ensure consistency

            # Log filter change
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    username=self.current_username,
                    user_type=self.user_type,
                    event_type="filter_change",
                    details=f"PAGASA filter toggled to {self.pagasa_var.get()}"
                )

        # Assign callbacks to the instance
        self.on_llda_checkbox_clicked = on_llda_checkbox_clicked
        self.on_pagasa_checkbox_clicked = on_pagasa_checkbox_clicked

    # Initialize basic variables
    def initialize_file_settings(self):
        # Set path to the existing Excel file in the CSV folder
        self.csv_folder = "CSV"
        self.excel_file = os.path.join(self.csv_folder, "merged_stations.xlsx")
        # Create folder if it doesn't exist
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
            print(f"Created folder: {self.csv_folder}")
        print(f"Excel file path: {self.excel_file}")

    def create_widgets(self):
        # Configure the main frame to expand with window and be scrollable
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Make the main container expandable

        # Create a canvas for scrolling
        self.canvas = ctk.CTkCanvas(self, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Add scrollbar
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas to hold all content
        self.content_frame = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Configure the content frame
        self.content_frame.columnconfigure(0, weight=1)

        # Title
        inputDatalb = ctk.CTkLabel(self.content_frame, text="INPUT DATA", font=("Segoe UI", 25, "bold"))
        inputDatalb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        # Create filters container frame with proper sizing
        self.filters_container = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.filters_container.grid(row=1, column=0, sticky="ew", padx=20, pady=10)

        # Use a strict horizontal layout with fixed positions
        self.filters_container.columnconfigure(0, weight=0)  # Station filter doesn't expand
        self.filters_container.columnconfigure(1, weight=0)  # Parameter filter doesn't expand
        self.filters_container.columnconfigure(2, weight=1)  # Empty space gets all expansion

        # Give stations plenty of room - make the frame wide enough for all 5 columns
        self.station_filter_frame = ctk.CTkFrame(self.filters_container, fg_color="transparent", width=550, height=140)
        self.station_filter_frame.grid(row=0, column=0, sticky="nw", padx=(10, 0), pady=5)
        self.station_filter_frame.grid_propagate(False)  # Prevent frame from shrinking

        # Position parameter filter with enough gap after stations section
        self.param_filter_frame = ctk.CTkFrame(self.filters_container, fg_color="transparent", width=220, height=140)
        self.param_filter_frame.grid(row=0, column=1, sticky="nw", padx=(50, 10), pady=5)
        self.param_filter_frame.grid_propagate(False)  # Prevent frame from shrinking

        # Add filter labels at the top of each column with consistent styling and identical spacing
        station_label = ctk.CTkLabel(self.station_filter_frame, text="Filter Stations:", font=("Segoe UI", 12, "bold"))
        station_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        param_label = ctk.CTkLabel(self.param_filter_frame, text="Filter Parameters:", font=("Segoe UI", 12, "bold"))
        param_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Add checkboxes for each station with specific layout
        checkbox_frame = ctk.CTkFrame(self.station_filter_frame, fg_color="transparent")
        checkbox_frame.grid(row=1, column=0, sticky="w", padx=5, pady=(0, 0))

        # Map station code to row and column position - ensure all stations have positions
        # First row: "I", "IV", "VIII", "XVI", "XVIII"
        # Second row: "II", "V", "XV", "XVII"
        station_positions = {
            "I": (0, 0),  # Station 1 - first row, first column
            "II": (1, 0),  # Station 2 - second row, first column
            "IV": (0, 1),  # Station 4 - first row, second column
            "V": (1, 1),  # Station 5 - second row, second column
            "VIII": (0, 2),  # Station 8 - first row, third column
            "XV": (1, 2),  # Station 15 - second row, third column
            "XVI": (0, 3),  # Station 16 - first row, fourth column
            "XVII": (1, 3),  # Station 17 - second row, fourth column
            "XVIII": (0, 4)  # Station 18 - first row, fifth column
        }

        # Set a consistent width for each column in the checkbox grid - give ample space
        for i in range(5):
            checkbox_frame.columnconfigure(i, minsize=100)  # Ensure plenty of room for station names

        # Create all checkboxes with full station names
        for station in self.stations:
            self.station_vars[station] = ctk.BooleanVar(value=True)
            # Set up trace to automatically apply filter when checkbox state changes
            self.station_vars[station].trace_add("write", self.on_checkbox_change)

            # Use full station names for all checkboxes
            self.station_checkboxes[station] = ctk.CTkCheckBox(
                checkbox_frame,
                text=f"Station {station}",
                variable=self.station_vars[station],
                onvalue=True,
                offvalue=False,
                fg_color=self.button_color,
                hover_color=self.button_color
            )

            # Get position from the map and place checkbox
            if station in station_positions:
                row, col = station_positions[station]
                self.station_checkboxes[station].grid(row=row, column=col, sticky="w", padx=5, pady=2)

        # Add parameter checkboxes with EXACTLY the same structure as station checkboxes
        param_checkbox_frame = ctk.CTkFrame(self.param_filter_frame, fg_color="transparent")
        param_checkbox_frame.grid(row=1, column=0, sticky="w", padx=5)

        # Create first row checkbox (LLDA)
        self.llda_var = ctk.BooleanVar(value=True)
        self.llda_var.trace_add("write", self.on_param_checkbox_change)

        self.llda_checkbox = ctk.CTkCheckBox(
            param_checkbox_frame,
            text="LLDA Parameters",
            variable=self.llda_var,
            onvalue=True,
            offvalue=False,
            fg_color=self.button_color,
            hover_color=self.button_color,
            command=self.on_llda_checkbox_clicked
        )
        self.llda_checkbox.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        # Create second row checkbox (PAGASA)
        self.pagasa_var = ctk.BooleanVar(value=False)
        self.pagasa_var.trace_add("write", self.on_param_checkbox_change)

        self.pagasa_checkbox = ctk.CTkCheckBox(
            param_checkbox_frame,
            text="PAGASA Parameters",
            variable=self.pagasa_var,
            onvalue=True,
            offvalue=False,
            fg_color=self.button_color,
            hover_color=self.button_color,
            command=self.on_pagasa_checkbox_clicked
        )
        self.pagasa_checkbox.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        # Add station filter controls in a fixed position
        station_filter_controls = ctk.CTkFrame(self.station_filter_frame, fg_color="transparent")
        station_filter_controls.grid(row=2, column=0, sticky="w", padx=5, pady=(15, 5))

        # Buttons for filtering with blue color
        select_all_btn = ctk.CTkButton(station_filter_controls, text="Select All", command=self.select_all_stations,
                                       fg_color=self.button_color, hover_color="#18558a")
        select_all_btn.grid(row=0, column=0, padx=5)

        deselect_all_btn = ctk.CTkButton(station_filter_controls, text="Deselect All",
                                         command=self.deselect_all_stations,
                                         fg_color=self.button_color, hover_color="#18558a")
        deselect_all_btn.grid(row=0, column=1, padx=5)

        # Create main data frame that will contain all station entries
        self.main_data_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.main_data_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)

        # Configure the main_data_frame to expand properly
        self.main_data_frame.columnconfigure(0, weight=0)  # Station labels column

        # Make parameter columns expandable to fill available space evenly
        for i in range(1, len(self.headers) + 1):
            self.main_data_frame.columnconfigure(i, weight=1)

        # Make rows expandable as well
        for i in range(len(self.stations) + 1):
            self.main_data_frame.rowconfigure(i, weight=0)

        # Labeling for headers with larger font and better spacing
        station_lbl = ctk.CTkLabel(self.main_data_frame, text="STATIONS", font=("Segoe UI", 12, "bold"))
        station_lbl.grid(column=0, row=0, padx=10, pady=10, sticky="w")

        # Store header widgets for visibility management
        self.header_widgets = {}

        # Header labels with wraplength for better display
        for col, header in enumerate(self.headers, start=1):
            # Add valid range to header label
            min_val, max_val = self.valid_ranges[header]
            header_text = f"{header}\nRange: {min_val}-{max_val}"

            header_lbl = ctk.CTkLabel(
                self.main_data_frame,
                text=header_text,
                font=("Segoe UI", 10),
                wraplength=100,  # Allow text to wrap
                justify="center"
            )
            header_lbl.grid(column=col, row=0, padx=10, pady=10, sticky="ew")

            # Store the column position of this header
            self.header_columns[header] = col

            # Store the header widget for easier access
            self.header_widgets[header] = header_lbl

            # Initially hide PAGASA headers since they're not selected by default
            if header in self.pagasa_headers:
                header_lbl.grid_remove()

        # Create entry widgets for each station and parameter with consistent width
        for row, station in enumerate(self.stations, start=1):
            # Create label for station
            station_label = ctk.CTkLabel(
                self.main_data_frame,
                text=f"Station {station}:",
                anchor="w"
            )
            station_label.grid(column=0, row=row, padx=10, pady=5, sticky="w")

            # Store station frame reference
            self.station_frames[station] = station_label
            self.entries[station] = {}

            # Create entry widgets for each parameter - all with same fixed width
            for col, header in enumerate(self.headers, start=1):
                # Make entry widgets wider to better fill the space
                entry = ctk.CTkEntry(self.main_data_frame, width=80, border_color=self.button_color)
                entry.grid(column=col, row=row, padx=10, pady=5, sticky="ew")

                # If this is the first entry, store its default border color
                if self.default_border_color is None:
                    self.default_border_color = self.button_color  # Use blue as default border color

                # No placeholder text
                self.entries[station][header] = entry

                # Set up focus event to clear error when clicked
                entry.bind("<FocusIn>", lambda e, entry=entry: self.reset_entry_format(e, entry))

                # Initially hide entries for PAGASA headers since they're not selected by default
                if header in self.pagasa_headers:
                    entry.grid_remove()

        # Calculate the row position for the controls
        control_row = len(self.stations) + 1

        # Create a frame for controls that spans across the entire width
        control_frame = ctk.CTkFrame(self.main_data_frame, fg_color="transparent")
        control_frame.grid(column=0, row=control_row, columnspan=len(self.headers) + 1, padx=10, pady=5, sticky="ew")

        # Configure the control frame to allow centering
        control_frame.columnconfigure(0, weight=1)  # Left spacer
        control_frame.columnconfigure(1, weight=0)  # Content
        control_frame.columnconfigure(2, weight=1)  # Right spacer

        # Create an inner frame to hold the actual controls
        inner_control_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        inner_control_frame.grid(column=1, row=0, padx=10, pady=5)

        # Month selection
        month_label = ctk.CTkLabel(inner_control_frame, text="Select Month:")
        month_label.grid(column=0, row=0, padx=5, pady=5, sticky="e")

        # Month dropdown
        month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        self.month_var = ctk.StringVar(value=month_names[datetime.now().month - 1])
        self.month_dropdown = ctk.CTkOptionMenu(
            inner_control_frame,
            values=month_names,
            variable=self.month_var,
            width=100,
            fg_color=self.button_color,
            button_color=self.button_color,
            button_hover_color="#18558a"
        )
        self.month_dropdown.grid(column=1, row=0, padx=5, pady=5, sticky="w")

        # Year selection
        year_label = ctk.CTkLabel(inner_control_frame, text="Select Year:")
        year_label.grid(column=2, row=0, padx=5, pady=5, sticky="e")

        # Year dropdown
        current_year = datetime.now().year
        year_range = [str(year) for year in range(current_year - 10, current_year + 11)]
        self.year_var = ctk.StringVar(value=str(current_year))
        self.year_dropdown = ctk.CTkOptionMenu(
            inner_control_frame,
            values=year_range,
            variable=self.year_var,
            width=70,
            fg_color=self.button_color,
            button_color=self.button_color,
            button_hover_color="#18558a"
        )
        self.year_dropdown.grid(column=3, row=0, padx=5, pady=5, sticky="w")

        # Submit button
        self.submit_button = ctk.CTkButton(
            inner_control_frame,
            text="Submit",
            command=self.submit_data,
            width=80,
            fg_color=self.button_color,
            hover_color="#18558a"
        )
        self.submit_button.grid(column=4, row=0, padx=5, pady=5, sticky="w")

        # Clear All button
        self.clear_button = ctk.CTkButton(
            inner_control_frame,
            text="Clear All",
            command=self.clear_all,
            width=80,
            fg_color=self.button_color,
            hover_color="#18558a"
        )
        self.clear_button.grid(column=5, row=0, padx=5, pady=5, sticky="w")

        # ========== ML MODEL SECTION ==========
        # Create a frame for model-related widgets DIRECTLY AFTER the main data frame
        self.model_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.model_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=20)
        self.model_frame.columnconfigure(0, weight=1)  # Make frame expandable

        # Create a label for the model section
        model_label = ctk.CTkLabel(self.model_frame, text="Machine Learning Model", font=("Segoe UI", 14, "bold"))
        model_label.grid(row=0, column=0, padx=10, pady=(0, 5), sticky="w")

        # Create a button to run the model
        self.run_model_button = ctk.CTkButton(
            self.model_frame,
            text="Run Model",
            command=self.run_model,
            width=200,
            height=40,
            fg_color=self.button_color,
            hover_color="#18558a",
            font=("Segoe UI", 12, "bold")
        )
        self.run_model_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Create a text box to display model output
        self.model_output = ctk.CTkTextbox(
            self.model_frame,
            width=800,
            height=300,
            corner_radius=5,
            border_width=2,
            border_color=self.button_color,
            font=("Courier New", 11)
        )
        self.model_output.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        self.model_output.insert("1.0", "Model output will appear here after running the analysis...")
        self.model_output.configure(state="disabled")  # Make read-only initially

        # Apply parameter filter initially
        self.apply_param_filter()

        # Update canvas scroll region when widgets change size
        self.content_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Bind mousewheel for scrolling
        self.bind_mousewheel()

    def on_frame_configure(self, event):
        """Update the scroll region based on the content frame size"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Adjust the width of the canvas window when canvas is resized"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def bind_mousewheel(self):
        """Bind mousewheel to scroll the canvas"""

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind for Windows and macOS
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def run_model(self):
        """Run the model functions from model_trainer.py in a separate thread to keep UI responsive"""
        # Log the model run
        if self.audit_logger:
            self.audit_logger.log_system_event(
                username=self.current_username,
                user_type=self.user_type,
                event_type="model_run",
                details="User ran water quality analysis model"
            )

        # Clear the output textbox and make it editable
        self.model_output.configure(state="normal")
        self.model_output.delete("1.0", "end")

        # Add initial header
        self.model_output.insert("end", "=== Water Quality Model Analysis ===\n\n")
        self.model_output.update()  # Update the UI immediately

        # Disable the run button while model is running
        self.run_model_button.configure(state="disabled", text="Algorithm Running...")

        # Create a queue for thread-safe communication
        output_queue = queue.Queue()

        # Function to update UI from the main thread
        def update_ui():
            try:
                # Get all available messages from the queue
                while not output_queue.empty():
                    message = output_queue.get_nowait()
                    self.model_output.insert("end", message)
                    self.model_output.see("end")

                # Continue updating if thread is still alive
                if thread.is_alive():
                    self.after(100, update_ui)  # Check again in 100ms
                else:
                    # Thread is done, re-enable button
                    self.run_model_button.configure(state="normal", text="Run Water Quality Analysis")
                    self.model_output.configure(state="disabled")
            except Exception as e:
                print(f"Error in UI update: {e}")

        # Function to run in the background thread
        def run_model_thread():
            try:
                # Create a custom stdout to capture and process output
                class CustomStdout:
                    def __init__(self, queue_obj):
                        self.queue = queue_obj
                        self.feature_count = 0
                        self.in_features_section = False

                    def write(self, text):
                        # Process each line of text
                        for line in text.splitlines(True):  # Keep line endings
                            # Process based on content
                            if "=== Loading and preprocessing data ===" in line:
                                self.queue.put("Loading and preprocessing data...\n")
                            elif "Processing station information..." in line:
                                self.queue.put("Processing station information...\n")
                            elif "=== Handling missing values" in line:
                                self.queue.put("Handling missing values...\n")
                            elif "Skewness of Chlorophyll-a\n" in line:
                                self.queue.put(line)
                            elif "Applying log transformation" in line:
                                self.queue.put("Applying log transformation to target\n")
                            elif "=== Creating enhanced features" in line:
                                self.queue.put("Creating time-series enhanced features\n")
                            elif "Setting up time series cross-validation" in line:
                                self.queue.put("Setting up time series validation\n\n")
                            elif "Tuning Gradient Boosting hyperparameters" in line:
                                if self.in_features_section:
                                    self.in_features_section = False
                                    self.queue.put("\n")
                                self.queue.put("Tuning and training the gradient boosting hyperparameters...\n")
                            elif "Training Gradient Boosting with best parameters" in line:
                                self.queue.put("Model has found the best parameters and successfully trained the "
                                               "dataset!\n\n")
                            elif "SUMMARY OF RESULTS" in line:
                                self.queue.put("\nModel analysis completed successfully!")
                            elif "Model analysis completed successfully" in line:
                                self.queue.put("\nYou may now use the forecasting!")

                    def flush(self):
                        pass

                # Redirect stdout to our custom handler
                original_stdout = sys.stdout
                sys.stdout = CustomStdout(output_queue)

                try:
                    # Run the model
                    model_trainer.main()
                finally:
                    # Restore original stdout
                    sys.stdout = original_stdout

            except Exception as e:
                # If an error occurs, put the error in the queue
                error_message = f"Error running model: {str(e)}\n\n"
                import traceback
                error_message += traceback.format_exc()
                output_queue.put(error_message)

                # Log the error
                if self.audit_logger:
                    self.audit_logger.log_system_event(
                        username=self.current_username,
                        user_type=self.user_type,
                        event_type="model_error",
                        details=f"Error in model run: {str(e)}"
                    )

        # Create and start the thread
        thread = threading.Thread(target=run_model_thread, daemon=True)
        thread.start()

        # Start the UI update function
        update_ui()

    def reset_entry_format(self, event, entry):
        """ Separate method to reset entry format with direct reference to the entry """
        # Check if the entry contains error text
        current_text = entry.get()
        if current_text in ["Invalid Input", "ⓘ", "Out of Range"]:
            entry.delete(0, "end")  # Clear the content

        # Reset formatting using the button color as the default border color
        entry.configure(border_color=self.button_color, text_color=("gray10", "#DCE4EE"))

    def clear_error(self, event):
        """ Original method kept for backward compatibility """
        widget = event.widget

        # Check if the entry contains error text
        current_text = widget.get()
        if current_text in ["Invalid Input", "ⓘ", "Out of Range"]:
            widget.delete(0, "end")  # Clear the content

        # Reset the entry formatting to default
        widget.configure(border_color=self.button_color, text_color=self.default_text_color)

    # Called when station checkbox state changes
    def on_checkbox_change(self, *args):
        self.apply_filter()

    # Called when parameter checkbox state changes
    def on_param_checkbox_change(self, *args):
        self.apply_param_filter()
        print(f"Parameter filter changed - LLDA: {self.llda_var.get()}, PAGASA: {self.pagasa_var.get()}")
        print(f"Visible headers: {self.visible_headers}")

    # Station filter-related methods
    def select_all_stations(self):
        for station in self.stations:
            self.station_vars[station].set(True)

    def deselect_all_stations(self):
        for station in self.stations:
            self.station_vars[station].set(False)

    def apply_filter(self):
        """Apply station filtering while respecting parameter filtering"""
        print("Applying station filter...")
        for station in self.stations:
            if self.station_vars[station].get():
                # Get the corresponding label and make it visible
                station_label = self.station_frames[station]
                station_label.grid()

                # Only show entries for currently visible parameters
                for header in self.headers:
                    entry = self.entries[station][header]
                    if header in self.visible_headers:
                        entry.grid()
                    else:
                        entry.grid_remove()
            else:
                # Hide this station's label and all associated entry widgets
                station_label = self.station_frames[station]
                station_label.grid_remove()

                # Hide all entry widgets for this station
                for header, entry in self.entries[station].items():
                    entry.grid_remove()

    # Parameter filter-related methods
    def apply_param_filter(self):
        """Apply parameter filtering while maintaining station filtering"""
        print("Applying parameter filter...")
        # Determine which parameters to show based on checkbox states
        self.visible_headers = []

        if self.llda_var.get():
            self.visible_headers.extend(self.llda_headers)
            print("LLDA parameters visible")

        if self.pagasa_var.get():
            self.visible_headers.extend(self.pagasa_headers)
            print("PAGASA parameters visible")

        print(f"Total visible headers: {len(self.visible_headers)}")

        # Special case: When only PAGASA is selected, move PAGASA headers to appear next to station header
        if self.pagasa_var.get() and not self.llda_var.get():
            # Temporarily remove all header widgets
            for header in self.headers:
                header_widget = self.header_widgets[header]
                header_widget.grid_remove()

            # Show only PAGASA headers directly after station column
            for col, header in enumerate(self.pagasa_headers, start=1):
                header_widget = self.header_widgets[header]
                # Reconfigure to show in correct order
                header_widget.grid(column=col, row=0, padx=10, pady=10, sticky="ew")
        else:
            # Normal case: restore original column positions and visibility
            for header in self.headers:
                header_widget = self.header_widgets[header]
                col = self.header_columns[header]  # Get original column position

                # Configure to original position
                header_widget.grid(column=col, row=0, padx=10, pady=10, sticky="ew")

                # Then hide if not visible
                if header not in self.visible_headers:
                    header_widget.grid_remove()

        # Show or hide entry widgets based on both parameter and station visibility
        for station in self.stations:
            # Only update visible stations
            if self.station_vars[station].get():
                if self.pagasa_var.get() and not self.llda_var.get():
                    # When only PAGASA selected, reposition entry widgets to match header positions
                    for idx, header in enumerate(self.pagasa_headers):
                        entry = self.entries[station][header]
                        entry.grid(column=idx + 1, row=int(self.station_frames[station].grid_info()['row']), padx=10,
                                   pady=5, sticky="ew")
                else:
                    # Normal case: use original column positions
                    for header, entry in self.entries[station].items():
                        if header in self.visible_headers:
                            entry.grid(column=self.header_columns[header],
                                       row=int(self.station_frames[station].grid_info()['row']), padx=10, pady=5,
                                       sticky="ew")
                        else:
                            entry.grid_remove()

    # Method to validate a single parameter value against its valid range
    def validate_parameter(self, param, value):
        """
        Validate a parameter value against its defined valid range.
        Returns (is_valid, error_message)
        """
        if value == "":
            return False, "Empty value"

        try:
            float_value = float(value)
            min_val, max_val = self.valid_ranges[param]

            if min_val <= float_value <= max_val:
                return True, ""
            else:
                return False, f"Value must be between {min_val} and {max_val}"

        except ValueError:
            return False, "Not a valid number"

    def submit_data(self, skip_validation=False):
        # Store invalid entries to keep their highlighting
        invalid_entries = []

        if not skip_validation:
            is_valid = True

            # Get the selected month and year
            month_str = self.month_var.get()
            year_str = self.year_var.get()

            # Convert month name to month number (1-12)
            month_names = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            month_num = month_names.index(month_str) + 1

            # Format as YYYY-MM-DD for Excel compatibility
            # Use the first day of the month
            date_str = f"{year_str}-{month_num:02d}-01 00:00:00"

            for station, headers in self.entries.items():
                # Only validate visible stations
                if not self.station_vars[station].get():
                    continue

                for param, entry in headers.items():
                    # Only validate visible parameters
                    if param not in self.visible_headers:
                        continue

                    value = entry.get().strip()

                    # Skip entries that already have error messages
                    if value in ["Invalid Input", "ⓘ", "Out of Range"]:
                        value = ""

                    # Check if value is empty first
                    if not value:
                        entry.delete(0, "end")
                        entry.insert(0, "ⓘ")
                        entry.configure(border_color="red", text_color="red")
                        invalid_entries.append(entry)
                        is_valid = False
                        continue

                    # Validate against parameter ranges
                    valid, error_msg = self.validate_parameter(param, value)
                    if not valid:
                        entry.delete(0, "end")
                        entry.insert(0, "Out of Range")
                        entry.configure(border_color="red", text_color="red")
                        invalid_entries.append(entry)
                        is_valid = False
                    else:
                        entry.configure(border_color=self.button_color, text_color=self.default_text_color)

            if not is_valid:
                # Show error popup without showing the valid ranges popup
                self.validPopUp(invalid_entries)
                return

        # Define the output file path for the Excel file
        excel_file = self.excel_file

        # Create new data to add to the Excel file
        new_data = []

        # Get formatted date for all entries
        month_str = self.month_var.get()
        year_str = self.year_var.get()
        month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        month_num = month_names.index(month_str) + 1
        date_str = f"{year_str}-{month_num:02d}-01 00:00:00"  # Format similar to existing data

        # Track submitted stations for audit logging
        submitted_stations = []
        total_data_points = 0

        # Collect data from all visible stations
        for station_code, headers in self.entries.items():
            # Only process visible stations
            if not self.station_vars[station_code].get():
                continue

            # Get the full station name
            station_name = self.station_names.get(station_code)
            if not station_name:
                print(f"Warning: No station name mapping for Station {station_code}")
                continue

            # Create a row for each station's data
            row_data = {
                "Date": date_str,
                "Station": station_name
            }

            # Add parameter values for all parameters, regardless of visibility
            for param in self.headers:
                entry = headers[param]

                # Get value if parameter is visible, otherwise set to None
                if param in self.visible_headers:
                    value = entry.get().strip()
                    # Skip error messages
                    if value in ["Invalid Input", "ⓘ", "Out of Range"]:
                        value = None
                    elif value:  # Only convert non-empty values
                        try:
                            # Convert to float if possible
                            value = float(value)
                        except ValueError:
                            value = None
                    else:
                        value = None
                else:
                    value = None

                row_data[param] = value

            # Add default value for Occurrences column that might be in the dataset
            row_data["Occurrences"] = 0

            new_data.append(row_data)
            submitted_stations.append(station_code)
            total_data_points += len([v for v in row_data.values() if v is not None])

        try:
            print(f"Looking for Excel file at: {excel_file}")
            # Check if the Excel file exists
            if os.path.exists(excel_file):
                print(f"Excel file found. Loading existing data...")
                # Load existing data
                existing_df = pd.read_excel(excel_file)
                print(f"Loaded {len(existing_df)} existing rows")

                # Convert new data to DataFrame
                new_df = pd.DataFrame(new_data)
                print(f"Adding {len(new_df)} new rows")

                # Combine existing and new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                # Save back to Excel
                combined_df.to_excel(excel_file, index=False)
                print(f"Saved combined data with {len(combined_df)} total rows")
            else:
                print(f"Excel file not found. Creating new file...")
                # Create folder if it doesn't exist (double-check)
                if not os.path.exists(self.csv_folder):
                    os.makedirs(self.csv_folder)
                    print(f"Created folder: {self.csv_folder}")

                # Create new Excel file with the data
                new_df = pd.DataFrame(new_data)
                new_df.to_excel(excel_file, index=False)
                print(f"Created new Excel file with {len(new_df)} rows")

            # Log the data submission to the audit log
            if self.audit_logger:
                # Create details string with information about what was submitted
                details = f"Submitted data for {len(submitted_stations)} stations ({', '.join(submitted_stations)}), "
                details += f"Month: {month_str}, Year: {year_str}, "
                details += f"Total data points: {total_data_points}"

                # Log the system event
                self.audit_logger.log_system_event(
                    username=self.current_username,
                    user_type=self.user_type,
                    event_type="data_input",
                    details=details
                )
                print(f"Audit log entry created for data submission by {self.current_username}")

            self.show_success_message()

        except Exception as e:
            print(f"Error saving data: {e}")
            import traceback
            traceback.print_exc()

            # Log the error to the audit log
            if self.audit_logger:
                error_details = f"Error during data submission: {str(e)}"
                self.audit_logger.log_system_event(
                    username=self.current_username,
                    user_type=self.user_type,
                    event_type="data_input_error",
                    details=error_details
                )
                print(f"Audit log entry created for error by {self.current_username}")

            self.show_error_message(str(e))

    def center_popup(self, popup, width, height):
        """
        Center a popup window on the screen or relative to the parent window
        
        Args:
            popup: The CTkToplevel window to center
            width: Width of the popup
            height: Height of the popup
        """
        # Get screen dimensions
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        
        # Calculate position to center the popup
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set the geometry with calculated position
        popup.geometry(f"{width}x{height}+{x}+{y}")
        
        # Alternative: Center relative to parent window
        # Uncomment the code below if you prefer centering relative to the main window
        """
        # Get parent window position and size
        parent_x = self.winfo_rootx()
        parent_y = self.winfo_rooty()
        parent_width = self.winfo_width()
        parent_height = self.winfo_height()
        
        # Calculate position to center relative to parent
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2
        
        # Ensure popup doesn't go off screen
        x = max(0, min(x, screen_width - width))
        y = max(0, min(y, screen_height - height))
        
        popup.geometry(f"{width}x{height}+{x}+{y}")
        """
    
    def validPopUp(self, invalid_entries=None):
        # Define popup dimensions
        popup_width = 350
        popup_height = 170
        
        popup = ctk.CTkToplevel(self)
        popup.title("Invalid Data Detected")
        popup.grab_set()
        
        # Center the popup using our helper method
        self.center_popup(popup, popup_width, popup_height)

        msg = ctk.CTkLabel(popup, text="Some fields have invalid or missing data.\nDo you want to continue anyway?",
                        font=("Segoe UI", 11))
        msg.pack(pady=20)

        hint = ctk.CTkLabel(popup, text="Please check the column headers for\nacceptable parameter ranges.",
                            font=("Segoe UI", 9), text_color="#1E90FF")
        hint.pack(pady=0)

        button_frame = ctk.CTkFrame(popup, fg_color="transparent")
        button_frame.pack(pady=20)

        # Store invalid entries for reference when popup is closed
        popup.invalid_entries = invalid_entries

        def continue_anyway():
            # Log the validation override in the audit log
            if self.audit_logger:
                invalid_count = len(invalid_entries) if invalid_entries else 0
                details = f"User bypassed validation for {invalid_count} invalid fields"
                self.audit_logger.log_system_event(
                    username=self.current_username,
                    user_type=self.user_type,
                    event_type="validation_override",
                    details=details
                )
                print(f"Audit log entry created for validation override by {self.current_username}")

            popup.destroy()
            self.submit_data(skip_validation=True)

        def go_back():
            # Make sure highlighting remains when popup is closed
            if popup.invalid_entries:
                for entry in popup.invalid_entries:
                    if entry.winfo_exists():  # Check if widget still exists
                        entry.configure(border_color="red", text_color="red")
            popup.destroy()

        continueBtn = ctk.CTkButton(button_frame, text="Continue Anyway", width=120, command=continue_anyway,
                                    fg_color=self.button_color, hover_color="#18558a")
        continueBtn.grid(row=0, column=0, padx=10)

        cancelBtn = ctk.CTkButton(button_frame, text="Go Back", width=120, command=go_back,
                                fg_color=self.button_color, hover_color="#18558a")
        cancelBtn.grid(row=0, column=1, padx=10)

        popup.protocol("WM_DELETE_WINDOW", go_back)

    def show_success_message(self):
        """Show a success message when data is saved successfully"""
        popup_width = 300
        popup_height = 150
        
        popup = ctk.CTkToplevel(self)
        popup.title("Success")
        popup.grab_set()
        
        # Center the popup
        self.center_popup(popup, popup_width, popup_height)

        msg = ctk.CTkLabel(popup, text="Data saved successfully!", font=("Segoe UI", 12))
        msg.pack(pady=30)

        ok_btn = ctk.CTkButton(popup, text="OK", width=100, command=popup.destroy,
                            fg_color=self.button_color, hover_color="#18558a")
        ok_btn.pack(pady=10)

    # Method to clear all existing data in textbox
    def clear_all(self):
        # Log the clear action
        if self.audit_logger:
            self.audit_logger.log_system_event(
                username=self.current_username,
                user_type=self.user_type,
                event_type="clear_data_form",
                details="User cleared all input fields"
            )
            print(f"Audit log entry created for clear action by {self.current_username}")

        for station, headers in self.entries.items():
            # Only clear visible stations
            if self.station_vars[station].get():
                for param, entry_widget in headers.items():
                    # Only clear visible parameters
                    if param in self.visible_headers:
                        entry_widget.delete(0, "end")
                        # Explicitly reset the border color to default
                        entry_widget.configure(border_color=self.button_color, text_color=self.default_text_color)

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")

        # Log page access
        if self.audit_logger:
            self.audit_logger.log_system_event(
                username=self.current_username,
                user_type=self.user_type,
                event_type="page_access",
                details="Accessed Input Data page"
            )
            print(f"Audit log entry created for page access by {self.current_username}")

        self.apply_param_filter()
        self.apply_filter()

    def show_error_message(self, error_message):
        """Show an error message when data saving fails"""
        popup_width = 400
        popup_height = 200
        
        popup = ctk.CTkToplevel(self)
        popup.title("Error")
        popup.grab_set()
        
        # Center the popup
        self.center_popup(popup, popup_width, popup_height)

        msg = ctk.CTkLabel(popup, text=f"Error saving data:\n{error_message}",
                        font=("Segoe UI", 11), wraplength=350)
        msg.pack(pady=30)

        ok_btn = ctk.CTkButton(popup, text="OK", width=100, command=popup.destroy,
                            fg_color=self.button_color, hover_color="#18558a")
        ok_btn.pack(pady=10)