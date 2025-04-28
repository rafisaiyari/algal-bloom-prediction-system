import customtkinter as ctk
import pandas as pd
import os
from datetime import datetime


class InputDataPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        self.headers = ["pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)",
                        "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)", "Phytoplankton"]
        self.stations = ["I", "II", "IV", "V", "VIII", "XV", "XVI", "XVII", "XVIII"]
        self.entries = {}
        self.station_frames = {}  # To store frames for each station row
        self.station_checkboxes = {}  # To store checkboxes for filtering
        self.station_vars = {}  # To store checkbox variables

        # Map station codes to their full names
        self.station_names = {
            "I": "Station_1_CWB",
            "II": "Station_2_EastB",
            "IV": "Station_4_CentralB",
            "V": "Station_5_NorthernWestBay",
            "VIII": "Station_8_SouthB",
            "XV": "Station_15_SanPedro",
            "XVI": "Station_16_Sta. Rosa",
            "XVII": "Station_17_Sanctuary",
            "XVIII": "Station_18_Pagsanjan",
        }

        # Define valid ranges for each parameter
        self.valid_ranges = {
            "pH (units)": (0, 14),  # pH scale 0-14
            "Ammonia (mg/L)": (0, 10),  # Ammonia typical range in water bodies
            "Nitrate (mg/L)": (0, 100),  # Nitrate typical range
            "Inorganic Phosphate (mg/L)": (0, 10),  # Phosphate typical range
            "Dissolved Oxygen (mg/L)": (0, 20),  # DO typical range
            "Temperature": (0, 40),  # Water temperature in Celsius
            "Chlorophyll-a (ug/L)": (0, 300),  # Chlorophyll-a typical range
            "Phytoplankton": (0, 1000000)  # Phytoplankton count (cells/mL)
        }

        # Store the default border color to use when resetting fields
        self.default_border_color = None
        self.default_text_color = ("gray10", "#DCE4EE")

        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.initialize_file_settings()
        self.create_widgets()

    # Initialize basic variables
    def initialize_file_settings(self):
        # Set path to the existing Excel file in the CSV folder
        self.csv_folder = "CSV"
        self.excel_file = os.path.join(self.csv_folder, "merged_stationsCopy.xlsx")
        # Create folder if it doesn't exist
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
            print(f"Created folder: {self.csv_folder}")
        print(f"Excel file path: {self.excel_file}")

    def create_widgets(self):
        # Configure the main frame to expand with window
        self.columnconfigure(0, weight=1)

        inputDatalb = ctk.CTkLabel(self, text="INPUT DATA", font=("Arial", 25, "bold"))
        inputDatalb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        # Create filter frame
        self.filter_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.filter_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        self.filter_frame.columnconfigure(0, weight=1)  # Make filter frame expandable

        # Add filter label
        ctk.CTkLabel(self.filter_frame, text="Filter Stations:", font=("Arial", 12)).grid(row=0, column=0,
                                                                                          sticky="w", padx=5,
                                                                                          pady=5)

        # Add checkboxes for each station with specific layout
        checkbox_frame = ctk.CTkFrame(self.filter_frame, fg_color="transparent")
        checkbox_frame.grid(row=1, column=0, sticky="w", padx=5)
        
        # Map station code to row and column position
        # First row: "I", "IV", "VIII", "XVI", "XVIII"
        # Second row: "II", "V", "XV", "XVII"
        station_positions = {
            "I": (0, 0),      # Station 1 - first row, first column
            "II": (1, 0),     # Station 2 - second row, first column
            "IV": (0, 1),     # Station 4 - first row, second column
            "V": (1, 1),      # Station 5 - second row, second column
            "VIII": (0, 2),   # Station 8 - first row, third column
            "XV": (1, 2),     # Station 15 - second row, third column
            "XVI": (0, 3),    # Station 16 - first row, fourth column
            "XVII": (1, 3),   # Station 17 - second row, fourth column
            "XVIII": (0, 4)   # Station 18 - first row, fifth column
        }
        
        for station in self.stations:
            self.station_vars[station] = ctk.BooleanVar(value=True)
            # Set up trace to automatically apply filter when checkbox state changes
            self.station_vars[station].trace_add("write", self.on_checkbox_change)

            self.station_checkboxes[station] = ctk.CTkCheckBox(
                checkbox_frame,
                text=f"Station {station}",
                variable=self.station_vars[station],
                onvalue=True,
                offvalue=False
            )
            
            # Place according to the predefined positions
            if station in station_positions:
                row, col = station_positions[station]
                self.station_checkboxes[station].grid(row=row, column=col, sticky="w", padx=10, pady=2)

        # Add filter controls - only Select All and Deselect All buttons
        filter_controls = ctk.CTkFrame(self.filter_frame, fg_color="transparent")
        filter_controls.grid(row=2, column=0, sticky="w", padx=5, pady=10)

        # Buttons for filtering
        select_all_btn = ctk.CTkButton(filter_controls, text="Select All", command=self.select_all_stations)
        select_all_btn.grid(row=0, column=0, padx=5)

        deselect_all_btn = ctk.CTkButton(filter_controls, text="Deselect All", command=self.deselect_all_stations)
        deselect_all_btn.grid(row=0, column=1, padx=5)

        # Create main data frame that will contain all station entries
        self.main_data_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_data_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)

        # Make all columns expandable
        for i in range(len(self.headers) + 1):
            self.main_data_frame.columnconfigure(i, weight=1)

        # Labeling for headers with larger font and better spacing
        station_lbl = ctk.CTkLabel(self.main_data_frame, text="STATIONS", font=("Arial", 12, "bold"))
        station_lbl.grid(column=0, row=0, padx=10, pady=10, sticky="w")

        # Header labels with wraplength for better display
        for col, header in enumerate(self.headers, start=1):
            # Add valid range to header label
            min_val, max_val = self.valid_ranges[header]
            header_text = f"{header}\nRange: {min_val}-{max_val}"
            
            header_lbl = ctk.CTkLabel(
                self.main_data_frame,
                text=header_text,
                font=("Arial", 10),
                wraplength=100,  # Allow text to wrap
                justify="center"
            )
            header_lbl.grid(column=col, row=0, padx=10, pady=10)

        # Create entry widgets for each station and parameter
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

            # Create entry widgets for each parameter
            for col, header in enumerate(self.headers, start=1):
                entry = ctk.CTkEntry(self.main_data_frame, width=120)  # Fixed width entry
                entry.grid(column=col, row=row, padx=10, pady=5, sticky="ew")
                
                # If this is the first entry, store its default border color
                if self.default_border_color is None:
                    self.default_border_color = entry.cget("border_color")
                
                # No placeholder text
                self.entries[station][header] = entry

                # Set up focus event to clear error when clicked
                entry.bind("<FocusIn>", lambda e, entry=entry: self.reset_entry_format(e, entry))

        # Creates a new frame for controls
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.grid(row=3, column=0, pady=20, sticky="ew")

        # Center the control frame content
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(4, weight=1)

        # Month and Year selection in centered inner frame
        control_inner_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        control_inner_frame.grid(row=0, column=1, columnspan=3)

        # Month selection
        ctk.CTkLabel(control_inner_frame, text="Select Month:").grid(column=0, row=0, padx=5, pady=5)
        
        # Create month dropdown with optimized width
        month_names = ["January", "February", "March", "April", "May", "June", 
                      "July", "August", "September", "October", "November", "December"]
        self.month_var = ctk.StringVar(value=month_names[datetime.now().month-1])
        self.month_dropdown = ctk.CTkOptionMenu(
            control_inner_frame,
            values=month_names,
            variable=self.month_var,
            width=100  # Set appropriate width for month dropdown
        )
        self.month_dropdown.grid(column=1, row=0, padx=10, pady=5)
        
        # Year selection
        ctk.CTkLabel(control_inner_frame, text="Select Year:").grid(column=2, row=0, padx=5, pady=5)
        
        # Create year dropdown with optimized width
        current_year = datetime.now().year
        year_range = [str(year) for year in range(current_year-10, current_year+11)]
        self.year_var = ctk.StringVar(value=str(current_year))
        self.year_dropdown = ctk.CTkOptionMenu(
            control_inner_frame,
            values=year_range,
            variable=self.year_var,
            width=70  # Set appropriate width for year dropdown
        )
        self.year_dropdown.grid(column=3, row=0, padx=10, pady=5)

        # Buttons
        self.submit_button = ctk.CTkButton(control_inner_frame, text="Submit", command=self.submit_data)
        self.submit_button.grid(column=4, row=0, padx=10, pady=5)

        self.clear_button = ctk.CTkButton(control_inner_frame, text="Clear All", command=self.clear_all)
        self.clear_button.grid(column=5, row=0, padx=10, pady=5)

    def reset_entry_format(self, event, entry):
        """ Separate method to reset entry format with direct reference to the entry """
        # Check if the entry contains error text
        current_text = entry.get()
        if current_text in ["Invalid Input", "ⓘ", "Out of Range"]:
            entry.delete(0, "end")  # Clear the content
        
        # Reset formatting using hardcoded defaults 
        # or use the stored defaults if they're properly captured
        entry.configure(border_color="grey", text_color=("gray10", "#DCE4EE"))

    def clear_error(self, event):
        """ Original method kept for backward compatibility """
        widget = event.widget
        
        # Check if the entry contains error text
        current_text = widget.get()
        if current_text in ["Invalid Input", "ⓘ", "Out of Range"]:
            widget.delete(0, "end")  # Clear the content
        
        # Reset the entry formatting to default
        widget.configure(border_color=self.default_border_color, text_color=self.default_text_color)

    # Called when checkbox state changes
    def on_checkbox_change(self, *args):
        self.apply_filter()

    # Filter-related methods
    def select_all_stations(self):
        for station in self.stations:
            self.station_vars[station].set(True)
        # No need to call apply_filter() as it will be triggered by the trace

    def deselect_all_stations(self):
        for station in self.stations:
            self.station_vars[station].set(False)
        # No need to call apply_filter() as it will be triggered by the trace

    def apply_filter(self):
        for station in self.stations:
            if self.station_vars[station].get():
                # Get the corresponding label and all associated entry widgets
                station_label = self.station_frames[station]
                station_label.grid()

                # Make all entry widgets for this station visible
                for header, entry in self.entries[station].items():
                    entry.grid()
            else:
                # Hide this station's label and all associated entry widgets
                station_label = self.station_frames[station]
                station_label.grid_remove()

                # Hide all entry widgets for this station
                for header, entry in self.entries[station].items():
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
                        entry.configure(border_color=self.default_border_color, text_color=self.default_text_color)

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
            
            # Add parameter values
            for param, entry in headers.items():
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
                
                row_data[param] = value
                
            # Add default values for Solar columns that might be in the dataset
            row_data["Solar Mean"] = None
            row_data["Solar Max"] = None
            row_data["Solar Min"] = None
            row_data["Occurences"] = 0
                
            new_data.append(row_data)
        
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
                
            # Show successful message
            self.show_success_message()
            
        except Exception as e:
            print(f"Error saving data: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message(str(e))

    def validPopUp(self, invalid_entries=None):
        popup = ctk.CTkToplevel(self)
        popup.title("Invalid Data Detected")
        popup.geometry("350x170")
        popup.grab_set()

        msg = ctk.CTkLabel(popup, text="Some fields have invalid or missing data.\nDo you want to continue anyway?",
                           font=("Arial", 11))
        msg.pack(pady=20)

        # Updated hint text to reference the header labels
        hint = ctk.CTkLabel(popup, text="Please check the column headers for\nacceptable parameter ranges.",
                            font=("Arial", 9), text_color="#1E90FF")
        hint.pack(pady=0)

        button_frame = ctk.CTkFrame(popup, fg_color="transparent")
        button_frame.pack(pady=20)

        # Store invalid entries for reference when popup is closed
        popup.invalid_entries = invalid_entries

        # Configure actions for buttons
        def continue_anyway():
            popup.destroy()
            self.submit_data(skip_validation=True)

        def go_back():
            # Make sure highlighting remains when popup is closed
            if popup.invalid_entries:
                for entry in popup.invalid_entries:
                    if entry.winfo_exists():  # Check if widget still exists
                        entry.configure(border_color="red", text_color="red")
            popup.destroy()

        continueBtn = ctk.CTkButton(button_frame, text="Continue Anyway", width=120, command=continue_anyway)
        continueBtn.grid(row=0, column=0, padx=10)

        cancelBtn = ctk.CTkButton(button_frame, text="Go Back", width=120, command=go_back)
        cancelBtn.grid(row=0, column=1, padx=10)

        # Make sure widget highlighting is preserved when popup is closed
        popup.protocol("WM_DELETE_WINDOW", go_back)
        
    def show_success_message(self):
        """Show a success message when data is saved successfully"""
        popup = ctk.CTkToplevel(self)
        popup.title("Success")
        popup.geometry("300x150")
        popup.grab_set()
        
        msg = ctk.CTkLabel(popup, text="Data saved successfully!", font=("Arial", 12))
        msg.pack(pady=30)
        
        ok_btn = ctk.CTkButton(popup, text="OK", width=100, command=popup.destroy)
        ok_btn.pack(pady=10)
        
    def show_error_message(self, error_message):
        """Show an error message when data saving fails"""
        popup = ctk.CTkToplevel(self)
        popup.title("Error")
        popup.geometry("400x200")
        popup.grab_set()
        
        msg = ctk.CTkLabel(popup, text=f"Error saving data:\n{error_message}", 
                           font=("Arial", 11), wraplength=350)
        msg.pack(pady=30)
        
        ok_btn = ctk.CTkButton(popup, text="OK", width=100, command=popup.destroy)
        ok_btn.pack(pady=10)

    # Method to clear all existing data in textbox
    def clear_all(self):
        for station, headers in self.entries.items():
            # Only clear visible stations
            if self.station_vars[station].get():
                for entry_widget in headers.values():
                    entry_widget.delete(0, "end")
                    # Explicitly reset the border color to default
                    entry_widget.configure(border_color=self.default_border_color, text_color=self.default_text_color)

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")
        # Apply filter initially to show all stations
        self.apply_filter()