import customtkinter as ctk
import csv
from pathlib import Path
from tkcalendar import DateEntry  # Note: tkcalendar may need to be adapted for CTk


class InputDataPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        self.headers = ["pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)",
                        "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)", "Phytoplankton"]
        self.stations = ["I", "II", "IV", "V", "VIII", "XV", "XVI", "XVII", "XVIII"]
        self.entries = {}
        self.station_frames = {}  # To store frames for each station row
        self.station_checkboxes = {}  # To store checkboxes for filtering
        self.station_vars = {}  # To store checkbox variables

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

        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.create_folders()
        self.create_widgets()

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

        # Add checkboxes for each station - now with automatic filtering
        checkbox_frame = ctk.CTkFrame(self.filter_frame, fg_color="transparent")
        checkbox_frame.grid(row=1, column=0, sticky="w", padx=5)

        for i, station in enumerate(self.stations):
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
            row, col = divmod(i, 3)  # Arrange checkboxes in rows of 3
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

        # Create tooltip frame for parameter validation ranges
        tooltip_frame = ctk.CTkFrame(self.main_data_frame, fg_color="transparent")
        tooltip_frame.grid(row=0, column=len(self.headers) + 1, padx=5, pady=5, sticky="e")

        tooltip_btn = ctk.CTkButton(tooltip_frame, text="ⓘ Valid Ranges", command=self.show_valid_ranges)
        tooltip_btn.grid(padx=5, pady=5)

        # Header labels with wraplength for better display
        for col, header in enumerate(self.headers, start=1):
            header_lbl = ctk.CTkLabel(
                self.main_data_frame,
                text=header,
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

            # Store station frame reference (though we're not using separate frames now)
            self.station_frames[station] = station_label
            self.entries[station] = {}

            # Create entry widgets for each parameter
            for col, header in enumerate(self.headers, start=1):
                entry = ctk.CTkEntry(self.main_data_frame, width=120)  # Fixed width entry
                entry.grid(column=col, row=row, padx=10, pady=5, sticky="ew")
                self.entries[station][header] = entry

                # Set up focus event to clear error when clicked
                entry.bind("<FocusIn>", self.clear_error)

        # Creates a new frame for controls
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.grid(row=3, column=0, pady=20, sticky="ew")

        # Center the control frame content
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(4, weight=1)

        # Date Entry and buttons in centered inner frame
        control_inner_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        control_inner_frame.grid(row=0, column=1, columnspan=3)

        ctk.CTkLabel(control_inner_frame, text="Select Year and Month:").grid(column=0, row=0, padx=5, pady=5)

        # For tkcalendar's DateEntry, we can use it directly with a custom style or wrapper
        # For this conversion, we'll place it in a frame to keep it working with CTk
        date_frame = ctk.CTkFrame(control_inner_frame)
        date_frame.grid(column=1, row=0, padx=5, pady=5)
        self.yearInput = DateEntry(date_frame, selectmode="day", width=10, date_pattern="y-mm-dd")
        self.yearInput.pack(padx=2, pady=2)

        # Buttons
        self.submit_button = ctk.CTkButton(control_inner_frame, text="Submit", command=self.submit_data)
        self.submit_button.grid(column=2, row=0, padx=10, pady=5)

        self.clear_button = ctk.CTkButton(control_inner_frame, text="Clear All", command=self.clear_all)
        self.clear_button.grid(column=3, row=0, padx=10, pady=5)

    def show_valid_ranges(self):
        # Create a popup to display valid parameter ranges
        popup = ctk.CTkToplevel(self)
        popup.title("Valid Parameter Ranges")
        popup.geometry("400x350")
        popup.grab_set()

        # Create main frame
        main_frame = ctk.CTkFrame(popup)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Add header
        ctk.CTkLabel(main_frame, text="Valid Parameter Ranges", font=("Arial", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )

        # Add column headers
        ctk.CTkLabel(main_frame, text="Parameter", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        ctk.CTkLabel(main_frame, text="Min", font=("Arial", 10, "bold")).grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )
        ctk.CTkLabel(main_frame, text="Max", font=("Arial", 10, "bold")).grid(
            row=1, column=2, sticky="w", padx=5, pady=5
        )

        # Add parameter ranges
        for i, (param, (min_val, max_val)) in enumerate(self.valid_ranges.items(), start=2):
            ctk.CTkLabel(main_frame, text=param).grid(
                row=i, column=0, sticky="w", padx=5, pady=3
            )
            ctk.CTkLabel(main_frame, text=str(min_val)).grid(
                row=i, column=1, sticky="w", padx=5, pady=3
            )
            ctk.CTkLabel(main_frame, text=str(max_val)).grid(
                row=i, column=2, sticky="w", padx=5, pady=3
            )

        # Add close button
        ctk.CTkButton(main_frame, text="Close", command=popup.destroy).grid(
            row=len(self.valid_ranges) + 2, column=0, columnspan=3, pady=10
        )

    # Method to clear error highlight when user clicks on a field
    def clear_error(self, event):
        """ Clears the error when the user clicks the textbox. """
        widget = event.widget
        if hasattr(widget, '_text_color') and widget._text_color == "red":
            widget.delete(0, "end")
            widget.configure(border_color=None, text_color=("gray10", "#DCE4EE"))

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

    # Method for main folder and respective folder for each stations
    def create_folders(self):
        self.main_folder = Path("Station Data")
        self.main_folder.mkdir(exist_ok=True)

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

    # Method to handle inputted data
    def submit_data(self, skip_validation=False):
        # Store invalid entries to keep their highlighting
        invalid_entries = []

        if not skip_validation:
            is_valid = True
            selected_date = self.yearInput.get_date()

            # Format the date as (M/01/YYYY). Sets the day as 1 for model purposes.
            selected_month = selected_date.replace(day=1).strftime("%m/%d/%Y")

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
                        entry.configure(border_color=None, text_color=("gray10", "#DCE4EE"))

            if not is_valid:
                self.validPopUp(invalid_entries)
                return

        folder_path = "Station Data"

        # Dictionary to map stations to filenames
        station_filenames = {
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

        # Iterate through all the stations
        for station, headers in self.entries.items():
            # Skip stations that are filtered out
            if not self.station_vars[station].get():
                continue

            folder_path = Path("Station Data")  # Define folder
            filename = station_filenames.get(station)

            if not filename:
                print(f"Warning: No CSV file mapped for Station {station}")
                continue  # Skip if no matching file

            file_path = folder_path / f"{filename}.csv"

            # Check if the file exists and load existing data
            existing_data = []
            if file_path.exists():
                with open(file_path, "r", newline="") as file:
                    reader = csv.reader(file)
                    existing_data = list(reader)

            # Prepare the new row to append
            row_data = [selected_month]  # Start with the month
            for param in self.headers:
                row_data.append(headers[param].get())  # Append parameter values

            # Append new data to the existing CSV file
            with open(file_path, "a", newline="") as file:  # "a" mode appends without overwriting
                writer = csv.writer(file)
                writer.writerow(row_data)  # Append the new row

            print(f"Data saved to {filename}")

    def validPopUp(self, invalid_entries=None):
        popup = ctk.CTkToplevel(self)
        popup.title("Invalid Data Detected")
        popup.geometry("350x170")
        popup.grab_set()

        msg = ctk.CTkLabel(popup, text="Some fields have invalid or missing data.\nDo you want to continue anyway?",
                           font=("Arial", 11))
        msg.pack(pady=20)

        # Add hint about valid ranges
        hint = ctk.CTkLabel(popup, text="Click 'Valid Ranges' button to see\nacceptable values for each parameter.",
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

    # Method to clear all existing data in textbox
    def clear_all(self):
        for station, headers in self.entries.items():
            # Only clear visible stations
            if self.station_vars[station].get():
                for entry_widget in headers.values():
                    entry_widget.delete(0, "end")
                    entry_widget.configure(border_color=None, text_color=("gray10", "#DCE4EE"))

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")
        # Apply filter initially to show all stations
        self.apply_filter()