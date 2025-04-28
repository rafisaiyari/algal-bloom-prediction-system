import tkinter as tk
import csv
from pathlib import Path
from tkcalendar import DateEntry


class InputDataPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
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

        super().__init__(parent, bg=bg)
        self.parent = parent
        self.create_folders()
        self.create_widgets()

    def create_widgets(self):
        # Configure the main frame to expand with window
        self.columnconfigure(0, weight=1)

        inputDatalb = tk.Label(self, text="INPUT DATA", font=("Arial", 25, "bold"))
        inputDatalb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        # Create filter frame
        self.filter_frame = tk.Frame(self, bg="#F1F1F1")
        self.filter_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        self.filter_frame.columnconfigure(0, weight=1)  # Make filter frame expandable

        # Add filter label
        tk.Label(self.filter_frame, text="Filter Stations:", font=("Arial", 12), bg="#F1F1F1").grid(row=0, column=0,
                                                                                                    sticky="w", padx=5,
                                                                                                    pady=5)

        # Add checkboxes for each station - now with automatic filtering
        checkbox_frame = tk.Frame(self.filter_frame, bg="#F1F1F1")
        checkbox_frame.grid(row=1, column=0, sticky="w", padx=5)

        for i, station in enumerate(self.stations):
            self.station_vars[station] = tk.BooleanVar(value=True)
            # Set up trace to automatically apply filter when checkbox state changes
            self.station_vars[station].trace_add("write", self.on_checkbox_change)

            self.station_checkboxes[station] = tk.Checkbutton(
                checkbox_frame,
                text=f"Station {station}",
                variable=self.station_vars[station],
                bg="#F1F1F1",
                onvalue=True,
                offvalue=False
            )
            row, col = divmod(i, 3)  # Arrange checkboxes in rows of 3
            self.station_checkboxes[station].grid(row=row, column=col, sticky="w", padx=10, pady=2)

        # Add filter controls - only Select All and Deselect All buttons
        filter_controls = tk.Frame(self.filter_frame, bg="#F1F1F1")
        filter_controls.grid(row=2, column=0, sticky="w", padx=5, pady=10)

        # Buttons for filtering
        select_all_btn = tk.Button(filter_controls, text="Select All", command=self.select_all_stations)
        select_all_btn.grid(row=0, column=0, padx=5)

        deselect_all_btn = tk.Button(filter_controls, text="Deselect All", command=self.deselect_all_stations)
        deselect_all_btn.grid(row=0, column=1, padx=5)

        # Create main data frame that will contain all station entries
        self.main_data_frame = tk.Frame(self, bg="#F1F1F1")
        self.main_data_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)

        # Make all columns expandable
        for i in range(len(self.headers) + 1):
            self.main_data_frame.columnconfigure(i, weight=1)

        # Labeling for headers with larger font and better spacing
        station_lbl = tk.Label(self.main_data_frame, text="STATIONS", font=("Arial", 12, "bold"), bg="#F1F1F1")
        station_lbl.grid(column=0, row=0, padx=10, pady=10, sticky="w")

        # Create tooltip frame for parameter validation ranges
        tooltip_frame = tk.Frame(self.main_data_frame, bg="#F1F1F1")
        tooltip_frame.grid(row=0, column=len(self.headers) + 1, padx=5, pady=5, sticky="e")

        tooltip_btn = tk.Button(tooltip_frame, text="ⓘ Valid Ranges", command=self.show_valid_ranges)
        tooltip_btn.pack(padx=5, pady=5)

        # Header labels with wraplength for better display
        for col, header in enumerate(self.headers, start=1):
            header_lbl = tk.Label(
                self.main_data_frame,
                text=header,
                font=("Arial", 10),
                bg="#F1F1F1",
                wraplength=100,  # Allow text to wrap
                justify="center"
            )
            header_lbl.grid(column=col, row=0, padx=10, pady=10)

        # Create entry widgets for each station and parameter
        for row, station in enumerate(self.stations, start=1):
            # Create label for station
            station_label = tk.Label(
                self.main_data_frame,
                text=f"Station {station}:",
                bg="#F1F1F1",
                anchor="w"
            )
            station_label.grid(column=0, row=row, padx=10, pady=5, sticky="w")

            # Store station frame reference (though we're not using separate frames now)
            self.station_frames[station] = station_label
            self.entries[station] = {}

            # Create entry widgets for each parameter
            for col, header in enumerate(self.headers, start=1):
                entry = tk.Entry(self.main_data_frame, width=15)  # Fixed width entry
                entry.grid(column=col, row=row, padx=10, pady=5, sticky="ew")
                self.entries[station][header] = entry

                # Set up focus event to clear error when clicked
                entry.bind("<FocusIn>", self.clear_error)

        # Creates a new frame for controls
        self.control_frame = tk.Frame(self, bg="#F1F1F1")
        self.control_frame.grid(row=3, column=0, pady=20, sticky="ew")

        # Center the control frame content
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(4, weight=1)

        # Date Entry and buttons in centered inner frame
        control_inner_frame = tk.Frame(self.control_frame, bg="#F1F1F1")
        control_inner_frame.grid(row=0, column=1, columnspan=3)

        tk.Label(control_inner_frame, text="Select Year and Month:", bg="#F1F1F1").grid(column=0, row=0, padx=5, pady=5)
        self.yearInput = DateEntry(control_inner_frame, selectmode="day", width=10, date_pattern="y-mm-dd")
        self.yearInput.grid(column=1, row=0, padx=5, pady=5)

        # Buttons
        self.submit_button = tk.Button(control_inner_frame, text="Submit", command=self.submit_data)
        self.submit_button.grid(column=2, row=0, padx=10, pady=5)

        self.clear_button = tk.Button(control_inner_frame, text="Clear All", command=self.clear_all)
        self.clear_button.grid(column=3, row=0, padx=10, pady=5)

    def show_valid_ranges(self):
        # Create a popup to display valid parameter ranges
        popup = tk.Toplevel(self)
        popup.title("Valid Parameter Ranges")
        popup.geometry("400x350")
        popup.grab_set()

        # Create main frame
        main_frame = tk.Frame(popup)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add header
        tk.Label(main_frame, text="Valid Parameter Ranges", font=("Arial", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )

        # Add column headers
        tk.Label(main_frame, text="Parameter", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        tk.Label(main_frame, text="Min", font=("Arial", 10, "bold")).grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )
        tk.Label(main_frame, text="Max", font=("Arial", 10, "bold")).grid(
            row=1, column=2, sticky="w", padx=5, pady=5
        )

        # Add parameter ranges
        for i, (param, (min_val, max_val)) in enumerate(self.valid_ranges.items(), start=2):
            tk.Label(main_frame, text=param).grid(
                row=i, column=0, sticky="w", padx=5, pady=3
            )
            tk.Label(main_frame, text=str(min_val)).grid(
                row=i, column=1, sticky="w", padx=5, pady=3
            )
            tk.Label(main_frame, text=str(max_val)).grid(
                row=i, column=2, sticky="w", padx=5, pady=3
            )

        # Add close button
        tk.Button(main_frame, text="Close", command=popup.destroy).grid(
            row=len(self.valid_ranges) + 2, column=0, columnspan=3, pady=10
        )

    # Method to clear error highlight when user clicks on a field
    def clear_error(self, event):
        """ Clears the error when the user clicks the textbox. """
        widget = event.widget
        if widget.cget("fg") == "red":
            widget.delete(0, tk.END)
            widget.config(highlightthickness=1, highlightbackground="black", fg="black")

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
                        entry.delete(0, tk.END)
                        entry.insert(0, "ⓘ")
                        entry.config(highlightthickness=2, highlightbackground="red", fg="red")
                        invalid_entries.append(entry)
                        is_valid = False
                        continue

                    # Validate against parameter ranges
                    valid, error_msg = self.validate_parameter(param, value)
                    if not valid:
                        entry.delete(0, tk.END)
                        entry.insert(0, "Out of Range")
                        entry.config(highlightthickness=2, highlightbackground="red", fg="red")
                        invalid_entries.append(entry)
                        is_valid = False
                    else:
                        entry.config(highlightthickness=1, highlightbackground="black", fg="black")

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
        popup = tk.Toplevel(self)
        popup.title("Invalid Data Detected")
        popup.geometry("350x170")
        popup.grab_set()

        msg = tk.Label(popup, text="Some fields have invalid or missing data.\nDo you want to continue anyway?",
                       font=("Arial", 11))
        msg.pack(pady=20)

        # Add hint about valid ranges
        hint = tk.Label(popup, text="Click 'Valid Ranges' button to see\nacceptable values for each parameter.",
                        font=("Arial", 9), fg="blue")
        hint.pack(pady=0)

        button_frame = tk.Frame(popup)
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
                        entry.config(highlightthickness=2, highlightbackground="red", fg="red")
            popup.destroy()

        continueBtn = tk.Button(button_frame, text="Continue Anyway", width=15, command=continue_anyway)
        continueBtn.grid(row=0, column=0, padx=10)

        cancelBtn = tk.Button(button_frame, text="Go Back", width=15, command=go_back)
        cancelBtn.grid(row=0, column=1, padx=10)

        # Make sure widget highlighting is preserved when popup is closed
        popup.protocol("WM_DELETE_WINDOW", go_back)

    # Method to clear all existing data in textbox
    def clear_all(self):
        for station, headers in self.entries.items():
            # Only clear visible stations
            if self.station_vars[station].get():
                for entry_widget in headers.values():
                    entry_widget.delete(0, tk.END)
                    entry_widget.config(highlightthickness=1, highlightbackground="black", fg="black")

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")
        # Apply filter initially to show all stations
        self.apply_filter()
