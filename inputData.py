import tkinter as tk
import csv
from pathlib import Path
from tkcalendar import DateEntry


class InputDataPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        self.headers = ["pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)",
                        "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)"]
        self.stations = ["I", "II", "IV", "V", "VIII", "XV", "XVI", "XVII", "XVIII"]
        self.entries = {}

        super().__init__(parent, bg=bg)
        self.parent = parent
        self.create_folders()
        self.create_widgets()

    def create_widgets(self):
        inputDatalb = tk.Label(self, text="INPUT DATA", font=("Arial", 25, "bold"))
        inputDatalb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        # Labeling for Stations
        tk.Label(self, text="STATIONS", font=("Arial", 12, "bold")).grid(column=0, row=1, padx=5, pady=5)
        for col, header in enumerate(self.headers, start=1):
            tk.Label(self, text=header).grid(column=col, row=1, padx=5, pady=5)

        # Textbox for each stations (Rows)
        for row, station in enumerate(self.stations, start=2):
            tk.Label(self, text=f"Station {station}:").grid(column=0, row=row, padx=5, pady=5)
            self.entries[station] = {}

            # Textbox for each paramters (Columns)
            for col, header in enumerate(self.headers, start=1):
                entry = tk.Entry(self)
                entry.grid(column=col, row=row, padx=5, pady=5)
                self.entries[station][header] = entry

        # Creates a new frame for controls
        self.control_frame = tk.Frame(self, bg="#F1F1F1")
        self.control_frame.grid(row=len(self.stations) + 2, column=0, columnspan=len(self.headers) + 1, pady=20)

        # Date Entry
        tk.Label(self.control_frame, text="Select Year and Month:", bg="#F1F1F1").grid(column=0, row=0, padx=5, pady=5)
        self.yearInput = DateEntry(self.control_frame, selectmode="day", width=10, date_pattern="y-mm-dd")
        self.yearInput.grid(column=1, row=0, padx=5, pady=5)

        # Buttons
        self.submit_button = tk.Button(self.control_frame, text="Submit", command=self.submit_data)
        self.submit_button.grid(column=2, row=0, padx=10, pady=5)

        self.clear_button = tk.Button(self.control_frame, text="Clear All", command=self.clear_all)
        self.clear_button.grid(column=3, row=0, padx=10, pady=5)

    # Method for main folder and respective folder for each stations
    def create_folders(self):
        self.main_folder = Path("Station Data")
        self.main_folder.mkdir(exist_ok=True)

    # Method to handle inputted data
    def submit_data(self):
        is_valid = True
        selected_date = self.yearInput.get_date()

        # Format the date as (M/01/YYYY). Sets the day as 1 for model purposes.
        selected_month = selected_date.replace(day=1).strftime("%m/%d/%Y")

        # Method for validating inputs
        def clear_error(event):
            """ Clears the error when the user clicks the textbox. """
            widget = event.widget
            if widget.cget("fg") == "red":
                widget.delete(0, tk.END)
                widget.config(highlightthickness=1, highlightbackground="black", fg="black")  # Reset border

        for station, headers in self.entries.items():
            for param, entry in headers.items():
                value = entry.get().strip()
                entry.bind("<FocusIn>", clear_error)

                if value in ["Invalid Input", "ⓘ"]:
                    value = ""

                try:
                    if not value:
                        entry.delete(0, tk.END)
                        entry.insert(0, "ⓘ")
                        entry.config(highlightthickness=1, highlightbackground="red", fg="red")
                        is_valid = False
                        continue

                    float_value = float(value)
                    entry.config(highlightthickness=1, highlightbackground="black", fg="black")

                except ValueError:
                    entry.delete(0, tk.END)
                    entry.insert(0, "Invalid Input")
                    entry.config(highlightthickness=1, highlightbackground="red", fg="red")
                    is_valid = False

        if not is_valid:
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

    # Method to clear all existing data in textbox
    def clear_all(self):
        for station, headers in self.entries.items():
            for entry_widget in headers.values():
                entry_widget.delete(0, tk.END)

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")