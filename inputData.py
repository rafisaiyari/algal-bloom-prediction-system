import tkinter as tk
import csv
import os

from tkcalendar import DateEntry

class inputDataPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        self.headers = ["pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)", "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)"]
        self.stations = ["I", "II", "IV", "V", "VII", "XV", "XVI", "XVII", "XVIII"]
        self.entries = {}

        super().__init__(parent, bg=bg)
        self.parent = parent
        self.create_widgets()
    
    def create_widgets(self):
        inputDatalb = tk.Label(self, text="INPUT DATA", font=("Arial", 25, "bold"))
        inputDatalb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        tk.Label(self, text="STATIONS").grid(column=0, row=1, padx=5, pady=5)
        for col, header in enumerate(self.headers, start=1):
            tk.Label(self, text=header).grid(column=col, row=1, padx=5, pady=5)

        for row, station in enumerate(self.stations, start=2):
            tk.Label(self, text=f"Station {station}:").grid(column=0, row=row, padx=5, pady=5)
            self.entries[station] = {}

            for col, header in enumerate(self.headers, start=1):
                entry = tk.Entry(self)
                entry.grid(column=col, row=row, padx=5, pady=5)
                self.entries[station][header] = entry
        
        self.control_frame = tk.Frame(self, bg="#F1F1F1")
        self.control_frame.grid(row=len(self.stations) + 2, column=0, columnspan=len(self.headers) + 1, pady=20)
        
        tk.Label(self.control_frame, text="Select Year and Month:", bg="#F1F1F1").grid(column=0, row=0, padx=5, pady=5)
        self.yearInput = DateEntry(self.control_frame, selectmode="day", width=10, date_pattern="y-mm-dd")
        self.yearInput.grid(column=1, row=0, padx=5, pady=5)

        self.submit_button = tk.Button(self.control_frame, text="Submit", command=self.submit_data)
        self.submit_button.grid(column=2, row=0, padx=10, pady=5)

        self.clear_button = tk.Button(self.control_frame, text="Clear All", command=self.clear_all)
        self.clear_button.grid(column=3, row=0, padx=10, pady=5)
    
    def submit_data(self):
        selected_date = self.yearInput.get_date()
        selected_month = selected_date.strftime("%b %Y")  # Format the date as 'Mon YYYY'

        # Iterate through all the stations
        for station, headers in self.entries.items():
            # Define the filename based on the station (e.g., Station_I_Data.csv)
            filename = f"Station_{station}_Data.csv"

            # Check if the file exists and load existing data
            existing_data = []
            
            if os.path.exists(filename):
                with open(filename, "r") as file:
                    reader = csv.reader(file)
                    existing_data = list(reader)  # Read all existing data into the list

            # If the file doesn't exist, initialize headers
            if not existing_data:
                existing_data.append(["Date", "pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)", "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)"])

            # Prepare the collected data for the current input
            row_data = [selected_month]  # Start with the month
            for param in [
                "pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)", 
                "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)"
            ]:
                row_data.append(headers[param].get())  # Append each parameter value

            # Append the new row to existing data
            existing_data.append(row_data)

            # Write the updated data back to the file
            with open(filename, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(existing_data)  # Write all rows at once, including headers and data

            print(f"Data saved to {filename}")

    def clear_all(self):
        for station, headers in self.entries.items():
            for entry_widget in headers.values():
                entry_widget.delete(0, tk.END)


    def show(self):
        self.grid(row=0, column=1, sticky="nsew")