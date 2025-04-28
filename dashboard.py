import customtkinter as ctk
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DashboardPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent

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
        self.csv_file = "train/merged_stations.xlsx"  # Update this to your actual file path

        self.current_station = "Station 1"  # Default station
        self.station_filter_value = self.station_names[self.current_station]  # Value to filter by in the CSV

        # Load the full dataset once
        self.full_df = self.load_all_data(self.csv_file)

        # Filter for the initial station
        self.df = self.filter_by_station(self.station_filter_value)

        self.create_widgets()

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
        """Filter the full dataset for the selected station"""
        try:
            filtered_df = self.full_df[self.full_df["Station"] == station_name].copy()
            print(f"Filtered to {len(filtered_df)} rows for {station_name}")
            years = sorted(filtered_df["Year"].dropna().unique())
            print(f"Years found for {station_name}: {years}")
            return filtered_df
        except Exception as e:
            print(f"Error filtering data for {station_name}: {e}")
            return pd.DataFrame()

    def create_widgets(self):
        dashboardlb = ctk.CTkLabel(self, text="DASHBOARD", font=("Arial", 25, "bold"))
        dashboardlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

        # Station Selection Dropdown
        self.station_label = ctk.CTkLabel(self, text="Select Station:", font=("Arial", 12))
        self.station_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.station_var = ctk.StringVar()
        self.station_var.set(self.current_station)  # Default station

        # Use CTkOptionMenu instead of tk.OptionMenu
        self.station_dropdown = ctk.CTkOptionMenu(
            self,
            variable=self.station_var,
            values=list(self.station_names.keys()),
            command=self.update_station
        )
        self.station_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Dropdown to select year
        self.year_label = ctk.CTkLabel(self, text="Select Year:", font=("Arial", 12))
        self.year_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")  # Moved to column 2

        self.year_var = ctk.StringVar()

        # Get available years from data
        available_years = sorted(self.df["Year"].dropna().unique())
        default_year = "2016" if 2016 in available_years else str(int(available_years[0])) if len(
            available_years) > 0 else "2016"
        self.year_var.set(default_year)  # Default year

        # Use available years or fallback to range
        years_for_dropdown = [str(int(y)) for y in available_years] if len(available_years) > 0 else [str(y) for y in
                                                                                                      range(2016, 2026)]

        self.year_dropdown = ctk.CTkOptionMenu(self, variable=self.year_var, values=years_for_dropdown)
        self.year_dropdown.grid(row=1, column=3, padx=10, pady=5, sticky="w")  # Moved to column 3

        # Radio buttons for parameters
        # Get parameter names from our column mappings
        parameter_names = list(self.column_mappings.values())

        self.param_var = ctk.StringVar(value="pH")  # Default selection
        self.param_buttons = []

        self.param_var_cb = {param: ctk.IntVar() for param in parameter_names}
        self.param_checkboxes = []

        self.mode = "yearly"
        self.b = ctk.CTkButton(self)

        def disp_btn():
            # Clear existing buttons before adding new ones
            for btn in self.param_checkboxes:
                btn.grid_forget()
            self.param_checkboxes.clear()

            for btn in self.param_buttons:
                btn.grid_forget()
            self.param_buttons.clear()

            if self.mode == "yearly":
                for i, param in enumerate(self.param_var_cb.keys()):
                    cb = ctk.CTkCheckBox(self, text=param, variable=self.param_var_cb[param], font=("Arial", 10))
                    cb.grid(row=2, column=i, padx=5, pady=5, sticky="w")
                    self.param_checkboxes.append(cb)

            elif self.mode == "monthly":
                for i, param in enumerate(parameter_names):
                    rb = ctk.CTkRadioButton(self, text=param, variable=self.param_var, value=param, font=("Arial", 10))
                    rb.grid(row=2, column=i, padx=5, pady=5, sticky="w")
                    self.param_buttons.append(rb)

        disp_btn()

        self.error_message = ctk.CTkLabel(self, text="No Parameter Selected.", font=("Arial", 12))

        def display_selected_data_bar():
            self.mode = "monthly"
            disp_btn()
            self.error_message.grid_forget()
            self.year_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
            self.year_dropdown.grid(row=1, column=3, padx=10, pady=5, sticky="w")

            try:
                selected_year = int(self.year_var.get())
                selected_param = self.param_var.get()

                if not selected_param:
                    self.error_message.configure(text="No Parameter Selected.")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                    return

                filtered_df = self.df[self.df["Year"] == selected_year]

                if filtered_df.empty:
                    print(f"No data available for the year {selected_year}!")
                    self.error_message.configure(text=f"No data available for {selected_year}.")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                    return

                # Create bar graph
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

                # Filter out NaN values before plotting
                valid_data = filtered_df.dropna(subset=[selected_param, "Month"])

                if valid_data.empty:
                    print(f"No valid data for {selected_param} in {selected_year}")
                    self.error_message.configure(text=f"No valid data for {selected_param} in {selected_year}")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                    return

                # Group by month and plot - this will automatically skip NaN values
                monthly_avg = valid_data.groupby("Month")[selected_param].mean()

                if not monthly_avg.empty:
                    month_nums = monthly_avg.index.tolist()
                    values = monthly_avg.values

                    ax.bar(month_nums, values, label=selected_param, alpha=0.7)  # Bar chart
                    ax.set_title(f"{self.current_station} - Monthly {selected_param} for {selected_year}")
                    ax.set_xlabel("Month")
                    ax.set_ylabel(f"{selected_param} Value")
                    ax.set_xticks(range(1, 13))  # Show ticks for all months even if no data
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    print("No valid monthly data available.")
                    self.error_message.configure(text="No valid monthly data available.")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                    return

                # Update canvas (Clear the old plot)
                if hasattr(self, 'canvas1'):
                    self.canvas1.get_tk_widget().destroy()
                self.canvas1 = FigureCanvasTkAgg(fig, master=self)
                self.canvas1.draw()
                self.canvas1.get_tk_widget().grid(row=3, column=0, columnspan=6, padx=30, pady=30)

            except Exception as e:
                print(f"Error displaying monthly data: {e}")
                self.error_message.configure(text=f"Error: {str(e)}")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

        def display_selected_data_line():
            self.mode = "yearly"
            disp_btn()
            self.error_message.grid_forget()
            self.year_label.grid_forget()
            self.year_dropdown.grid_forget()

            try:
                # Get selected parameters from checkboxes
                selected_params = [param for param, var in self.param_var_cb.items() if var.get() == 1]

                if not selected_params:
                    self.error_message.configure(text="No Parameter Selected.")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                    return

                # Create the plot
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

                # Keep track if we successfully plotted anything
                plotted_any = False

                # Plot each parameter separately - this approach skips NaN values automatically
                for param in selected_params:
                    # Drop NaN values for this parameter and year
                    valid_data = self.df.dropna(subset=[param, "Year"])

                    if valid_data.empty:
                        continue

                    # Group by year and calculate mean
                    yearly_avg = valid_data.groupby("Year")[param].mean()

                    if not yearly_avg.empty:
                        ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linestyle='-', label=param)
                        plotted_any = True

                if not plotted_any:
                    print("No valid data available for the selected parameters!")
                    self.error_message.configure(text="No valid data available for selected parameters.")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=6)
                    return

                ax.set_title(f"{self.current_station} - Yearly Data for Selected Parameters")
                ax.set_xlabel("Year")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Update canvas (Clear old plot)
                if hasattr(self, 'canvas1'):
                    self.canvas1.get_tk_widget().destroy()
                self.canvas1 = FigureCanvasTkAgg(fig, master=self)
                self.canvas1.draw()
                self.canvas1.get_tk_widget().grid(row=3, column=0, columnspan=4, padx=30, pady=30)

            except Exception as e:
                print(f"Error displaying yearly data: {e}")
                self.error_message.configure(text=f"Error: {str(e)}")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

        # Button to update graph
        monthly_btn = ctk.CTkButton(self, text="Get Monthly Data", command=display_selected_data_bar,
                                    font=("Arial", 12),
                                    fg_color="#1d97bd", text_color="white", hover_color="#1a86a8")
        monthly_btn.grid(row=5, column=0, padx=10, pady=5, sticky="w")

        yearly_btn = ctk.CTkButton(self, text=" Get Yearly Data", command=display_selected_data_line,
                                   font=("Arial", 12),
                                   fg_color="#1d97bd", text_color="white", hover_color="#1a86a8")
        yearly_btn.grid(row=5, column=1, padx=10, pady=5, sticky="w")

        # Store references to these functions to use in update_station
        self.display_selected_data_bar = display_selected_data_bar
        self.display_selected_data_line = display_selected_data_line

        # Initial yearly average line graph
        def plot_yearly_average():
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)

            try:
                if not self.df.empty:
                    # Get the parameters with sufficient data
                    params_to_plot = ["Ammonia", "Nitrate", "Phosphate"]
                    plotted_any = False

                    for param in params_to_plot:
                        # Skip NaN values by dropping them before grouping
                        valid_data = self.df.dropna(subset=[param, "Year"])

                        if not valid_data.empty:
                            yearly_avg = valid_data.groupby("Year")[param].mean()
                            ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linestyle='-', label=param)
                            plotted_any = True

                    if plotted_any:
                        ax.set_title(f"{self.current_station} - Yearly Average of Water Quality Parameters")
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Average Value")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, "No Data Available", fontsize=14, ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "No Data Available", fontsize=14, ha='center', va='center')
            except Exception as e:
                print(f"Error plotting yearly average: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", fontsize=14, ha='center', va='center')

            return fig

        # Create initial plot
        initial_fig = plot_yearly_average()
        self.canvas1 = FigureCanvasTkAgg(initial_fig, master=self)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(row=3, column=0, columnspan=4, padx=30, pady=30)

        # Store reference to this function to use in update_station
        self.plot_yearly_average = plot_yearly_average

    # Method to update station and reload data when station selection changes
    def update_station(self, selection):
        self.current_station = selection

        # Get the station filter value from our mapping
        self.station_filter_value = self.station_names.get(selection)

        try:
            # Filter data for the selected station
            self.df = self.filter_by_station(self.station_filter_value)

            # Update year dropdown with available years for this station
            available_years = sorted(self.df["Year"].dropna().unique())

            if len(available_years) > 0:
                # Update the year dropdown menu
                # CustomTkinter approach is different - need to update the menu
                year_strings = [str(int(y)) for y in available_years]
                self.year_dropdown.configure(values=year_strings)

                # Set to first available year
                self.year_var.set(str(int(available_years[0])))

            # Update the graph based on current mode
            if self.mode == "yearly":
                # Update the yearly graph
                if hasattr(self, 'canvas1'):
                    self.canvas1.get_tk_widget().destroy()
                self.canvas1 = FigureCanvasTkAgg(self.plot_yearly_average(), master=self)
                self.canvas1.draw()
                self.canvas1.get_tk_widget().grid(row=3, column=0, columnspan=4, padx=30, pady=30)
            else:
                # Update the monthly graph
                self.display_selected_data_bar()

        except Exception as e:
            print(f"Error updating station: {e}")
            if hasattr(self, 'error_message'):
                self.error_message.configure(text=f"Error loading station data: {str(e)}")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")