import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DashboardPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.parent = parent
        # Initialize the station CSV files dictionary
        self.station_files = {
            "Station 1": "CSV/Station_1_CWB.csv",
            "Station 2": "CSV/Station_2_EB.csv",
            "Station 4": "CSV/Station_4_CB.csv",
            "Station 5": "CSV/Station_5_NWB.csv",
            "Station 8": "CSV/Station_8_SouthB.csv",
            "Station 15": "CSV/Station_15_SP.csv",
            "Station 16": "CSV/Station_16_SR.csv",
            "Station 17": "CSV/Station_17_Sanctuary.csv",
            "Station 18": "CSV/Station_18_Pagsanjan.csv"
        }
        self.current_station = "Station 1"  # Default station
        self.csv_file = self.station_files[self.current_station]
        self.create_widgets()

    def create_widgets(self):
        dashboardlb = tk.Label(self, text="DASHBOARD", font=("Arial", 25, "bold"))
        dashboardlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

        # Station Selection Dropdown
        self.station_label = tk.Label(self, text="Select Station:", font=("Arial", 12))
        self.station_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.station_var = tk.StringVar()
        self.station_var.set(self.current_station)  # Default station
        
        self.station_dropdown = tk.OptionMenu(
            self, 
            self.station_var, 
            *self.station_files.keys(),
            command=self.update_station
        )
        self.station_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        def load_csv_data_for_dash(filename):
            try:
                # Load CSV file
                df = pd.read_csv(filename)
                
                # Standardize Date column format across all CSVs
                # First, try to parse the date in the format 'MMM-YY'
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format="%b-%y", errors='coerce')
                except:
                    # If that fails, try other common formats
                    try:
                        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                    except:
                        print(f"Warning: Unable to parse dates in {filename}")
                
                # Create year and month columns
                df["Year"] = df["Date"].dt.year
                df["Month"] = df["Date"].dt.month
                
                # Convert numeric columns and handle NaN values
                numeric_columns = ["pH", "Ammonia", "Nitrate", "Phosphate", 
                                  "Chlorophyll-a", "Temperature", "DO", "Phytoplankton"]
                
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                
                print(f"Loaded {len(df)} rows from {filename}")
                years = sorted(df["Year"].dropna().unique())
                print(f"Years found: {years}")
                
                return df
            except Exception as e:
                print(f"Error loading CSV {filename}: {e}")
                return pd.DataFrame()

        # Load Data for the initial station
        self.df = load_csv_data_for_dash(self.csv_file)
        
        # Function to reload data when station changes
        self.load_csv_data = load_csv_data_for_dash  # Save reference to use later

        # Dropdown to select year
        self.year_label = tk.Label(self, text="Select Year:", font=("Arial", 12))
        self.year_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")  # Moved to column 2

        self.year_var = tk.StringVar()
        
        # Get available years from data
        available_years = sorted(self.df["Year"].dropna().unique())
        default_year = "2016" if 2016 in available_years else str(int(available_years[0])) if len(available_years) > 0 else "2016"
        self.year_var.set(default_year)  # Default year
        
        # Use available years or fallback to range
        years_for_dropdown = [str(int(y)) for y in available_years] if len(available_years) > 0 else [str(y) for y in range(2016, 2026)]
        
        self.year_dropdown = tk.OptionMenu(self, self.year_var, *years_for_dropdown)
        self.year_dropdown.grid(row=1, column=3, padx=10, pady=5, sticky="w")  # Moved to column 3

        # Radio buttons for parameters
        self.param_var = tk.StringVar(value="pH")  # Default selection
        self.param_buttons = []

        self.param_var_cb = {param: tk.IntVar() for param in
                             ["pH", "Ammonia", "Nitrate", "Phosphate", "Chlorophyll-a", "Temperature", "DO", "Phytoplankton"]}
        self.param_checkboxes = []

        self.mode = "yearly"
        self.b = tk.Button()

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
                    cb = tk.Checkbutton(self, text=param, variable=self.param_var_cb[param], font=("Arial", 10))
                    cb.grid(row=2, column=i, padx=5, pady=5, sticky="w")
                    self.param_checkboxes.append(cb)

            elif self.mode == "monthly":
                for i, param in enumerate(
                        ["pH", "Ammonia", "Nitrate", "Phosphate", "Chlorophyll-a", "Temperature", "DO", "Phytoplankton"]):
                    rb = tk.Radiobutton(self, text=param, variable=self.param_var, value=param, font=("Arial", 10))
                    rb.grid(row=2, column=i, padx=5, pady=5, sticky="w")
                    self.param_buttons.append(rb)

        disp_btn()

        self.error_message = tk.Label(self, text="No Parameter Selected.", font=("Arial", 12))

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
                    self.error_message.config(text="No Parameter Selected.")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                    return

                filtered_df = self.df[self.df["Year"] == selected_year]

                if filtered_df.empty:
                    print(f"No data available for the year {selected_year}!")
                    self.error_message.config(text=f"No data available for {selected_year}.")
                    self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                    return

                # Create bar graph
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                
                # Filter out NaN values before plotting
                valid_data = filtered_df.dropna(subset=[selected_param, "Month"])
                
                if valid_data.empty:
                    print(f"No valid data for {selected_param} in {selected_year}")
                    self.error_message.config(text=f"No valid data for {selected_param} in {selected_year}")
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
                    self.error_message.config(text="No valid monthly data available.")
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
                self.error_message.config(text=f"Error: {str(e)}")
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
                    self.error_message.config(text="No Parameter Selected.")
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
                    self.error_message.config(text="No valid data available for selected parameters.")
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
                self.error_message.config(text=f"Error: {str(e)}")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

        # Button to update graph
        monthly_btn = tk.Button(self, text="Get Monthly Data", command=display_selected_data_bar, font=("Arial", 12),
                                bg="#1d97bd", fg="white")
        monthly_btn.grid(row=5, column=0, padx=10, pady=5, sticky="w")

        yearly_btn = tk.Button(self, text=" Get Yearly Data", command=display_selected_data_line, font=("Arial", 12),
                               bg="#1d97bd", fg="white")
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
        
        # Get the corresponding CSV file
        csv_file = self.station_files.get(selection, "CSV/Station_1_CWB.csv")  # Default to Station 1 if not found
        
        # Update the CSV file path
        self.csv_file = csv_file
        
        try:
            # Reload data for the selected station
            self.df = self.load_csv_data(self.csv_file)
            
            # Update year dropdown with available years for this station
            available_years = sorted(self.df["Year"].dropna().unique())
            
            if len(available_years) > 0:
                # Update the year dropdown menu
                menu = self.year_dropdown["menu"]
                menu.delete(0, "end")
                
                for year in [str(int(y)) for y in available_years]:
                    menu.add_command(label=year, command=lambda y=year: self.year_var.set(y))
                
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
                self.error_message.config(text=f"Error loading station data: {str(e)}")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")