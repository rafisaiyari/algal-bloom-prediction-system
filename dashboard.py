import customtkinter as ctk
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator


class DashboardPage(ctk.CTkFrame):
    _data_cache = {
        'full_df': None,
        'station_data': {},
        'initialized': False
    }

    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color="#FFFFFF")
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

        self.csv_file = "CSV/merged_stations.xlsx"
        self.monthly_canvas = None
        self.yearly_canvas = None
        self.monthly_df = None
        self.yearly_df = None

        # Fix fixed dimensions for figures - prevents constant recreation
        self.fig_width = 8.0
        self.fig_height = 5.0
        self.fig_dpi = 72
        
        # Remove current_fig variables that could cause resizing loops
        # self.current_fig_width = self.fig_width
        # self.current_fig_height = self.fig_height

        # Prevent resize propagation
        self.pack_propagate(False)

        # Fixed dimensions to avoid resizing
        self.configure(width=1200, height=680)

        # Initialize frames and widgets
        self.setup_monthly_frame()
        self.setup_yearly_frame()

        if not self._data_cache['initialized']:
            self.preload_data()

        # Delay these calls to reduce initial load time
        self.after(100, self.display_monthly_data)
        self.after(200, self.display_yearly_data)

        self.is_visible = True
        
        # Add resize protection
        self._resize_protection = False

    def preload_data(self):
        """Preload all data once to avoid repeated loading"""
        print("Preloading all station data...")
        try:
            self._data_cache['full_df'] = self.load_all_data(self.csv_file)
            for station_name, station_code in self.station_names.items():
                filtered_data = self._data_cache['full_df'][self._data_cache['full_df']["Station"] == station_code].copy()
                self._data_cache['station_data'][station_name] = filtered_data
                print(f"Cached {len(filtered_data)} rows for {station_name}")
            
            self._data_cache['initialized'] = True
            print("Data preloading complete")
        except Exception as e:
            print(f"Error during data preloading: {e}")
            # Create empty dataframe to prevent crashes
            self._data_cache['full_df'] = pd.DataFrame()
            self._data_cache['initialized'] = True

    def load_all_data(self, filename):
        """Load data from Excel file with proper error handling"""
        try:
            df = pd.read_excel(filename)
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            
            # Define the columns we want to process
            numeric_columns = [
                "pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phosphate (mg/L)",
                "Dissolved Oxygen (mg/L)", "Temperature", "Chlorophyll-a (ug/L)"
            ]
            
            # Map column names for convenience
            self.column_mappings = {
                "pH (units)": "pH", "Ammonia (mg/L)": "Ammonia", "Nitrate (mg/L)": "Nitrate",
                "Inorganic Phosphate (mg/L)": "Phosphate", "Dissolved Oxygen (mg/L)": "DO",
                "Temperature": "Temperature", "Chlorophyll-a (ug/L)": "Chlorophyll-a"
            }
            
            # Create short name columns
            for full_name, short_name in self.column_mappings.items():
                if full_name in df.columns:
                    df[short_name] = df[full_name]
            
            # Convert columns to numeric, handling errors
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            print(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            print(f"Error loading Excel file {filename}: {e}")
            return pd.DataFrame()

    def filter_by_station(self, station_name):
        """Get data for a specific station from cache"""
        try:
            # First check if we have this station in our cache
            station_display_name = next((name for name, code in self.station_names.items() if code == station_name), None)
            if station_display_name and station_display_name in self._data_cache['station_data']:
                return self._data_cache['station_data'][station_display_name]
            
            # If not in cache, filter from main dataframe
            if self._data_cache['full_df'] is not None:
                return self._data_cache['full_df'][self._data_cache['full_df']["Station"] == station_name].copy()
            
            # Return empty dataframe if all else fails
            return pd.DataFrame()
        except Exception as e:
            print(f"Error retrieving data for {station_name}: {e}")
            return pd.DataFrame()

    def setup_monthly_frame(self):
        """Create frame for monthly data visualization"""
        # Create a frame to hold the monthly data visualization
        self.monthly_frame = ctk.CTkFrame(self, fg_color="#FFFFFF")
        self.monthly_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 5))
        
        # Add title label for monthly data
        self.monthly_title = ctk.CTkLabel(
            self.monthly_frame, 
            text="Monthly Water Quality Parameters", 
            font=("Arial", 16, "bold"),
            text_color="#1d97bd"
        )
        self.monthly_title.pack(side="top", pady=(5, 10))
        
        # Create a frame to hold the matplotlib figure for monthly data
        self.monthly_plot_frame = ctk.CTkFrame(self.monthly_frame, fg_color="#FFFFFF")
        self.monthly_plot_frame.pack(side="top", fill="both", expand=True)

    def setup_yearly_frame(self):
        """Create frame for yearly data visualization"""
        # Create a frame to hold the yearly data visualization
        self.yearly_frame = ctk.CTkFrame(self, fg_color="#FFFFFF")
        self.yearly_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(5, 10))
        
        # Add title label for yearly data
        self.yearly_title = ctk.CTkLabel(
            self.yearly_frame, 
            text="Yearly Water Quality Parameters", 
            font=("Arial", 16, "bold"),
            text_color="#1d97bd"
        )
        self.yearly_title.pack(side="top", pady=(5, 10))
        
        # Create a frame to hold the matplotlib figure for yearly data
        self.yearly_plot_frame = ctk.CTkFrame(self.yearly_frame, fg_color="#FFFFFF")
        self.yearly_plot_frame.pack(side="top", fill="both", expand=True)

    def display_monthly_data(self):
        """Display monthly water quality data visualization"""
        if self._resize_protection:
            return
            
        # Clean up existing canvas if any
        if self.monthly_canvas is not None:
            self.monthly_canvas.get_tk_widget().destroy()
        
        try:
            # Create sample monthly data if needed (replace with your actual data logic)
            if self._data_cache['full_df'] is None or self._data_cache['full_df'].empty:
                # Create some sample data for demonstration
                months = range(1, 13)
                ph_values = [7.2, 7.3, 7.5, 7.4, 7.6, 7.8, 7.7, 7.5, 7.4, 7.3, 7.4, 7.5]
                do_values = [5.1, 5.2, 5.0, 4.9, 4.8, 4.7, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1]
                temp_values = [28, 29, 30, 31, 32, 31, 30, 29, 28, 27, 26, 27]
                
                sample_data = {
                    'Month': months,
                    'pH': ph_values,
                    'DO': do_values,
                    'Temperature': temp_values
                }
                self.monthly_df = pd.DataFrame(sample_data)
            else:
                # Use actual data (replace with your data processing logic)
                # For demonstration, just take average of each month
                df = self._data_cache['full_df'].copy()
                self.monthly_df = df.groupby('Month')[['pH', 'DO', 'Temperature']].mean().reset_index()
            
            # Create figure with fixed size to prevent resize loops
            fig = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
            
            # Create subplots for each parameter
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            
            # Plot each parameter
            if not self.monthly_df.empty:
                ax1.plot(self.monthly_df['Month'], self.monthly_df['pH'], 'b-', marker='o')
                ax2.plot(self.monthly_df['Month'], self.monthly_df['DO'], 'g-', marker='o')
                ax3.plot(self.monthly_df['Month'], self.monthly_df['Temperature'], 'r-', marker='o')
            
            # Set titles and labels
            ax1.set_title('pH')
            ax2.set_title('Dissolved Oxygen (mg/L)')
            ax3.set_title('Temperature (°C)')
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('Month')
                ax.set_xticks(range(1, 13))
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout to prevent overlap
            fig.tight_layout()
            
            # Create canvas and display the figure
            self.monthly_canvas = FigureCanvasTkAgg(fig, master=self.monthly_plot_frame)
            self.monthly_canvas.draw()
            self.monthly_canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error displaying monthly data: {e}")
            # Create a blank figure as fallback
            fig = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Error loading monthly data", horizontalalignment='center', verticalalignment='center')
            fig.tight_layout()
            self.monthly_canvas = FigureCanvasTkAgg(fig, master=self.monthly_plot_frame)
            self.monthly_canvas.draw()
            self.monthly_canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_yearly_data(self):
        """Display yearly water quality data visualization"""
        if self._resize_protection:
            return
            
        # Clean up existing canvas if any
        if self.yearly_canvas is not None:
            self.yearly_canvas.get_tk_widget().destroy()
        
        try:
            # Create sample yearly data if needed (replace with your actual data logic)
            if self._data_cache['full_df'] is None or self._data_cache['full_df'].empty:
                # Create some sample data for demonstration
                years = range(2018, 2025)
                ph_values = [7.3, 7.4, 7.5, 7.3, 7.2, 7.1, 7.0]
                do_values = [5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4]
                temp_values = [28.5, 28.7, 29.0, 29.2, 29.5, 29.7, 30.0]
                
                sample_data = {
                    'Year': years,
                    'pH': ph_values,
                    'DO': do_values,
                    'Temperature': temp_values
                }
                self.yearly_df = pd.DataFrame(sample_data)
            else:
                # Use actual data (replace with your data processing logic)
                # For demonstration, just take average of each year
                df = self._data_cache['full_df'].copy()
                self.yearly_df = df.groupby('Year')[['pH', 'DO', 'Temperature']].mean().reset_index()
            
            # Create figure with fixed size
            fig = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
            
            # Create subplots for each parameter
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            
            # Plot each parameter
            if not self.yearly_df.empty:
                ax1.plot(self.yearly_df['Year'], self.yearly_df['pH'], 'b-', marker='o')
                ax2.plot(self.yearly_df['Year'], self.yearly_df['DO'], 'g-', marker='o')
                ax3.plot(self.yearly_df['Year'], self.yearly_df['Temperature'], 'r-', marker='o')
            
            # Set titles and labels
            ax1.set_title('pH')
            ax2.set_title('Dissolved Oxygen (mg/L)')
            ax3.set_title('Temperature (°C)')
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('Year')
                if not self.yearly_df.empty:
                    ax.set_xticks(self.yearly_df['Year'])
                    ax.set_xticklabels(self.yearly_df['Year'], rotation=45)
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout to prevent overlap
            fig.tight_layout()
            
            # Create canvas and display the figure
            self.yearly_canvas = FigureCanvasTkAgg(fig, master=self.yearly_plot_frame)
            self.yearly_canvas.draw()
            self.yearly_canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error displaying yearly data: {e}")
            # Create a blank figure as fallback
            fig = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Error loading yearly data", horizontalalignment='center', verticalalignment='center')
            fig.tight_layout()
            self.yearly_canvas = FigureCanvasTkAgg(fig, master=self.yearly_plot_frame)
            self.yearly_canvas.draw()
            self.yearly_canvas.get_tk_widget().pack(fill="both", expand=True)

    def show(self):
        """Show the dashboard page"""
        # Use resize protection to prevent multiple redraws
        self._resize_protection = True
        self.grid(row=0, column=0, sticky="nsew")
        self.is_visible = True
        
        # Reset protection after a delay to allow UI to settle
        self.after(300, self._reset_protection)
    
    def _reset_protection(self):
        """Reset resize protection after UI stabilizes"""
        self._resize_protection = False

    def hide(self):
        """Hide the dashboard page"""
        self.grid_forget()
        self.is_visible = False