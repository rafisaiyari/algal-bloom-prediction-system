import customtkinter as ctk
import webview
import os
from heatmapper import Heatmapper as hm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PredictionPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Define paths to data files
        self.excel_path = "CSV/merged_stations.xlsx"
        self.geojson_path = "heatmapper/stations_final.geojson"
        
        # Initialize the heatmap generator
        self.heatmap = None
        
        # Create UI widgets
        self.create_widgets()
        
        # Initialize status
        self.initialized = False
        self.canvas = None
        
        # Initialize the heatmap generator in a separate thread to avoid blocking UI
        self.after(100, self.initialize_heatmap)
        self.rowconfigure(0, weight=1)
        self.columnconfigure((0,1), weight=1)

    def initialize_heatmap(self):
        """Initialize the heatmap generator in the background"""
        try:
            self.heatmap = hm.HeatmapByParameter(self.excel_path, self.geojson_path)
            self.initialized = True
            self.status_label.configure(text="Status: Ready", text_color="green")
            
            # Load available years from data
            self.load_available_years()
            
        except Exception as e:
            self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")
    
    def load_available_years(self):
        """Load available years from the data"""
        try:
            if self.heatmap and hasattr(self.heatmap, 'data') and not self.heatmap.data.empty:
                years = sorted(self.heatmap.data["Year"].unique())
                self.year_dropdown.configure(values=[str(y) for y in years])
                
                # Set default to most recent year
                if years:
                    self.year_var.set(str(years[-1]))
        except Exception as e:
            print(f"Error loading available years: {e}")

    def create_widgets(self):
        """Create all UI widgets"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # Left panel for controls
        left_panel = ctk.CTkFrame(self)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure left panel grid
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_columnconfigure(1, weight=1)
        
        # Title
        ctk.CTkLabel(
            left_panel, 
            text="WATER QUALITY HEATMAP", 
            font=("Arial", 20, "bold"),
            text_color="#1d97bd"
        ).grid(row=0, column=0, columnspan=2, pady=(10, 20))

        # Initialize variables
        self.year_var = ctk.StringVar(value="2023")
        self.month_var = ctk.StringVar(value="1")
        self.param_var = ctk.StringVar(value="Nitrate")

        # Year selection
        ctk.CTkLabel(left_panel, text="Year:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.year_dropdown = ctk.CTkOptionMenu(
            left_panel, 
            variable=self.year_var,
            values=[str(y) for y in range(2016, 2025)],
            command=self.update_preview
        )
        self.year_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Month selection
        ctk.CTkLabel(left_panel, text="Month:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        month_options = [
            "1 (Jan)", "2 (Feb)", "3 (Mar)", "4 (Apr)", 
            "5 (May)", "6 (Jun)", "7 (Jul)", "8 (Aug)",
            "9 (Sep)", "10 (Oct)", "11 (Nov)", "12 (Dec)"
        ]
        self.month_dropdown = ctk.CTkOptionMenu(
            left_panel, 
            variable=self.month_var,
            values=month_options,
            command=self.update_preview
        )
        self.month_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Parameter selection
        ctk.CTkLabel(left_panel, text="Parameter:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.param_dropdown = ctk.CTkOptionMenu(
            left_panel, 
            variable=self.param_var,
            values=[
                "Nitrate", "Phosphate", "Dissolved Oxygen", "pH", 
                "Ammonia", "Chlorophyll-a", "Temperature"
            ],
            command=self.update_preview
        )
        self.param_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Status label
        self.status_label = ctk.CTkLabel(
            left_panel, 
            text="Status: Initializing...",
            text_color="orange"
        )
        self.status_label.grid(row=4, column=0, columnspan=2, pady=(15, 5))

        # Information frame
        info_frame = ctk.CTkFrame(left_panel, fg_color=("#F0F0F0", "#2A2A2A"))
        info_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            info_frame, 
            text="Parameter Information",
            font=("Arial", 14, "bold")
        ).pack(pady=(10, 5))
        
        self.info_text = ctk.CTkTextbox(info_frame, height=100, width=250)
        self.info_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.info_text.insert("1.0", "Select a parameter to view information about its thresholds and significance.")
        self.info_text.configure(state="disabled")

        # Generate button
        self.generate_button = ctk.CTkButton(
            left_panel, 
            text="Generate Heatmap", 
            command=self.generate_heatmap,
            fg_color="#1d97bd",
            hover_color="#176d8a"
        )
        self.generate_button.grid(row=6, column=0, columnspan=2, pady=15)

        # Right panel for map preview
        self.map_frame = ctk.CTkFrame(self)
        self.map_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.map_frame.grid_rowconfigure(0, weight=1)
        self.map_frame.grid_columnconfigure(0, weight=1)
        
        # Title for preview
        ctk.CTkLabel(
            self.map_frame, 
            text="PREVIEW", 
            font=("Arial", 16, "bold"),
            text_color="#1d97bd"
        ).pack(pady=(10, 5))
        
        # Preview frame
        self.preview_frame = ctk.CTkFrame(self.map_frame, fg_color=("#F8F8F8", "#222222"))
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initial preview message
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Select parameters to see a preview",
            font=("Arial", 12)
        )
        self.preview_label.pack(fill="both", expand=True)

    def update_parameter_info(self):
        """Update the parameter information text box"""
        param = self.param_var.get()
        
        # Enable text box for editing
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        
        # Parameter information dictionary
        param_info = {
            "Nitrate": "Nitrate is a key nutrient for algal growth. High levels (>1 mg/L) can contribute to eutrophication and algal blooms.\n\nThresholds:\n• <0.5 mg/L: Good\n• 0.5-1.0 mg/L: Fair\n• 1.0-2.0 mg/L: Moderate\n• 2.0-5.0 mg/L: Poor\n• >5.0 mg/L: Very Poor",
            
            "Phosphate": "Phosphate is often the limiting nutrient in freshwater systems. Even small increases can trigger algal blooms.\n\nThresholds:\n• <0.05 mg/L: Good\n• 0.05-0.1 mg/L: Fair\n• 0.1-0.2 mg/L: Moderate\n• 0.2-0.5 mg/L: Poor\n• >0.5 mg/L: Very Poor",
            
            "Dissolved Oxygen": "Dissolved oxygen is crucial for aquatic life. Low levels can indicate eutrophication or organic pollution.\n\nThresholds:\n• >8.0 mg/L: Excellent\n• 6.0-8.0 mg/L: Good\n• 4.0-6.0 mg/L: Fair\n• 2.0-4.0 mg/L: Poor\n• <2.0 mg/L: Very Poor (Hypoxic)",
            
            "pH": "pH affects the solubility of nutrients and metals. Most aquatic life prefers pH between 6.5-8.5.\n\nThresholds:\n• 6.5-7.5: Optimal\n• 6.0-6.5: Slightly Acidic\n• 7.5-8.5: Slightly Alkaline\n• <6.0 or >8.5: Potentially Harmful",
            
            "Ammonia": "Ammonia is toxic to aquatic life, especially at higher pH levels. It indicates fresh organic pollution.\n\nThresholds:\n• <0.1 mg/L: Good\n• 0.1-0.5 mg/L: Fair\n• 0.5-1.0 mg/L: Moderate\n• 1.0-2.0 mg/L: Poor\n• >2.0 mg/L: Very Poor",
            
            "Chlorophyll-a": "Chlorophyll-a is a direct indicator of algal biomass. High levels indicate algal blooms.\n\nThresholds:\n• <5 μg/L: Good\n• 5-10 μg/L: Fair\n• 10-20 μg/L: Moderate Bloom\n• 20-40 μg/L: Severe Bloom\n• >40 μg/L: Very Severe Bloom",
            
            "Temperature": "Temperature affects metabolic rates, dissolved oxygen, and many chemical processes in water.\n\nThresholds:\n• <25°C: Good for most tropical species\n• 25-28°C: Optimal for many species\n• 28-30°C: High\n• 30-32°C: Very High\n• >32°C: Potentially Stressful"
        }
        
        # Insert the information for the selected parameter
        if param in param_info:
            self.info_text.insert("1.0", param_info[param])
        else:
            self.info_text.insert("1.0", f"No specific information available for {param}.")
        
        # Disable editing
        self.info_text.configure(state="disabled")

    def update_preview(self, *args):
        """Update the preview based on selected parameters"""
        if not self.initialized or not self.heatmap:
            return
            
        # Update parameter information
        self.update_parameter_info()
        
        # Clear previous preview
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            
        # Remove previous label if exists
        if hasattr(self, 'preview_label') and self.preview_label:
            self.preview_label.pack_forget()
        
        # Get selected parameters
        try:
            year = int(self.year_var.get())
            month = int(self.month_var.get().split()[0])  # Extract number from "1 (Jan)"
            param = self.param_var.get()
            
            # Create figure for preview
            fig = plt.figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Get data for the specified parameter from all stations
            param_column = self.heatmap.get_parameter_column(param)
            
            # Filter data for selected year and month
            filtered_data = self.heatmap.data[
                (self.heatmap.data["Year"] == year) &
                (self.heatmap.data["Month"] == month)
            ]
            
            # Check if we have data
            if filtered_data.empty or param_column not in filtered_data.columns:
                ax.text(0.5, 0.5, "No data available for selected parameters", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
            else:
                # Group by station and calculate mean
                station_data = filtered_data.groupby("Station")[param_column].mean().reset_index()
                
                # Map station codes to names for better display
                station_names = []
                for station_code in station_data["Station"]:
                    # Find display name by matching code
                    display_name = next((k for k, v in self.heatmap.station_id_mapping.items() 
                                      if v == station_code), station_code)
                    # Just keep the number part (e.g., "Station_1" -> "1")
                    if "_" in display_name:
                        display_name = display_name.split("_")[1]
                    station_names.append(display_name)
                
                # Create bar chart
                bars = ax.bar(station_names, station_data[param_column])
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
                
                # Set title and labels
                month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
                ax.set_title(f"{param} Levels - {month_name} {year}")
                ax.set_ylabel(param_column)
                ax.set_xlabel("Station")
                
                # Color bars based on thresholds
                if param_column in self.heatmap.color_mappings:
                    thresholds = self.heatmap.color_mappings[param_column]["thresholds"]
                    colors = self.heatmap.color_mappings[param_column]["colors"]
                    
                    for i, bar in enumerate(bars):
                        value = bar.get_height()
                        color_idx = 0
                        for j, threshold in enumerate(thresholds):
                            if value < threshold:
                                color_idx = j
                                break
                            color_idx = j + 1
                        bar.set_color(colors[color_idx])
                
                # Rotate x labels for better readability
                plt.xticks(rotation=45)
            
            # Adjust layout
            fig.tight_layout()
            
            # Display in the preview frame
            self.canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error updating preview: {e}")
            # Display error message
            if hasattr(self, 'preview_label') and self.preview_label:
                self.preview_label.configure(text=f"Error: {str(e)}")
                self.preview_label.pack(fill="both", expand=True)
            else:
                self.preview_label = ctk.CTkLabel(
                    self.preview_frame,
                    text=f"Error: {str(e)}",
                    text_color="red"
                )
                self.preview_label.pack(fill="both", expand=True)

    def generate_heatmap(self):
        """Generate and display the heatmap"""
        if not self.initialized:
            self.status_label.configure(text="Status: Not ready yet. Please wait...", text_color="orange")
            return
            
        try:
            # Get selected parameters
            year = int(self.year_var.get())
            month = int(self.month_var.get().split()[0])  # Extract number from "1 (Jan)"
            param = self.param_var.get()
            
            # Update status
            self.status_label.configure(text="Status: Generating heatmap...", text_color="blue")
            
            # Define output path
            output_dir = "heatmapper"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "station_heatmap.html")
            
            # Generate the heatmap
            success = self.heatmap.generate_map(param, year, month, output_path=output_path)
            
            if success:
                self.status_label.configure(text="Status: Heatmap generated successfully", text_color="green")
                
                # Open the heatmap in a webview
                webview.create_window(
                    f"Water Quality Heatmap - {param} ({month}/{year})", 
                    output_path,
                    width=900,
                    height=700
                )
                webview.start()
            else:
                self.status_label.configure(text="Status: No data available for selected parameters", text_color="orange")
                
                # Still open the heatmap as it will show a message about no data
                webview.create_window(
                    f"Water Quality Heatmap - No Data", 
                    output_path,
                    width=900,
                    height=700
                )
                webview.start()
                
        except Exception as e:
            self.status_label.configure(text=f"Status: Error - {str(e)}", text_color="red")
            print(f"Error generating heatmap: {e}")

    def month_str_to_number(self, m):
        """Convert month string to number (kept for compatibility)"""
        months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                  "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                  "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        return months.get(m, 1)  # Default to January if not found

    def show(self):
        """Show the prediction page"""
        self.grid(row=0, column=0, sticky="nsew")
        
        # Initialize grid configuration for proper display
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # If not initialized yet, try to initialize
        if not self.initialized and self.heatmap is None:
            self.initialize_heatmap()