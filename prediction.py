import customtkinter as ctk
import webview
import os
import numpy as np
import pandas as pd
import folium
from folium import plugins
import geopandas as gpd
from datetime import datetime
from heatmapper.Heatmapper import HeatmapByParameter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import OpenMeteo libraries
import openmeteo_requests
import requests_cache
from retry_requests import retry



class PredictionPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="#FFFFFF")
        
        # File paths and coordinates
        self.geojson_path = "heatmapper/stations_final.geojson"
        self.laguna_coords = {"lat": 14.35, "lon": 121.2}
        self.station_coords = []  
        
        # Add extreme values toggle flag
        self.use_extreme_values = ctk.BooleanVar(value=True)
        
        self.data = pd.read_csv("CSV/chlorophyll_predictions_by_station.csv")
        self.data["Date"] = pd.to_datetime(self.data["Date"])

        self.data = self.data.dropna(subset=['Date'])

        self.data["Year"] = self.data["Date"].dt.year
        self.data["Month"] = self.data["Date"].dt.month_name()

        
        # Load station coordinates on initialization
        self.load_station_coordinates()

        self.title_label = ctk.CTkLabel(
            self,
            text="PREDICTION PAGE",
            font=("Segoe UI", 25, "bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        # Create top control frame with fixed height
        self.control_frame = ctk.CTkFrame(
            self, 
            height=200, 
            fg_color="#1f6aa5",
            border_width=0  # Remove shadow/border
        )
        self.columnconfigure(0, weight=1)  # Allow the control frame to expand
        self.rowconfigure(1, weight=1)
        self.control_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        self.control_frame.grid_rowconfigure(0, weight=1)
        
        self.controls_container = ctk.CTkFrame(
            self.control_frame, 
            fg_color="#FFFFFF",  # White background
            border_width=0
        )
        self.controls_container.pack(pady=10, padx=20, anchor="center")

        # Add month selection controls
        self.year_label = ctk.CTkLabel(
            self.controls_container,
            text="Year :",
            font=("Segoe UI", 14)
        )
        self.year_label.grid(row=0, column=0, padx=(0,5), pady=5)

        # Year Dropdown
        self.year_var = ctk.StringVar(value=str(self.data["Year"].iloc[0]))
        year_values = [str(year) for year in self.data["Date"].dt.year.unique()]
        self.year_dropdown = ctk.CTkOptionMenu(
            self.controls_container,
            values=year_values,
            variable=self.year_var,
            width=100,
            height=30,
            font=("Segoe UI", 14),
            fg_color="#1f6aa5",

        )
        self.year_dropdown.grid(row=0, column=1, padx=5, pady=5)

        self.month_label = ctk.CTkLabel(
            self.controls_container,
            text="Month :",
            font=("Segoe UI", 14)
        )
        self.month_label.grid(row=0, column=2, padx=(10,5), pady=5)

        # Month Dropdown
        self.month_var = ctk.StringVar(value=self.data["Month"].iloc[0])
        month_values = self.data["Month"].dropna().unique().tolist()
        self.month_dropdown = ctk.CTkOptionMenu(
            self.controls_container,
            values=month_values,
            variable=self.month_var,
            width=100,
            height=30,
            font=("Segoe UI", 14),
            fg_color = "#1f6aa5",
        )
        self.month_dropdown.grid(row=0, column=3, padx=5, pady=5)

        # Add extreme values toggle switch
        self.extreme_values_label = ctk.CTkLabel(
            self.controls_container,
            text="Use Extreme Values:",
            font=("Segoe UI", 14)
        )
        self.extreme_values_label.grid(row=0, column=4, padx=(20,5), pady=5)
        
        self.extreme_values_switch = ctk.CTkSwitch(
            self.controls_container,
            text="",
            variable=self.use_extreme_values,
            onvalue=True,
            offvalue=False,
            width=40,
            height=20,
            switch_height=16,
            switch_width=36,
            fg_color="#FF5555",
            progress_color="#1f6aa5"
        )
        self.extreme_values_switch.grid(row=0, column=5, padx=5, pady=5)

        # Add buttons to the same container
        self.combined_map_button = ctk.CTkButton(
            self.controls_container,
            text="Combined Map",
            width=120,
            height=30,
            fg_color="#1f6aa5",
            hover_color="#18558a",
            command=self.show_combined_map
        )
        self.combined_map_button.grid(row=1, column=0, padx=10, pady=10)

        self.preview_button = ctk.CTkButton(
            self.controls_container,
            text="Preview Data",
            width=150,
            height=30,
            fg_color="#1f6aa5",
            hover_color="#18558a",
            command=self.show_chlorophyll_preview
        )
        self.preview_button.grid(row=1, column=1, padx=10, pady=10)

        self.run_button = ctk.CTkButton(
            self.controls_container,
            text="Run Model",
            width=150,
            height=30,
            fg_color="#1f6aa5",
            hover_color="#18558a",
            command=self.run_chlorophyll_model
        )
        self.run_button.grid(row=1, column=2, padx=10, pady=10)

        # Set up the trace for dropdown changes
        self.year_var.trace_add("write", self.update_data_selection)
        self.month_var.trace_add("write", self.update_data_selection)

        # Change control frame and content frame background to white
        self.control_frame.configure(fg_color="#FFFFFF")
        self.content_frame = ctk.CTkFrame(self.control_frame, fg_color="#FFFFFF", )
        self.content_frame.pack(fill="both", expand=True)

    def update_data_selection(self, *args):
        """Update filtered data when year or month selection changes"""
        try:
            selected_year = int(self.year_var.get())
            selected_month = self.month_var.get()
            
            print(f"Selection changed to {selected_month} {selected_year}")
            
            # Update UI for the current selection
            # If there's a plot already displayed, update it
            if len(self.content_frame.winfo_children()) > 0:
                self.show_chlorophyll_preview()
        except Exception as e:
            print(f"Error updating selection: {e}")

    def load_station_coordinates(self):
        """Load station coordinates from GeoJSON file"""
        try:
            # Check if file exists
            if not os.path.exists(self.geojson_path):
                print(f"Error: Station GeoJSON file not found: {self.geojson_path}")
                self.station_coords = []
                return
                
            # Read GeoJSON using geopandas
            gdf = gpd.read_file(self.geojson_path)
            
            # Extract coordinates from each station
            self.station_coords = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                station_id = str(row["id"])
                
                # Extract point coordinates
                if geom.geom_type == 'MultiPoint':
                    point = geom.geoms[0]
                elif geom.geom_type == 'Point':
                    point = geom
                else:
                    print(f"Skipping non-point geometry: {geom.geom_type}")
                    continue

                # Store station data with coordinates
                self.station_coords.append({
                    "id": station_id,
                    "lat": point.y,
                    "lon": point.x,
                    "name": row.get("name", f"Station {station_id}")
                })
                
            print(f"Loaded {len(self.station_coords)} station coordinates")
            
        except Exception as e:
            print(f"Error loading GeoJSON file {self.geojson_path}: {e}")
            import traceback
            traceback.print_exc()
            self.station_coords = []

    def show_combined_map(self):
        """Show a combined map with station heatmap and direction data"""
        try:
            # Create directory for output if it doesn't exist
            os.makedirs("combined_maps", exist_ok=True)
            
            # Path to lake boundary file
            lake_boundary_path = "heatmapper/laguna_lakeFinal.geojson"
            
            # Verify file paths exist
            if not os.path.exists(self.geojson_path):
                print(f"Error: Station GeoJSON file not found: {self.geojson_path}")
                return
                
            if not os.path.exists(lake_boundary_path):
                print(f"Warning: Lake boundary file not found: {lake_boundary_path}")
            
            # Setup the Open-Meteo API client with cache and retry
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            # Dictionary to store direction data by station ID
            wind_data_d = {}
            wind_data_s = {}
            
            # Collect wind direction data for each station
            print("Collecting wind direction data...")
            for station in self.station_coords:
                station_id = station['id']
                
                # Set parameters for API request
                params = {
                    "latitude": station["lat"],
                    "longitude": station["lon"],
                    "daily": ["wind_speed_10m_max", "wind_direction_10m_dominant"],
                    "timezone": "Asia/Singapore",
                    "forecast_days": 16
                }
                
                try:

                    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
                    response = responses[0]
                    
                    # Process daily data
                    daily = response.Daily()
                    wind_direction = int(daily.Variables(1).ValuesAsNumpy()[0])
                    
                    # Get wind speed (keep it in km/h)
                    wind_speed = float(daily.Variables(0).ValuesAsNumpy()[0])
                    
                    # Verify that the unit is km/h
                    daily_variables = daily.Variables(1)  # Wind speed variable
                    variable_unit = daily_variables.Unit()
                    
                    print(f"Wind speed unit from API: {variable_unit}")
                    
                    # In case the API returns a different unit, handle conversion
                    if variable_unit == "m/s":
                        # Convert from m/s to km/h
                        wind_speed = wind_speed * 3.6
                        print(f"Converted wind speed from m/s to km/h: {wind_speed} km/h")
                    
                    # Additional safety cap - cap at 150 km/h as reasonable max
                    if wind_speed > 150:
                        print(f"Warning: Capping extremely high wind speed from {wind_speed} to 150 km/h")
                        wind_speed = 150.0
                    
                    # Store wind direction and speed in separate dictionaries
                    wind_data_d[station_id] = wind_direction
                    wind_data_s[station_id] = wind_speed
                    
                    # Debug output
                    print(f"Station {station_id}: Wind Direction={wind_direction}°, Wind Speed={wind_speed} km/h")
                    
                except Exception as e:
                    print(f"Error getting wind data for station {station_id}: {e}")
            
            
            
            # Get selected month and year
            selected_month = self.month_var.get()
            selected_year = int(self.year_var.get())

            # Improved filtering with diagnostics
            print(f"Filtering data for {selected_month} {selected_year}")
            print(f"Data shape before filtering: {self.data.shape}")
            
            # Filter by year correctly (making sure types match)
            filtered_year = self.data[self.data["Year"] == selected_year]
            print(f"After year filtering: {filtered_year.shape} rows remain")
            
            # Filter by month (case-insensitive matching for robustness)
            filtered_month = filtered_year[filtered_year["Month"].str.lower() == selected_month.lower()]
            print(f"After month filtering: {filtered_month.shape} rows remain")
            
            # Don't drop all NaN rows, just those with NaN in critical columns
            filtered_values = filtered_month.copy()
            
            # Debug: print columns and first few rows
            print(f"Available columns: {filtered_values.columns.tolist()}")
            if not filtered_values.empty:
                print("First few rows of filtered data:")
                print(filtered_values.head().to_string())
                
                # Check for chlorophyll data specifically
                if "Chlorophyll-a (ug/L)" in filtered_values.columns:
                    chloro_count = filtered_values["Chlorophyll-a (ug/L)"].notna().sum()
                    print(f"Rows with valid chlorophyll data: {chloro_count}")
            
            # Create combined map with both station markers and wind direction
            selected_month = self.month_var.get()
            selected_year = int(self.year_var.get())

            # Create combined map with both station markers and wind direction
            heatmap = HeatmapByParameter()
            output_path = heatmap.create_combined_map(
                self.geojson_path, 
                filtered_values, 
                lake_boundary_path, 
                wind_data_d, 
                wind_data_s,
                selected_year,
                selected_month
)
            
            if output_path and os.path.exists(output_path):
                # Open in webview
                webview.create_window(
                    "Laguna Lake Combined Map",
                    output_path,
                    width=900,
                    height=700
                )
                webview.start(debug=False)
            else:
                print("Failed to generate combined map or output file not found")
                
        except Exception as e:
            print(f"Error showing combined map: {e}")
            import traceback
            traceback.print_exc()

    def generate_geoheatmap(self):
        """Generate Folium heatmap based on station data"""
        try:
            # Verify stations are loaded
            if not self.station_coords:
                print("No station coordinates loaded. Attempting to reload...")
                self.load_station_coordinates()
                if not self.station_coords:
                    print("Still no station coordinates available. Cannot proceed.")
                    return
            
            # Use loaded station coordinates
            heat_data = [[station["lat"], station["lon"]] for station in self.station_coords]
            
            # Create base map centered on Laguna Lake
            m = folium.Map(
                location=[14.35, 121.2],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Add heatmap layer
            plugins.HeatMap(heat_data).add_to(m)
            
            # Add station markers
            for station in self.station_coords:
                folium.CircleMarker(
                    location=[station["lat"], station["lon"]],
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.7,
                    popup=f"Station: {station.get('name', station['id'])}"
                ).add_to(m)
            
            # Create directory if it doesn't exist
            os.makedirs("heatmaps", exist_ok=True)
            
            # Save map
            heatmap_path = "heatmaps/geoheatmap.html"
            m.save(heatmap_path)
            
            # Display in webview
            if os.path.exists(heatmap_path):
                webview.create_window(
                    "Laguna Lake Heatmap",
                    heatmap_path,
                    width=900,
                    height=700
                )
                webview.start(debug=False)
            else:
                print(f"Generated map not found at {heatmap_path}")
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()

    def show_chlorophyll_preview(self):
        """Display a bar graph showing chlorophyll values for each station based on selected month and year"""
        try:
            # Get selected year and month
            selected_year = int(self.year_var.get())
            selected_month = self.month_var.get()
            
            print(f"Generating chlorophyll preview for {selected_month} {selected_year}")
            
            # Filter data by selected year and month
            filtered_data = self.data[self.data["Year"] == selected_year]
            filtered_data = filtered_data[filtered_data["Month"] == selected_month]
            
            # Check if we have data for the selected period
            if filtered_data.empty:
                print(f"No data available for {selected_month} {selected_year}")
                self.show_no_data_message()
                return
                
            # Print debug info about the filtered data
            print(f"Filtered data shape: {filtered_data.shape}")
            print(f"Columns available: {filtered_data.columns.tolist()}")
            
            # Clear any existing content in the content frame
            for widget in self.content_frame.winfo_children():
                widget.destroy()
                
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract stations and their corresponding chlorophyll values
            stations = []
            chlorophyll_values = []
            station_names = []
            
            # Define a mapping of station IDs to more readable names
            station_name_map = {
                "Station_1_CWB": "Central West Bay",
                "Station_2_EastB": "East Bay",
                "Station_4_CentralB": "Central Bay",
                "Station_5_NorthernWestBay": "North West Bay",
                "Station_8_SouthB": "South Bay",
                "Station_15_SanPedro": "San Pedro",
                "Station_16_Sta. Rosa": "Sta. Rosa",
                "Station_17_Sanctuary": "Sanctuary",
                "Station_18_Pagsanjan": "Pagsanjan"
            }
            
            # Get unique stations in the filtered data
            unique_stations = filtered_data["Station"].unique()
            
            for station in unique_stations:
                # Get the station's data
                station_data = filtered_data[filtered_data["Station"] == station]
                
                # Check if we have chlorophyll data for this station
                if "Predicted_Chlorophyll" in station_data.columns and not station_data["Predicted_Chlorophyll"].isnull().all():
                    # Calculate average chlorophyll for the station (in case of multiple records)
                    avg_chlorophyll = station_data["Predicted_Chlorophyll"].mean()
                    
                    # Add to our lists
                    stations.append(station)
                    chlorophyll_values.append(avg_chlorophyll)
                    
                    # Use the mapping if available, otherwise use station ID
                    display_name = station_name_map.get(station, station)
                    station_names.append(display_name)
            
            # If we found no data, show message and return
            if not stations:
                print(f"No chlorophyll data available for {selected_month} {selected_year}")
                self.show_no_data_message()
                return
                
            print(f"Found data for {len(stations)} stations")
                
            # Set colors based on new chlorophyll value ranges
            colors = []
            for value in chlorophyll_values:
                if value > 150:  # Very High
                    colors.append("#006400")  # Dark Green
                elif value > 100:  # High
                    colors.append("#228B22")  # Forest Green
                elif value > 50:  # Medium
                    colors.append("#3CB371")  # Medium Sea Green
                elif value > 25:  # Low
                    colors.append("#90EE90")  # Light Green
                elif value >= 0:  # Very Low
                    colors.append("#98FB98")  # Pale Green
                else:  # No data or negative values
                    colors.append("#D5DBDB")  # Gray
            
            # Create bar plot
            bars = ax.bar(station_names, chlorophyll_values, color=colors)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, chlorophyll_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}',
                        ha='center', va='bottom', fontsize=9)
            
            # Set title and labels
            ax.set_title(f'Chlorophyll-a Levels by Station ({selected_month} {selected_year})')
            ax.set_xlabel('Station')
            ax.set_ylabel('Chlorophyll-a (μg/L)')
            
            # Rotate x-tick labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add grid for easier reading
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adjust layout to make room for rotated x-tick labels
            plt.tight_layout()
            
            # Create frame for the chart
            chart_frame = ctk.CTkFrame(self.content_frame)
            chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Embed the plot in the tkinter window
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Add legend explaining color coding
            legend_frame = ctk.CTkFrame(self.content_frame)
            legend_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            ctk.CTkLabel(legend_frame, text="Color Legend:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=10)
            
            # Add color boxes with labels
            legend_items = [
                {"color": "#98FB98", "label": "Very Low (< 25 μg/L)"},
                {"color": "#90EE90", "label": "Low (25-50 μg/L)"},
                {"color": "#3CB371", "label": "Medium (50-100 μg/L)"},
                {"color": "#228B22", "label": "High (100-150 μg/L)"},
                {"color": "#006400", "label": "Very High (> 150 μg/L)"},
                {"color": "#D5DBDB", "label": "No data"}
            ]
            
            for item in legend_items:
                box = ctk.CTkFrame(legend_frame, width=15, height=15, fg_color=item["color"])
                box.pack(side="left", padx=(10, 5))
                ctk.CTkLabel(legend_frame, text=item["label"]).pack(side="left", padx=(0, 10))
            
        except Exception as e:
            print(f"Error displaying chlorophyll preview: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message(str(e))

    def run_chlorophyll_model(self):
        """Run the chlorophyll forecasting model with station ordering"""
        try:
            # Show loading indicator
            for widget in self.content_frame.winfo_children():
                widget.destroy()
                
            loading_label = ctk.CTkLabel(
                self.content_frame,
                text="Running chlorophyll forecasting model...\nThis may take a few minutes.",
                font=("Arial", 16)
            )
            loading_label.pack(pady=50)
            self.update()
            
            # Define the standard station order (once)
            station_order = [
                "Station_1_CWB",
                "Station_2_EastB",
                "Station_4_CentralB",
                "Station_5_NorthernWestBay",
                "Station_8_SouthB",
                "Station_15_SanPedro",
                "Station_16_Sta. Rosa",
                "Station_17_Sanctuary",
                "Station_18_Pagsanjan"
            ]
            
            # Import chlorophyll_forecaster functions
            loading_label.configure(text="Importing chlorophyll forecaster...")
            self.update()
            
            from chlorophyll_forecaster import (
                load_existing_model_and_features,
                load_full_dataset,
                retrain_model_on_full_data,
                generate_future_dates,
                prepare_future_features,
                predict_future_values,
                plot_and_save_results,
                main
            )
            
            # Create a patched version of plot_and_save_results that sorts by station
            def patched_plot_and_save_results(df, future_pred_df, target='Chlorophyll-a (ug/L)'):
                """Patched version of plot_and_save_results that sorts by station"""
                import pandas as pd
                
                # Create a categorical type for station ordering
                station_cols = [col for col in future_pred_df.columns if 'station_' in col]
                if station_cols:
                    # Extract station names
                    station_names = []
                    for i, row in future_pred_df.iterrows():
                        station_name = "Unknown"
                        for col in station_cols:
                            if row[col] == 1:
                                parts = col.split('_')
                                if len(parts) > 2:
                                    station_name = '_'.join(parts[2:])
                                    break
                        future_pred_df.loc[i, 'Station'] = station_name
                    
                    # Sort by station using our defined order
                    if 'Station' in future_pred_df.columns:
                        # Create a categorical type with our custom order
                        future_pred_df["Station"] = pd.Categorical(
                            future_pred_df["Station"],
                            categories=station_order,
                            ordered=True
                        )
                        
                        # Sort by Station and Date
                        if "Date" in future_pred_df.columns:
                            future_pred_df = future_pred_df.sort_values(["Station", "Date"])
                        else:
                            future_pred_df = future_pred_df.sort_values("Station")
                
                # Call the original function with sorted data
                return plot_and_save_results(df, future_pred_df, target)
            
            # Load model and features
            loading_label.configure(text="Loading model and features...")
            self.update()
            model, selected_features, metadata = load_existing_model_and_features()
            
            # Define features and target
            features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
                    'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 
                    'Temperature', 'Phytoplankton']
            target = 'Chlorophyll-a (ug/L)'
            
            # Load dataset
            loading_label.configure(text="Loading and preprocessing data...")
            self.update()
            file_path = 'CSV/merged_stations.xlsx'
            df, last_date = load_full_dataset(file_path, features, target)
            
            # Retrain model
            loading_label.configure(text="Retraining model on full dataset...")
            self.update()
            retrained_model, full_r2, full_rmse = retrain_model_on_full_data(
                df, features, target, selected_features, model
            )
            
            # Get station info
            station_cols = [col for col in df.columns if 'station_' in col]
            unique_station_configs = df[station_cols].drop_duplicates().reset_index(drop=True)
            num_stations = len(unique_station_configs)
            
            # Generate future dates
            loading_label.configure(text="Generating future dates and features...")
            self.update()
            future_dates_df = generate_future_dates(last_date, months_ahead=3, num_stations=num_stations)
            
            # Prepare features
            future_df = prepare_future_features(
                df, future_dates_df, features, target, selected_features
            )
            
            # Make predictions
            loading_label.configure(text="Making predictions...")
            self.update()
            future_pred_df = predict_future_values(
                retrained_model, 
                future_df, 
                selected_features, 
                use_log=True,
                use_extreme_values=self.use_extreme_values.get()  # Pass the extreme values flag
            )
            
            # Add station names column
            loading_label.configure(text="Processing station information...")
            self.update()
            
            station_names = []
            for i, row in future_pred_df.iterrows():
                station_name = "Unknown"
                for col in station_cols:
                    if row[col] == 1:
                        parts = col.split('_')
                        if len(parts) > 2:
                            station_name = '_'.join(parts[2:])
                            break
                future_pred_df.loc[i, 'Station'] = station_name
            
            # Sort by station and date BEFORE plotting and saving
            if 'Station' in future_pred_df.columns:
                # Create a categorical type with our custom order
                future_pred_df["Station"] = pd.Categorical(
                    future_pred_df["Station"],
                    categories=station_order,
                    ordered=True
                )
                
                # Sort by Station and Date
            future_pred_df = future_pred_df.sort_values(["Station", "Date"])
            
            # Plot and save results (already sorted)
            loading_label.configure(text="Generating plots and saving results...")
            self.update()
            
            # Display whether extreme values are being used
            extreme_values_status = "enabled" if self.use_extreme_values.get() else "disabled"
            loading_label.configure(
                text=f"Generating plots and saving results...\nExtreme values are {extreme_values_status}"
            )
            self.update()
            
            summary = plot_and_save_results(df, future_pred_df, target)
            
            
            # Show completion message with extreme values status
            loading_label.configure(
                text=f"Model execution complete!\nResults saved with Station 1 first.\n"
                    f"Extreme values were {extreme_values_status}.\n"
                    "Please click 'Refresh Data' to view updated predictions."
            )
            
            # Add refresh button
            refresh_button = ctk.CTkButton(
                self.content_frame,
                text="Refresh Data",
                width=150,
                height=30,
                fg_color="#1f6aa5",
                hover_color="#18558a",
                command=self.refresh_data
            )
            refresh_button.pack(pady=20)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Show error message
            for widget in self.content_frame.winfo_children():
                widget.destroy()
                
            error_label = ctk.CTkLabel(
                self.content_frame,
                text=f"Error running model:\n{str(e)}",
                font=("Arial", 16),
                text_color="#FF5733"
            )
            error_label.pack(pady=50)

    def refresh_data(self):
        """Reload data after model execution"""
        try:
            # Reload the data from CSV
            self.data = pd.read_csv("CSV/chlorophyll_predictions_by_station.csv")
            self.data["Date"] = pd.to_datetime(self.data["Date"])
            self.data["Year"] = self.data["Date"].dt.year
            self.data["Month"] = self.data["Date"].dt.month_name()
            
            # Update dropdowns with new data
            year_values = [str(year) for year in self.data["Date"].dt.year.unique()]
            self.year_dropdown.configure(values=year_values)
            self.year_var.set(year_values[0] if year_values else "")
            
            month_values = self.data["Month"].dropna().unique().tolist()
            self.month_dropdown.configure(values=month_values)
            self.month_var.set(month_values[0] if month_values else "")
            
            # Show preview with new data
            self.show_chlorophyll_preview()
            
        except Exception as e:
            print(f"Error refreshing data: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error message
            for widget in self.content_frame.winfo_children():
                widget.destroy()
                
            error_label = ctk.CTkLabel(
                self.content_frame,
                text=f"Error refreshing data:\n{str(e)}",
                font=("Segoe UI", 16),
                text_color="#FF5733"
            )
            error_label.pack(pady=50)

    # Helper method for showing no data message
    def show_no_data_message(self):
        """Show a message when no data is available for the selected period"""
        # Clear any existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Create a message frame
        message_frame = ctk.CTkFrame(self.content_frame)
        message_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Add message
        ctk.CTkLabel(
            message_frame,
            text="No chlorophyll data available for the selected month and year.",
            font=("Segoe UI", 16)
        ).pack(pady=50)
        
        # Add suggestion
        ctk.CTkLabel(
            message_frame,
            text="Please select a different month or year.",
            font=("Segoe UI", 14)
        ).pack(pady=10)

    # Helper method for showing error message
    def show_error_message(self, error_text):
        """Show an error message"""
        # Clear any existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Create a message frame
        message_frame = ctk.CTkFrame(self.content_frame)
        message_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Add error message
        ctk.CTkLabel(
            message_frame,
            text="Error displaying chlorophyll data",
            font=("Segoe UI", 16, "bold"),
            text_color="#FF5733"
        ).pack(pady=(50, 10))
        
        # Add error details
        ctk.CTkLabel(
            message_frame,
            text=error_text,
            font=("Segoe UI", 12),
            wraplength=500
        ).pack(pady=10)

    def show(self):
        """Show the prediction page"""
        self.grid(row=0, column=0, sticky="nsew")