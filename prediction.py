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
        super().__init__(parent)
        
        # File paths and coordinates
        self.geojson_path = "heatmapper/stations_final.geojson"
        self.laguna_coords = {"lat": 14.35, "lon": 121.2}
        self.station_coords = []  # Will store coordinates from GeoJSON
        
        self.data = pd.read_excel("CSV/merged_stations.xlsx")
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
            fg_color="#E8E9F3",
            border_width=0  # Remove shadow/border
        )
        self.columnconfigure(0, weight=1)  # Allow the control frame to expand
        self.rowconfigure(1, weight=1)
        self.control_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        self.control_frame.grid_rowconfigure(0, weight=1)
        
        # Add month selection dropdown
        self.date_frame = ctk.CTkFrame(
            self.control_frame, 
            fg_color="#E8E9F3",
            border_width=0  # Remove shadow/border
        )
        self.date_frame.pack(pady=(10,0))

        self.year_label = ctk.CTkLabel(
            self.date_frame,
            text="Year :",
            font=("Arial", 14)
        )
        self.year_label.pack(side="left", padx=(0,10))
        
        # Year Dropdown
        self.year_var = ctk.StringVar(value=str(self.data["Year"].iloc[0]))  # Default to the first year
        year_values = [str(year) for year in self.data["Date"].dt.year.unique()]
        self.year_dropdown = ctk.CTkOptionMenu(
            self.date_frame,
            values=year_values,
            variable=self.year_var,
            width=100,
            height=30,
            font=("Arial", 14)
        )
        self.year_dropdown.pack(side="left", padx=5)

        self.month_label = ctk.CTkLabel(
            self.date_frame,
            text="Month :",
            font=("Arial", 14)
        )
        self.month_label.pack(side="left", padx=(0,10))
        
        # Month Dropdown
        self.month_var = ctk.StringVar(value=self.data["Month"].iloc[0])  # Default to the first month
        month_values = self.data["Month"].dropna().unique().tolist()
        self.month_dropdown = ctk.CTkOptionMenu(
            self.date_frame,
            values=month_values,
            variable=self.month_var,
            width=100,
            height=30,
            font=("Arial", 14)
        )
        self.month_dropdown.pack(side="left", padx=5)
        
        self.year_var.trace_add("write", self.update_data_selection)
        self.month_var.trace_add("write", self.update_data_selection)

        # Add generate heatmap controls
        self.generate_label = ctk.CTkLabel(
            self.control_frame,
            text="Generate heatmap",
            font=("Arial", 14)
        )
        self.generate_label.pack(pady=5)
        
        # Create buttons frame for organizing buttons horizontally
        self.buttons_frame = ctk.CTkFrame(
            self.control_frame, 
            fg_color="#E8E9F3",
            border_width=0
        )
        self.buttons_frame.pack(fill="x", pady=5)
        
        # Add generate heatmap button to buttons frame
        self.generate_button = ctk.CTkButton(
            self.buttons_frame,
            text="Station Map",
            width=100,
            height=30,
            command=self.generate_heatmap
        )
        self.generate_button.pack(side="left", padx=10)
        
        # Add wind direction button to buttons frame
        self.weather_button = ctk.CTkButton(
            self.buttons_frame,
            text="Wind Direction",
            width=120,
            height=30,
            command=self.show_wind_direction
        )
        self.weather_button.pack(side="left", padx=10)
        
        # Add combined map button to buttons frame
        self.combined_map_button = ctk.CTkButton(
            self.buttons_frame,
            text="Combined Map",
            width=120,
            height=30,
            command=self.show_combined_map
        )
        self.combined_map_button.pack(side="left", padx=10)

        self.preview_button = ctk.CTkButton(
            self.buttons_frame,
            text="Preview Data",
            width=150,
            height=30,
            command=self.show_chlorophyll_preview
        )

        self.preview_button.pack(side="left", padx=10)
        # Create content frame for the main area
        self.content_frame = ctk.CTkFrame(self.control_frame, fg_color="#F5F5F5")
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

    def generate_heatmap(self):
        """Generate heatmap based on selected months"""
        try:
            selected_month = self.month_var.get()
            print(f"Selected month: {selected_month}")
            
            # Check if geojson file exists
            if not os.path.exists(self.geojson_path):
                print(f"Error: Station GeoJSON file not found: {self.geojson_path}")
                return
            
            # Create and show map regardless of month selection
            heatmap = HeatmapByParameter()
            output_path = heatmap.create_pulse_map(self.geojson_path)
            
            if output_path and os.path.exists(output_path):
                webview.create_window(
                    "Laguna Lake Monitoring Stations",
                    output_path,
                    width=900,
                    height=700
                )
                webview.start(debug=False)
            else:
                print("Failed to generate heatmap or output file not found")
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()

    def show_wind_direction(self):
        """Show wind direction data from Open-Meteo API for each station"""
        try:
            # Create directory for output if it doesn't exist
            os.makedirs("weather_data", exist_ok=True)
            
            # Check if geojson file exists
            if not os.path.exists(self.geojson_path):
                print(f"Error: Station GeoJSON file not found: {self.geojson_path}")
                return
                
            # Check if we have stations loaded
            if not self.station_coords:
                print("No station coordinates loaded. Attempting to reload...")
                self.load_station_coordinates()
                if not self.station_coords:
                    print("Still no station coordinates available. Cannot proceed.")
                    return
            
# Setup the Open-Meteo API client with cache and retry
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            # Create a list to hold station data for table
            station_data = []
            
            # Dictionary to store wind direction data by station ID
            wind_data = {}
            
            # Loop through each station to get wind direction
            for station in self.station_coords:
                # Set parameters for API request
                params = {
                    "latitude": station["lat"],
                    "longitude": station["lon"],
                    "daily": "wind_direction_10m_dominant",
                    "timezone": "Asia/Singapore",
                    "forecast_days": 1
                }
                
                try:
                    # Make API request
                    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
                    response = responses[0]
                    
                    # Process daily data
                    daily = response.Daily()
                    wind_direction = int(daily.Variables(0).ValuesAsNumpy()[0])
                    
                    # Store wind direction data
                    wind_data[station["id"]] = wind_direction
                    
                    # Add to station data for table
                    station_data.append({
                        "Station": station.get("name", station["id"]),
                        "Latitude": station["lat"],
                        "Longitude": station["lon"],
                        "Wind Direction": f"{wind_direction}°"
                    })
                except Exception as e:
                    print(f"Error getting wind data for station {station['id']}: {e}")
            
            # Create a HeatmapByParameter instance
            heatmap = HeatmapByParameter()
            
            # Create wind direction map
            # First load stations into the heatmap instance
            heatmap.stations = self.station_coords
            
            # Check if any wind data was collected
            if not wind_data:
                print("No wind direction data collected. Cannot create wind direction map.")
                return
            
            # Create a map centered on Laguna Lake
            m = folium.Map(
                location=[14.35, 121.2],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Add wind direction arrows
            wind_group = folium.FeatureGroup(name="Wind Direction")
            
            for station in self.station_coords:
                station_id = station['id']
                if station_id in wind_data:
                    wind_direction = wind_data[station_id]
                    
                    # Calculate arrow endpoint based on wind direction
                    arrow_length = 0.01  # Adjust for visibility
                    
                    # Convert wind direction from meteorological to cartesian angle
                    wind_rad = np.radians((270 - wind_direction) % 360)
                    
                    # Calculate arrow endpoint
                    end_lat = station["lat"] + arrow_length * np.sin(wind_rad)
                    end_lon = station["lon"] + arrow_length * np.cos(wind_rad)
                    
                    # Add wind direction line
                    folium.PolyLine(
                        locations=[[station["lat"], station["lon"]], [end_lat, end_lon]],
                        color='red',
                        weight=3,
                        opacity=0.8,
                        popup=f"<b>{station.get('name', station['id'])}</b><br>Wind Direction: {wind_direction}°"
                    ).add_to(wind_group)
            
            wind_group.add_to(m)
            
            # Add table with station data
            station_df = pd.DataFrame(station_data)
            html_table = station_df.to_html(classes="table table-striped table-hover", index=False)
            
            # Add the table to the map
            table_html = f'''
            <div id="table-container" style="position: fixed; bottom: 10px; right: 10px; 
                 background-color: white; padding: 10px; border-radius: 5px; 
                 max-height: 300px; overflow-y: auto; opacity: 0.9; z-index: 1000;">
                <h4>Station Wind Direction Data</h4>
                {html_table}
                <p>Data from Open-Meteo API - {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(table_html))
            
            # Add toggle button for the table
            table_control_js = '''
            <script>
            var tableVisible = true;
            function toggleTable() {
                var tableContainer = document.getElementById('table-container');
                if (tableVisible) {
                    tableContainer.style.display = 'none';
                    tableVisible = false;
                    document.getElementById('toggle-button').innerText = 'Show Table';
                } else {
                    tableContainer.style.display = 'block';
                    tableVisible = true;
                    document.getElementById('toggle-button').innerText = 'Hide Table';
                }
            }
            </script>
            
            <div style="position: fixed; top: 10px; right: 10px; z-index: 1000;">
                <button id="toggle-button" onclick="toggleTable()" 
                        style="background-color: white; border: 2px solid #ccc; 
                               border-radius: 4px; padding: 5px 10px; cursor: pointer;">
                    Hide Table
                </button>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(table_control_js))
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color: white; padding: 10px; opacity: 0.8;
                        border-radius: 5px;">
                <p><b>Map Legend</b></p>
                <p><span style="color:red;">→</span> Wind Direction</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add title
            title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50%; transform: translateX(-50%);
                        z-index:9999; font-size:18px; font-weight: bold;
                        background-color: white; padding: 10px; 
                        border-radius: 5px; opacity: 0.9;">
                Laguna Lake Wind Direction
                <div style="font-size: 12px; font-weight: normal; text-align: center;">
                    {len(self.station_coords)} stations • Map generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Create output directory if it doesn't exist
            os.makedirs("weather_data", exist_ok=True)
            
            # Save map
            output_path = "weather_data/wind_direction_map.html"
            m.save(output_path)
            
            # Display in webview
            if os.path.exists(output_path):
                webview.create_window(
                    "Laguna Lake Wind Direction",
                    output_path,
                    width=900,
                    height=700
                )
                webview.start(debug=False)
            else:
                print(f"Generated map not found at {output_path}")
                
        except Exception as e:
            print(f"Error showing wind direction data: {e}")
            import traceback
            traceback.print_exc()

    def show_combined_map(self):
        """Show a combined map with station heatmap and wind direction data"""
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
            
            # Dictionary to store wind direction data by station ID
            wind_data = {}
            
            # Collect wind direction data for each station
            print("Collecting wind direction data...")
            for station in self.station_coords:
                station_id = station['id']
                
                # Set parameters for API request
                params = {
                    "latitude": station["lat"],
                    "longitude": station["lon"],
                    "daily": "wind_direction_10m_dominant",
                    "timezone": "Asia/Singapore",
                    "forecast_days": 1
                }
                
                try:
                    # Make API request
                    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
                    response = responses[0]
                    
                    # Process daily data
                    daily = response.Daily()
                    wind_direction = int(daily.Variables(0).ValuesAsNumpy()[0])
                    
                    # Store wind direction data
                    wind_data[station_id] = wind_direction
                    
                except Exception as e:
                    print(f"Error getting wind data for station {station_id}: {e}")
            
            print(f"Collected wind data for {len(wind_data)} stations")
            
            selected_month = self.month_var.get()
            selected_year = self.year_var.get()

            filtered_year = self.data[self.data["Year"] == selected_year]
            filtered_month = filtered_year[filtered_year["Month"] == selected_month]
            filtered_values = filtered_month.dropna()
            # Create combined map with both station markers and wind direction
            heatmap = HeatmapByParameter()
            output_path = heatmap.create_combined_map(self.geojson_path, filtered_values, lake_boundary_path, wind_data)
            
            if output_path and os.path.exists(output_path):
                # Open in browser to avoid webview issues
                # import webbrowser
                # webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
                # Alternative: Use webview if preferred
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
                if "Chlorophyll-a (ug/L)" in station_data.columns and not station_data["Chlorophyll-a (ug/L)"].isnull().all():
                    # Calculate average chlorophyll for the station (in case of multiple records)
                    avg_chlorophyll = station_data["Chlorophyll-a (ug/L)"].mean()
                    
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
                
            # Set colors based on chlorophyll values (red for high, green for medium, blue for low)
            colors = []
            for value in chlorophyll_values:
                if value > 20:  # High chlorophyll
                    colors.append("#FF5733")  # Red
                elif value > 10:  # Medium chlorophyll
                    colors.append("#33FF57")  # Green
                else:  # Low chlorophyll
                    colors.append("#3357FF")  # Blue
            
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
            
            ctk.CTkLabel(legend_frame, text="Color Legend:", font=("Arial", 12, "bold")).pack(side="left", padx=10)
            
            # Red legend
            red_box = ctk.CTkFrame(legend_frame, width=15, height=15, fg_color="#FF5733")
            red_box.pack(side="left", padx=(10, 5))
            ctk.CTkLabel(legend_frame, text="High (>20 μg/L)").pack(side="left", padx=(0, 10))
            
            # Green legend
            green_box = ctk.CTkFrame(legend_frame, width=15, height=15, fg_color="#33FF57")
            green_box.pack(side="left", padx=(10, 5))
            ctk.CTkLabel(legend_frame, text="Medium (10-20 μg/L)").pack(side="left", padx=(0, 10))
            
            # Blue legend
            blue_box = ctk.CTkFrame(legend_frame, width=15, height=15, fg_color="#3357FF")
            blue_box.pack(side="left", padx=(10, 5))
            ctk.CTkLabel(legend_frame, text="Low (<10 μg/L)").pack(side="left", padx=(0, 10))
            
        except Exception as e:
            print(f"Error displaying chlorophyll preview: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message(str(e))

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
            font=("Arial", 16)
        ).pack(pady=50)
        
        # Add suggestion
        ctk.CTkLabel(
            message_frame,
            text="Please select a different month or year.",
            font=("Arial", 14)
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
            font=("Arial", 16, "bold"),
            text_color="#FF5733"
        ).pack(pady=(50, 10))
        
        # Add error details
        ctk.CTkLabel(
            message_frame,
            text=error_text,
            font=("Arial", 12),
            wraplength=500
        ).pack(pady=10)

    def show(self):
        """Show the prediction page"""
        self.grid(row=0, column=0, sticky="nsew")