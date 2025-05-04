import customtkinter as ctk
import webview
import requests
import json
import os
import numpy as np
import pandas as pd
import folium
from folium import plugins
import geopandas as gpd
from datetime import datetime
from heatmapper.Heatmapper import HeatmapByParameter

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
        
        # Load station coordinates on initialization
        self.load_station_coordinates()
        
        # Create top control frame with fixed height
        self.control_frame = ctk.CTkFrame(
            self, 
            height=100, 
            fg_color="#E8E9F3",
            border_width=0  # Remove shadow/border
        )
        self.control_frame.pack(fill="x", padx=0, pady=0)
        self.control_frame.pack_propagate(False)
        
        # Add month selection dropdown
        self.month_frame = ctk.CTkFrame(
            self.control_frame, 
            fg_color="#E8E9F3",
            border_width=0  # Remove shadow/border
        )
        self.month_frame.pack(pady=(10,0))
        
        self.month_label = ctk.CTkLabel(
            self.month_frame,
            text="Month :",
            font=("Arial", 14)
        )
        self.month_label.pack(side="left", padx=(0,10))
        
        # Replace individual month labels with dropdown
        self.month_var = ctk.StringVar(value="1")
        self.month_dropdown = ctk.CTkOptionMenu(
            self.month_frame,
            values=["1", "2", "3"],
            variable=self.month_var,
            width=100,
            height=30,
            font=("Arial", 14)
        )
        self.month_dropdown.pack(side="left", padx=5)
        
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
        
        # Create content frame for the main area
        self.content_frame = ctk.CTkFrame(self, fg_color="#F5F5F5")
        self.content_frame.pack(fill="both", expand=True)

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
            
            # Create combined map with both station markers and wind direction
            heatmap = HeatmapByParameter()
            output_path = heatmap.create_combined_map(self.geojson_path, lake_boundary_path, wind_data)
            
            if output_path and os.path.exists(output_path):
                # Open in browser to avoid webview issues
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
                # Alternative: Use webview if preferred
                # webview.create_window(
                #     "Laguna Lake Combined Map",
                #     output_path,
                #     width=900,
                #     height=700
                # )
                # webview.start(debug=False)
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

    def show(self):
        """Show the prediction page"""
        self.grid(row=0, column=0, sticky="nsew")