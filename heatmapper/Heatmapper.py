import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import numpy as np
from datetime import datetime
import json

def fix_numeric_keys(data):
    """
    Recursively convert all numeric keys (float, int) to strings in dictionaries 
    to avoid Folium errors with camelize function.
    Fixes "'float'/'int' object has no attribute 'split'" errors.
    """
    if isinstance(data, dict):
        # Create a new dict to avoid modifying during iteration
        result = {}
        for key, value in data.items():
            # Convert numeric keys to strings
            if isinstance(key, (float, int)):
                key = str(key)
            # Recursively fix nested structures
            result[key] = fix_numeric_keys(value)
        return result
    elif isinstance(data, list):
        return [fix_numeric_keys(item) for item in data]
    else:
        return data
    
class HeatmapByParameter:
    def __init__(self, excel_path=None, geojson_path=None):
        """Initialize the heatmap generator"""
        self.data = None
        self.stations = None
        self.station_id_mapping = {}  # Initialize empty mapping
        
        if excel_path:
            self.data = self.load_merged_excel(excel_path)
            self.data["Date"] = self.data["Date"].to_datetime()

            self.data = self.data.dropna(subset=['Date', 'Station', 'Chlorophyll-a (ug/L)'])

            self.data["Year"] = self.data["Date"].dt.year
            self.data["Month"] = self.data["Date"].dt.month
            
        if geojson_path:
            self.stations = self.load_coordinates(geojson_path)

    def load_merged_excel(self, excel_path):
        """Load data from Excel file"""
        try:
            data = pd.read_excel(excel_path)
            print(f"Loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading Excel file {excel_path}: {e}")
            return None

    def load_coordinates(self, geojson_path):
        """Load station coordinates from GeoJSON file"""
        try:
            print(f"Opening GeoJSON file: {geojson_path}")
            if not os.path.exists(geojson_path):
                print(f"File not found: {geojson_path}")
                return []
                
            gdf = gpd.read_file(geojson_path)
            print(f"GeoJSON file loaded with {len(gdf)} features")
            
            stations = []
            for _, row in gdf.iterrows():
                try:
                    geom = row.geometry
                    station_id = str(row["id"])  # Convert to string
                    
                    # Extract point coordinates
                    if geom.geom_type == 'MultiPoint':
                        point = geom.geoms[0]
                    elif geom.geom_type == 'Point':
                        point = geom
                    else:
                        print(f"Skipping non-point geometry: {geom.geom_type}")
                        continue

                    # Simplified station data without Excel ID mapping
                    stations.append({
                        "id": station_id,
                        "lat": point.y,
                        "lon": point.x,
                        "name": row.get("name", f"Station {station_id}")
                    })
                except Exception as e:
                    print(f"Error processing a station in GeoJSON: {e}")
                    continue

            print(f"Successfully extracted {len(stations)} stations from GeoJSON file")
            return stations
            
        except Exception as e:
            print(f"Error loading GeoJSON file {geojson_path}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_no_data_map(self, output_path, parameter, year, month):
        """
        Create a map indicating no data is available
        
        Args:
            output_path (str): Path to save the HTML output
            parameter (str): Parameter name
            year (int): Year
            month (int): Month
        """
        center_lat = sum(s["lat"] for s in self.stations) / len(self.stations) if self.stations else 14.5
        center_lon = sum(s["lon"] for s in self.stations) / len(self.stations) if self.stations else 121.0
        
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=10,
            tiles='CartoDB positron'
        )
        
        # Add stations as gray markers
        for station in self.stations:
            folium.CircleMarker(
                location=[station["lat"], station["lon"]],
                radius=6,
                color='gray',
                weight=1,
                fill=True,
                fill_color='lightgray',
                fill_opacity=0.5,
                popup=f"<b>Station: {station.get('name', station['id'])}</b><br>No data available"
            ).add_to(m)
        
        # Add message about no data
        no_data_html = f'''
        <div style="position: fixed; 
                    top: 50%; left: 50%; transform: translate(-50%, -50%);
                    border:2px solid grey; z-index:9999; font-size:16px;
                    background-color: white; padding: 20px; text-align: center;
                    opacity: 0.9; border-radius: 5px;">
            <h3>No Data Available</h3>
            <p>No data is available for {parameter} in {month}/{year}.</p>
            <p>Please select a different parameter, month, or year.</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(no_data_html))
        
        m.save(output_path)
    
    def _create_error_map(self, output_path, error_message):
        """
        Create a map displaying an error message
        
        Args:
            output_path (str): Path to save the HTML output
            error_message (str): Error message to display
        """
        m = folium.Map(
            location=[14.5, 121.0],  # Default to Philippines
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Add error message
        error_html = f'''
        <div style="position: fixed; 
                    top: 50%; left: 50%; transform: translate(-50%, -50%);
                    border:2px solid red; z-index:9999; font-size:16px;
                    background-color: white; padding: 20px; text-align: center;
                    opacity: 0.9; border-radius: 5px;">
            <h3>Error Generating Map</h3>
            <p>{error_message}</p>
            <p>Please check your input parameters and try again.</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(error_html))
        
        m.save(output_path)

    def add_pulse_style(self, map_obj):
        """
        Add pulsing effect CSS and JavaScript to the map
        """
        # Add custom CSS for pulse effect
        pulse_css = """
            <style>
                .pulse {
                    display: block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: #1e88e5;
                    cursor: pointer;
                    box-shadow: 0 0 0 rgba(30, 136, 229, 0.4);
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0% {
                        box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.4);
                    }
                    70% {
                        box-shadow: 0 0 0 10px rgba(30, 136, 229, 0);
                    }
                    100% {
                        box-shadow: 0 0 0 0 rgba(30, 136, 229, 0);
                    }
                }
            </style>
        """
        
        map_obj.get_root().header.add_child(folium.Element(pulse_css))

    def get_cardinal_direction(self, degrees):
        """Convert degrees to cardinal direction name"""
        directions = [
            "N", "NNE", "NE", "ENE", 
            "E", "ESE", "SE", "SSE", 
            "S", "SSW", "SW", "WSW", 
            "W", "WNW", "NW", "NNW"
        ]
        index = round(degrees / 22.5) % 16
        return directions[index]

    def create_pulse_map(self, geojson_path):
        """Create a map with pulsing station markers and a heatmap"""
        try:
            # Ensure output directory exists
            os.makedirs("heatmapper", exist_ok=True)
            
            # Load station data
            self.stations = self.load_coordinates(geojson_path)
            
            if not self.stations:
                print("No station data found")
                return None
            
            # Create map centered on Laguna Lake
            m = folium.Map(
                location=[14.35, 121.2], 
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Add pulsing CSS style
            self.add_pulse_style(m)
            
            # List to collect coordinates for the heatmap
            heat_data = []

            # Add stations with pulse effect
            for station in self.stations:
                # Add pulsing marker for the station
                folium.Marker(
                    location=[station['lat'], station['lon']],
                    popup=f"<b>Station:</b> {station['name']}<br><b>ID:</b> {station['id']}",
                    icon=folium.DivIcon(
                        html=f'<div class="pulse"></div>'
                    )
                ).add_to(m)
                
                # Add the station's coordinates to the heatmap data
                heat_data.append([station['lat'], station['lon']])

            # Create a feature group for heatmap so it can be toggled
            heatmap_group = folium.FeatureGroup(name="Station Density")
            
            # Add HeatMap to the map with collected coordinates
            HeatMap(
                heat_data, 
                radius=20, 
                blur=15, 
                max_zoom=13,
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
            ).add_to(heatmap_group)
            
            heatmap_group.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color: white; padding: 10px; opacity: 0.8;
                        border-radius: 5px;">
                <p><b>Map Legend</b></p>
                <p><span style="color: #1e88e5;">●</span> Station Location</p>
                <p style="margin-top: 5px;"><b>Heat Map Colors:</b></p>
                <div style="width:20px; height:15px; background:blue; display:inline-block;"></div>
                <div style="width:20px; height:15px; background:lime; display:inline-block;"></div>
                <div style="width:20px; height:15px; background:red; display:inline-block;"></div>
                <p style="font-size:12px;">Low → Medium → High Density</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add title and info
            title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50%; transform: translateX(-50%);
                        z-index:9999; font-size:18px; font-weight: bold;
                        background-color: white; padding: 10px; 
                        border-radius: 5px; opacity: 0.9;">
                Laguna Lake Monitoring Stations
                <div style="font-size: 12px; font-weight: normal; text-align: center;">
                    {len(self.stations)} stations • Map generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Save map
            output_path = "heatmapper/laguna_stations_with_heatmap.html"
            m.save(output_path)
            return output_path
            
        except Exception as e:
            print(f"Error creating pulse map: {e}")
            import traceback
            traceback.print_exc()
            return None


    def create_combined_map(self, geojson_path, values, lake_boundary_path=None, wind_data_d=None, wind_data_s=None, selected_year=None, selected_month=None):
        """Create a combined map with station markers, wind direction, and heatmap constrained to lake boundary"""
        try:
            # Ensure output directory exists
            os.makedirs("heatmapper", exist_ok=True)
            os.makedirs("heatmapper/data_cache", exist_ok=True)
            
            # Load station data
            print(f"Loading coordinates from: {geojson_path}")
            self.stations = self.load_coordinates(geojson_path)
            
            if not self.stations:
                print(f"Warning: No station data found in {geojson_path}")
                return None
                
            print(f"Successfully loaded {len(self.stations)} stations")
            
            # DEBUG: Print what's in values DataFrame
            print("Values DataFrame info:")
            print(f"Columns: {values.columns.tolist() if not values.empty else 'Empty DataFrame'}")
            print(f"Number of rows: {len(values)}")
            if len(values) > 0:
                print("First few rows:")
                print(values.head().to_string())
            
            # Store permanent copy of values for persistence between sessions
            # This ensures we use the same data each time the map is opened
            if len(values) > 0 and 'Chlorophyll-a (ug/L)' in values.columns:
                # Create a directory for persistent data if it doesn't exist
                os.makedirs("heatmapper/data_cache", exist_ok=True)
                
                # Save the filtered values to a CSV file with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                cache_path = f"heatmapper/data_cache/chlorophyll_values_{timestamp}.csv"
                values.to_csv(cache_path, index=False)
                
                # Save the reference to the latest data file
                with open("heatmapper/data_cache/latest_data.txt", "w") as f:
                    f.write(cache_path)
            
            # Create map centered on Laguna Lake
            m = folium.Map(
                location=[14.35, 121.2], 
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Add lake boundary if provided
            filtered_stations = self.stations
            if lake_boundary_path and os.path.exists(lake_boundary_path):
                try:
                    lake_gdf = gpd.read_file(lake_boundary_path)
                    
                    # Add lake boundary as semi-transparent fill
                    folium.GeoJson(
                        lake_gdf,
                        name="Laguna Lake Boundary",
                        style_function=lambda x: {
                            'fillColor': '#b3d9ff',
                            'color': '#3366ff', 
                            'weight': 2, 
                            'fillOpacity': 0.2
                        },
                        tooltip="Laguna Lake"
                    ).add_to(m)
                    
                    # Filter stations to only those within the lake boundary
                    filtered_stations = []
                    for station in self.stations:
                        try:
                            point = gpd.GeoSeries(
                                gpd.points_from_xy([float(station['lon'])], [float(station['lat'])]), 
                                crs=lake_gdf.crs
                            )
                            
                            # Using a safer approach to check containment
                            is_within = False
                            for i, geom in enumerate(lake_gdf.geometry):
                                try:
                                    if point.iloc[0].within(geom):
                                        is_within = True
                                        break
                                except Exception as e:
                                    print(f"Error checking point within geometry {i}: {e}")
                                    continue
                            
                            if is_within:
                                filtered_stations.append(station)
                        except Exception as e:
                            print(f"Error processing station {station['id']} for lake boundary: {e}")
                            # If error occurs, include the station by default
                            filtered_stations.append(station)
                            
                    print(f"Filtered stations: {len(filtered_stations)} of {len(self.stations)} are within lake boundary")
                except Exception as e:
                    print(f"Error processing lake boundary: {e}")
                    filtered_stations = self.stations
            
            # Add pulsing CSS style
            self.add_pulse_style(m)
            
            # Create feature groups for different layers
            stations_group = folium.FeatureGroup(name="Stations")
            wind_group = folium.FeatureGroup(name="Wind Direction")
            heatmap_group = folium.FeatureGroup(name="Chloro Density")
            
            # List to collect station data for table
            station_data = []
            
            # Add stations with pulse effect
            for station in self.stations:
                station_id = station['id']
                
                # Add pulsing marker for the station
                folium.Marker(
                    location=[station['lat'], station['lon']],
                    popup=f"<b>Station:</b> {station.get('name', f'Station {station_id}')}<br><b>ID:</b> {station_id}",
                    icon=folium.DivIcon(
                        html=f'<div class="pulse"></div>'
                    )
                ).add_to(stations_group)
                
                # Add wind direction if available
                if wind_data_d and str(station_id) in wind_data_d:
                    wind_direction = wind_data_d[str(station_id)]
                    
                    # Get wind speed if available, otherwise use default
                    wind_speed = 20.0  # Default wind speed in km/h
                    if wind_data_s and str(station_id) in wind_data_s:
                        wind_speed = wind_data_s[str(station_id)]
                    
                    # Calculate arrow length based on wind speed (in km/h)
                    min_length = 0.005  # Minimum arrow length
                    max_length = 0.03   # Maximum arrow length
                    min_speed = 0       # Minimum expected wind speed
                    max_speed = 60      # Maximum expected wind speed in km/h (adjust as needed)
                    
                    # Scale arrow length based on wind speed
                    if wind_speed > max_speed:
                        arrow_length = max_length
                    elif wind_speed < min_speed:
                        arrow_length = min_length
                    else:
                        # Linear scaling
                        arrow_length = min_length + (wind_speed - min_speed) / (max_speed - min_speed) * (max_length - min_length)
                    
                    # Convert wind direction from meteorological to cartesian angle
                    wind_rad = np.radians((90 - wind_direction) % 360)
                    
                    # Calculate arrow endpoint
                    end_lat = station["lat"] + arrow_length * np.sin(wind_rad)
                    end_lon = station["lon"] + arrow_length * np.cos(wind_rad)
                    
                    # Get cardinal direction
                    cardinal = self.get_cardinal_direction(wind_direction)
                    
                    # Add wind direction line with weight based on wind speed
                    # Scale line weight with wind speed (km/h)
                    line_weight = 1 + (wind_speed / 20)  # Adjusted for km/h scale
                    
                    folium.PolyLine(
                        locations=[[station["lat"], station["lon"]], [end_lat, end_lon]],
                        color='red',
                        weight=line_weight,
                        opacity=0.8,
                        popup=f"<b>{station.get('name', f'Station {station_id}')}</b><br>Wind Direction: {cardinal} ({wind_direction}°)<br>Wind Speed: {wind_speed:.1f} km/h"
                    ).add_to(wind_group)
                    
                    # Add to station data for table
                    station_data.append({
                        "Station": station.get("name", f"Station {station_id}"),
                        "ID": station_id,
                        "Latitude": station["lat"],
                        "Longitude": station["lon"],
                        "Wind Direction": f"{cardinal} ({wind_direction}°)",
                        "Wind Speed": f"{wind_speed:.1f} km/h"  # Changed to km/h
                    })

            # IMPROVED CHLOROPHYLL DATA HANDLING
            heat_data = []
            station_chlorophyll_map = {}
            
            # Create a mapping between station names in Excel and station IDs in GeoJSON
            # This is the critical fix - creating a proper mapping between different ID formats
            station_name_to_id_map = {
                "Station_1_CWB": "1",
                "Station_2_EastB": "2", 
                "Station_4_CentralB": "4",
                "Station_5_NorthernWestBay": "5",
                "Station_8_SouthB": "8",
                "Station_15_SanPedro": "15",
                "Station_16_Sta. Rosa": "16",
                "Station_17_Sanctuary": "17",
                "Station_18_Pagsanjan": "18"
            }
            
            # Debug: Print out station IDs from both sources to verify mapping
            if len(values) > 0 and 'Station' in values.columns:
                print("Station IDs from Excel:", values['Station'].unique())
                print("Station IDs from GeoJSON:", [station['id'] for station in self.stations])
            
            # Create a mapping dictionary from station ID to station coordinates
            station_coord_map = {str(station['id']): (station['lat'], station['lon']) for station in self.stations}
            
            # Path to save/load the chlorophyll mapping
            chlorophyll_map_path = "heatmapper/data_cache/station_chlorophyll_map.json"
            
            # Try to extract chlorophyll data from values DataFrame
            has_chloro_data = False
            if len(values) > 0 and 'Station' in values.columns and 'Chlorophyll-a (ug/L)' in values.columns:
                print("Processing chlorophyll data for heatmap...")
                
                # Extract valid chlorophyll readings
                valid_readings = values.dropna(subset=['Chlorophyll-a (ug/L)'])
                
                if len(valid_readings) > 0:
                    has_chloro_data = True
                    print(f"Found {len(valid_readings)} rows with valid chlorophyll readings")
                    
                    # Process each row with valid chlorophyll data
                    for _, row in valid_readings.iterrows():
                        try:
                            excel_station_id = str(row['Station'])  # Station name from Excel
                            chlorophyll = float(row['Chlorophyll-a (ug/L)'])
                            
                            # Map Excel station name to GeoJSON station ID
                            geojson_station_id = station_name_to_id_map.get(excel_station_id, excel_station_id)
                            
                            # Store in our mapping using the GeoJSON ID
                            station_chlorophyll_map[geojson_station_id] = chlorophyll
                            
                            # Add to heat_data if coordinates are available
                            if geojson_station_id in station_coord_map:
                                lat, lon = station_coord_map[geojson_station_id]
                                heat_data.append([lat, lon, chlorophyll])
                                print(f"Added heat point for station {geojson_station_id} (Excel: {excel_station_id}) with value {chlorophyll}")
                            else:
                                print(f"WARNING: No coordinates found for station {geojson_station_id} (Excel: {excel_station_id})")
                        except (ValueError, TypeError) as e:
                            print(f"Error processing row for station {row.get('Station', 'unknown')}: {e}")
                    
                    # Save the mapping to file if we found any valid data
                    if station_chlorophyll_map:
                        try:
                            with open(chlorophyll_map_path, 'w') as f:
                                # Convert keys to strings for JSON serialization
                                json_compatible_map = {str(k): float(v) for k, v in station_chlorophyll_map.items()}
                                json.dump(json_compatible_map, f)
                            
                            print(f"Saved station-to-chlorophyll mapping with {len(station_chlorophyll_map)} entries")
                            print(f"Successfully created {len(heat_data)} heat data points")
                        except Exception as e:
                            print(f"Error saving chlorophyll mapping: {e}")
            
            # If no current data available, try to load previous data
            if not heat_data:
                print("No current chlorophyll data points, trying to load previous data...")
                try:
                    if os.path.exists(chlorophyll_map_path):
                        with open(chlorophyll_map_path, 'r') as f:
                            saved_map = json.load(f)
                        
                        # Recreate heat_data from saved mapping
                        for station_id, chlorophyll in saved_map.items():
                            if station_id in station_coord_map:
                                lat, lon = station_coord_map[station_id]
                                heat_data.append([lat, lon, float(chlorophyll)])
                        
                        has_chloro_data = len(heat_data) > 0
                        print(f"Loaded previous chlorophyll data for {len(heat_data)} stations")
                    else:
                        print("No previous chlorophyll data available")
                except Exception as e:
                    print(f"Error loading previous chlorophyll data: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Add chlorophyll data visualization to the map
            print(f"Total heat data points: {len(heat_data)}")
            if heat_data:
                try:
                    # Create a separate feature group for the heatmap
                    heatmap_layer = folium.FeatureGroup(name="Chlorophyll Heatmap")
                    
                    # Fix potential numeric keys in the gradient to avoid Folium errors
                    safe_gradient = fix_numeric_keys({0.4: 'blue', 0.65: 'lime', 1: 'red'})
                    
                    # Add heatmap with safe gradient
                    heatmap = HeatMap(
                        data=heat_data,
                        radius=20,
                        blur=15,
                        max_zoom=13,
                        gradient=safe_gradient
                    )
                    heatmap.add_to(heatmap_layer)
                    
                    # Add to the map
                    heatmap_layer.add_to(m)
                    print("Successfully added heatmap to map")
                    
                    # Add circle markers for each data point - using safe string keys for all parameters
                    for point in heat_data:
                        lat, lon, value = point
                        
                        # Scale the radius based on value (chlorophyll)
                        radius = min(max(value * 0.7, 10), 60)  # Min 10, max 30
                        
                        # Color based on value
                        if value < 50:
                            color = 'blue'
                        elif value < 100:
                            color = 'green'
                        else:
                            color = 'red'
                        
                        # Add circle marker with safe parameters
                        circle_params = fix_numeric_keys({
                            'location': [lat, lon],
                            'radius': radius,
                            'color': color,
                            'fill': True,
                            'fill_color': color,
                            'fill_opacity': 0.6,
                            'popup': f"Chlorophyll-a: {value:.2f} μg/L"
                        })
                        
                        #folium.CircleMarker(**circle_params).add_to(heatmap_group)
                    
                    has_chloro_data = True
                    print(f"Added {len(heat_data)} circle markers to the map")
                    
                except Exception as e:
                    print(f"Error adding chlorophyll visualization: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try alternative visualization method if heatmap fails
                    try:
                        print("Trying alternative visualization method...")
                        for point in heat_data:
                            lat, lon, value = point
                            
                            # Add simple circle markers with safe parameters
                            circle_params = fix_numeric_keys({
                                'location': [lat, lon],
                                'radius': 10,
                                'color': 'blue',
                                'fill': True,
                                'fill_opacity': 0.7,
                                'popup': f"Chlorophyll-a: {value:.2f} μg/L"
                            })
                            
                            #folium.CircleMarker(**circle_params).add_to(heatmap_group)
                            
                        print("Added alternative chlorophyll visualization")
                        has_chloro_data = True
                    except Exception as e2:
                        print(f"Alternative visualization also failed: {e2}")
                        has_chloro_data = False
            
            # Add "No Data" message if needed
            if not has_chloro_data:
                print("No chlorophyll data available to display")
                
                # Add a message to the map about missing data
                no_data_msg = folium.Element('''
                <div style="position: fixed; 
                            top: 50%; left: 50%; transform: translate(-50%, -50%);
                            border:2px solid orange; z-index:9999; font-size:14px;
                            background-color: white; padding: 10px; opacity: 0.9; 
                            border-radius: 5px; text-align: center;">
                    <h3>Chlorophyll Data Missing</h3>
                    <p>No chlorophyll data available for the selected time period.</p>
                    <p>The station locations and wind directions are still shown.</p>
                </div>
                ''')
                m.get_root().html.add_child(no_data_msg)
            
            # Add all feature groups to the map
            stations_group.add_to(m)
            if wind_data_d:
                wind_group.add_to(m)
            heatmap_group.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add table with station data
            if station_data:
                # Use string keys for all parameters to avoid camelize errors
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
                
                # Add toggle button for table
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
                    <p><span style="color: #1e88e5;">●</span> Station Location</p>
                    <p><span style="color:red;">→</span> Wind Direction</p>
                '''

                # Add wind speed legend (updated for km/h)
            legend_html += '''
                <p style="margin-top: 5px;"><b>Wind Speed:</b></p>
                <p><span style="height:3px; width:20px; background:red; display:inline-block;"></span> Low (<20 km/h)</p>
                <p><span style="height:6px; width:30px; background:red; display:inline-block;"></span> Medium (20-40 km/h)</p>
                <p><span style="height:9px; width:40px; background:red; display:inline-block;"></span> High (>40 km/h)</p>
            '''

            if lake_boundary_path and os.path.exists(lake_boundary_path):
                legend_html += '''
                <p><span style="color:#3366ff;">▬</span> Lake Boundary</p>
                '''
                    
            legend_html += '''
                <p style="margin-top: 5px;"><b>Heat Map Colors:</b></p>
                <div style="width:20px; height:15px; background:blue; display:inline-block;"></div>
                <div style="width:20px; height:15px; background:lime; display:inline-block;"></div>
                <div style="width:20px; height:15px; background:red; display:inline-block;"></div>
                <p style="font-size:12px;">Low → Medium → High Chlorophyll-a</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            prediction_date_text = ""
            if selected_month and selected_year:
                prediction_date_text = f"Map Prediction on: {selected_month} {selected_year}<br>"

            # Add title and info
            title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50%; transform: translateX(-50%);
                        z-index:9999; font-size:18px; font-weight: bold;
                        background-color: white; padding: 10px; 
                        border-radius: 5px; opacity: 0.9;">
                Laguna Lake Monitoring Combined Map
                <div style="font-size: 12px; font-weight: normal; text-align: center;">
                    {len(self.stations)} stations • Map generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
                    {prediction_date_text}
                </div>
            </div>
            '''
            
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Save map
            output_path = "heatmapper/laguna_combined_map.html"
            m.save(output_path)
            return output_path
            
        except Exception as e:
            print(f"Error creating combined map: {e}")
            import traceback
            traceback.print_exc()
            return None