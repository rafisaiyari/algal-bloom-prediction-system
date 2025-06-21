import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import numpy as np
from datetime import datetime
import json

import sys
import os
import locale

# Force UTF-8 encoding for Windows
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    # Reconfigure stdout and stderr to use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        # For older Python versions
        import codecs

        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller .exe"""
    try:
        base_path = sys._MEIPASS  # PyInstaller sets this attr
    except AttributeError:
        base_path = os.path.abspath("heatmapper")
    return os.path.join(base_path, relative_path)

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

        # Always try to load from CSV regardless of excel_path parameter
        csv_path = resource_path("heatmapper/chlorophyll_predictions_by_station.csv")
        if os.path.exists(csv_path):
            try:
                # Load the CSV file directly
                self.data = pd.read_csv(csv_path)
                print(f"Loaded CSV data with shape: {self.data.shape}")

                # Convert Date to datetime
                self.data["Date"] = pd.to_datetime(self.data["Date"])

                # Prepare station names for matching
                if "Station" in self.data.columns:
                    self.data["Station"] = self.data["Station"].apply(
                        lambda x: f"Station_{x}" if not str(x).startswith("Station_") else x
                    )

                # Ensure chlorophyll column exists with correct name
                if "Predicted_Chlorophyll" in self.data.columns and "Chlorophyll-a (ug/L)" not in self.data.columns:
                    self.data["Chlorophyll-a (ug/L)"] = self.data["Predicted_Chlorophyll"]

                # Generate Year and Month columns
                self.data["Year"] = self.data["Date"].dt.year
                self.data["Month"] = self.data["Date"].dt.month_name()

            except Exception as e:
                print(f"Error loading CSV file {csv_path}: {e}")
                self.data = None

        if geojson_path:
            self.stations = self.load_coordinates(geojson_path)

    def load_merged_excel(self, excel_path):
        """Redirects to load data from CSV instead of Excel"""
        csv_path = resource_path("CSV/chlorophyll_predictions_by_station.csv")
        try:
            data = pd.read_csv(csv_path)

            # Add "Station_" prefix to station names if needed
            if "Station" in data.columns:
                data["Station"] = data["Station"].apply(
                    lambda x: f"Station_{x}" if not str(x).startswith("Station_") else x
                )

            # Map Predicted_Chlorophyll to Chlorophyll-a (ug/L) if needed
            if "Predicted_Chlorophyll" in data.columns and "Chlorophyll-a (ug/L)" not in data.columns:
                data["Chlorophyll-a (ug/L)"] = data["Predicted_Chlorophyll"]

            print(f"Loaded CSV data with shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
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

    def get_chlorophyll_recommendation(self, chlorophyll_value):
        """
        Get a recommendation based on chlorophyll-a value
        
        Args:
            chlorophyll_value (float): Chlorophyll-a value in μg/L
            
        Returns:
            tuple: (recommendation text, color)
        """
        if chlorophyll_value is None or np.isnan(chlorophyll_value):
            return "No Data Available", "#D5DBDB"
        
        # Define thresholds and corresponding recommendations
        if chlorophyll_value < 25:  # Very Low
            return "No Action Required", "#98FB98"
        elif chlorophyll_value < 50:  # Low
            return "Monitor", "#90EE90"
        elif chlorophyll_value < 100:  # Medium
            return "Artificial Aeration", "#3CB371"
        elif chlorophyll_value < 150:  # High
            return "Aerate", "#228B22"
        else:  # Very High (> 150)
            return "Harvest", "#006400"

    def create_combined_map(self, geojson_path, values, lake_boundary_path=None, wind_data_d=None, wind_data_s=None,
                        selected_year=None, selected_month=None):
        """Create a combined map with station markers, wind direction, and heatmap constrained to lake boundary"""
        try:
            # Ensure output directory exists
            os.makedirs("heatmapper", exist_ok=True)

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

                    print(
                        f"Filtered stations: {len(filtered_stations)} of {len(self.stations)} are within lake boundary")
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

            # IMPORTANT - PROCESS CHLOROPHYLL DATA FIRST, BEFORE CREATING STATION MARKERS
            # Initialize heat data and chlorophyll map
            heat_data = []
            station_chlorophyll_map = {}

            # Create a mapping between station names in CSV and station IDs in GeoJSON
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

            # Create a mapping dictionary from station ID to station coordinates
            station_coord_map = {str(station['id']): (station['lat'], station['lon']) for station in self.stations}

            # Always load chlorophyll data directly from CSV
            csv_path = resource_path("CSV/chlorophyll_predictions_by_station.csv")
            has_chloro_data = False

            if os.path.exists(csv_path):
                try:
                    print(f"Loading chlorophyll data directly from: {csv_path}")
                    chloro_df = pd.read_csv(csv_path)

                    # Convert Date to datetime
                    chloro_df["Date"] = pd.to_datetime(chloro_df["Date"])

                    # Extract year and month
                    chloro_df["Year"] = chloro_df["Date"].dt.year
                    chloro_df["Month"] = chloro_df["Date"].dt.month_name()

                    # Filter by selected year and month if provided
                    if selected_year is not None and selected_month is not None:
                        print(f"Filtering for {selected_month} {selected_year}")
                        chloro_df = chloro_df[chloro_df["Year"] == selected_year]
                        chloro_df = chloro_df[chloro_df["Month"].str.lower() == selected_month.lower()]

                    # Identify chlorophyll column
                    chloro_column = None
                    if "Chlorophyll-a (ug/L)" in chloro_df.columns:
                        chloro_column = "Chlorophyll-a (ug/L)"
                    elif "Predicted_Chlorophyll" in chloro_df.columns:
                        chloro_column = "Predicted_Chlorophyll"

                    # Process chlorophyll data if column exists and has data
                    if chloro_column and len(chloro_df) > 0:
                        print(f"Found {len(chloro_df)} rows with chlorophyll data in CSV")

                        # Process each row with chlorophyll data
                        for _, row in chloro_df.iterrows():
                            try:
                                # Get station ID from CSV
                                csv_station_id = str(row['Station'])
                                chlorophyll = float(row[chloro_column])

                                # Fix station ID format for mapping to GeoJSON
                                if csv_station_id.startswith("Station_"):
                                    station_key = csv_station_id
                                else:
                                    station_key = f"Station_{csv_station_id}"

                                # Map to GeoJSON station ID
                                geojson_station_id = station_name_to_id_map.get(station_key,
                                                                                csv_station_id.replace("Station_", ""))

                                # Store in our mapping
                                station_chlorophyll_map[geojson_station_id] = chlorophyll

                                # Add to heat_data if coordinates are available
                                if geojson_station_id in station_coord_map:
                                    lat, lon = station_coord_map[geojson_station_id]
                                    heat_data.append([lat, lon, chlorophyll])
                                    print(
                                        f"Added heat point for station {geojson_station_id} (CSV: {csv_station_id}) with value {chlorophyll}")
                                else:
                                    print(
                                        f"WARNING: No coordinates found for station {geojson_station_id} (CSV: {csv_station_id})")
                            except (ValueError, TypeError) as e:
                                print(f"Error processing CSV row for station {row.get('Station', 'unknown')}: {e}")

                        has_chloro_data = len(heat_data) > 0
                    else:
                        print(f"No chlorophyll column or data found in CSV for the selected period")
                except Exception as e:
                    print(f"Error processing chlorophyll data from CSV: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Chlorophyll data CSV file not found: {csv_path}")

            # Debug station chlorophyll map
            print("Station chlorophyll mapping:")
            for station_id, value in station_chlorophyll_map.items():
                print(f"  Station {station_id}: {value} μg/L")

            # NOW ADD STATION MARKERS WITH CHLOROPHYLL DATA
            # Add stations with pulse effect
            for station in self.stations:
                station_id = station['id']
                
                # Get chlorophyll value for this station if available
                chlorophyll_value = station_chlorophyll_map.get(str(station_id), None)
                
                # Debug
                print(f"Station {station_id}: Chlorophyll value = {chlorophyll_value}")
                
                # Get recommendation based on chlorophyll value
                recommendation, color = self.get_chlorophyll_recommendation(chlorophyll_value)
                
                # Create popup content with chlorophyll value and recommendation
                popup_content = f"""
                <div style="min-width: 180px;">
                    <b>Station:</b> {station.get('name', f'Station {station_id}')}<br>
                    <b>ID:</b> {station_id}<br>
                """
                
                if chlorophyll_value is not None:
                    popup_content += f"""
                    <b>Chlorophyll-a:</b> {chlorophyll_value:.2f} μg/L<br>
                    <b>Recommendation:</b> <span style="color:{color}; font-weight:bold;">{recommendation}</span>
                    """
                else:
                    popup_content += """
                    <b>Chlorophyll-a:</b> No data available<br>
                    <b>Recommendation:</b> No recommendation available
                    """
                    
                popup_content += "</div>"
                
                # Add pulsing marker for the station
                folium.Marker(
                    location=[station['lat'], station['lon']],
                    popup=folium.Popup(popup_content, max_width=300),
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
                    max_length = 0.03  # Maximum arrow length
                    min_speed = 0  # Minimum expected wind speed
                    max_speed = 60  # Maximum expected wind speed in km/h (adjust as needed)

                    # Scale arrow length based on wind speed
                    if wind_speed > max_speed:
                        arrow_length = max_length
                    elif wind_speed < min_speed:
                        arrow_length = min_length
                    else:
                        # Linear scaling
                        arrow_length = min_length + (wind_speed - min_speed) / (max_speed - min_speed) * (
                                    max_length - min_length)

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

            # Create a mapping dictionary from station ID to station coordinates
            station_coord_map = {str(station['id']): (station['lat'], station['lon']) for station in self.stations}

            # Always load chlorophyll data directly from CSV
            csv_path = resource_path("CSV/chlorophyll_predictions_by_station.csv")
            has_chloro_data = False

            if os.path.exists(csv_path):
                try:
                    print(f"Loading chlorophyll data directly from: {csv_path}")
                    chloro_df = pd.read_csv(csv_path)

                    # Convert Date to datetime
                    chloro_df["Date"] = pd.to_datetime(chloro_df["Date"])

                    # Extract year and month
                    chloro_df["Year"] = chloro_df["Date"].dt.year
                    chloro_df["Month"] = chloro_df["Date"].dt.month_name()

                    # Filter by selected year and month if provided
                    if selected_year is not None and selected_month is not None:
                        print(f"Filtering for {selected_month} {selected_year}")
                        chloro_df = chloro_df[chloro_df["Year"] == selected_year]
                        chloro_df = chloro_df[chloro_df["Month"].str.lower() == selected_month.lower()]

                    # Identify chlorophyll column
                    chloro_column = None
                    if "Chlorophyll-a (ug/L)" in chloro_df.columns:
                        chloro_column = "Chlorophyll-a (ug/L)"
                    elif "Predicted_Chlorophyll" in chloro_df.columns:
                        chloro_column = "Predicted_Chlorophyll"

                    # Process chlorophyll data if column exists and has data
                    if chloro_column and len(chloro_df) > 0:
                        print(f"Found {len(chloro_df)} rows with chlorophyll data in CSV")

                        # Process each row with chlorophyll data
                        for _, row in chloro_df.iterrows():
                            try:
                                # Get station ID from CSV
                                csv_station_id = str(row['Station'])
                                chlorophyll = float(row[chloro_column])

                                # Fix station ID format for mapping to GeoJSON
                                if csv_station_id.startswith("Station_"):
                                    station_key = csv_station_id
                                else:
                                    station_key = f"Station_{csv_station_id}"

                                # Map to GeoJSON station ID
                                geojson_station_id = station_name_to_id_map.get(station_key,
                                                                                csv_station_id.replace("Station_", ""))

                                # Store in our mapping
                                station_chlorophyll_map[geojson_station_id] = chlorophyll

                                # Add to heat_data if coordinates are available
                                if geojson_station_id in station_coord_map:
                                    lat, lon = station_coord_map[geojson_station_id]
                                    heat_data.append([lat, lon, chlorophyll])
                                    print(
                                        f"Added heat point for station {geojson_station_id} (CSV: {csv_station_id}) with value {chlorophyll}")
                                else:
                                    print(
                                        f"WARNING: No coordinates found for station {geojson_station_id} (CSV: {csv_station_id})")
                            except (ValueError, TypeError) as e:
                                print(f"Error processing CSV row for station {row.get('Station', 'unknown')}: {e}")

                        has_chloro_data = len(heat_data) > 0
                    else:
                        print(f"No chlorophyll column or data found in CSV for the selected period")
                except Exception as e:
                    print(f"Error processing chlorophyll data from CSV: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Chlorophyll data CSV file not found: {csv_path}")

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
                        radius = min(max(value * 0.7, 10), 60)  # Min 10, max 60

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

                        # folium.CircleMarker(**circle_params).add_to(heatmap_group)

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

                            # folium.CircleMarker(**circle_params).add_to(heatmap_group)

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
                
                <p style="margin-top: 5px;"><b>Recommendations:</b></p>
                <p style="font-size:12px;"><span style="color:#98FB98;">●</span> Very Low (<25 μg/L): No Action Required</p>
                <p style="font-size:12px;"><span style="color:#90EE90;">●</span> Low (25-50 μg/L): Monitor</p>
                <p style="font-size:12px;"><span style="color:#3CB371;">●</span> Medium (50-100 μg/L): Artificial Aeration</p>
                <p style="font-size:12px;"><span style="color:#228B22;">●</span> High (100-150 μg/L): Aerate</p>
                <p style="font-size:12px;"><span style="color:#006400;">●</span> Very High (>150 μg/L): Harvest</p>
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