import geopandas as gpd
import pandas as pd
import folium
import os
import numpy as np

class HeatmapByParameter:
    def __init__(self, excel_path, geojson_path):
        """
        Initialize the heatmap generator using a merged Excel file instead of individual CSVs.
        
        Args:
            excel_path (str): Path to the merged_stations Excel file
            geojson_path (str): Path to the GeoJSON file with station coordinates
        """
        self.data = self.load_merged_excel(excel_path)
        self.stations = self.load_coordinates(geojson_path)
        
        # Create parameter mappings to handle different column names
        self.parameter_mapping = {
            "Nitrate": "Nitrate (mg/L)",
            "Phosphate": "Inorganic Phosphate (mg/L)",
            "Dissolved Oxygen": "Dissolved Oxygen (mg/L)",
            "DO": "Dissolved Oxygen (mg/L)",
            "pH": "pH (units)",
            "Ammonia": "Ammonia (mg/L)",
            "Chlorophyll-a": "Chlorophyll-a (ug/L)",
            "Temperature": "Temperature",
            "Phytoplankton": "Phytoplankton"
        }
        
        # Define color mappings and thresholds for different parameters
        self.color_mappings = {
            "Nitrate (mg/L)": {
                "thresholds": [0.5, 1.0, 2.0, 5.0],
                "colors": ['green', 'yellow', 'orange', 'red', 'darkred']
            },
            "Inorganic Phosphate (mg/L)": {
                "thresholds": [0.05, 0.1, 0.2, 0.5],
                "colors": ['green', 'yellow', 'orange', 'red', 'darkred']
            },
            "Dissolved Oxygen (mg/L)": {
                "thresholds": [2.0, 4.0, 6.0, 8.0],
                "colors": ['darkred', 'red', 'orange', 'yellow', 'green']  # Inverted for DO (higher is better)
            },
            "pH (units)": {
                "thresholds": [6.0, 6.5, 7.5, 8.5],
                "colors": ['red', 'orange', 'green', 'orange', 'red']  # pH is centered around neutral
            },
            "Ammonia (mg/L)": {
                "thresholds": [0.1, 0.5, 1.0, 2.0],
                "colors": ['green', 'yellow', 'orange', 'red', 'darkred']
            },
            "Chlorophyll-a (ug/L)": {
                "thresholds": [5, 10, 20, 40],
                "colors": ['green', 'yellow', 'orange', 'red', 'darkred']
            },
            "Temperature": {
                "thresholds": [25, 28, 30, 32],
                "colors": ['blue', 'green', 'yellow', 'orange', 'red']
            },
            "Phytoplankton": {
                "thresholds": [1000, 5000, 10000, 50000],
                "colors": ['green', 'yellow', 'orange', 'red', 'darkred']
            }
        }
        
        # Create station ID mapping for GeoJSON to Excel
        self.station_id_mapping = {
            "Station_1": "Station_1_CWB",
            "Station_2": "Station_2_EastB",
            "Station_4": "Station_4_CentralB", 
            "Station_5": "Station_5_NorthernWestBay",
            "Station_8": "Station_8_SouthB",
            "Station_15": "Station_15_SanPedro",
            "Station_16": "Station_16_Sta. Rosa",
            "Station_17": "Station_17_Sanctuary",
            "Station_18": "Station_18_Pagsanjan"
        }

    def load_merged_excel(self, excel_path):
        """
        Load data from the merged Excel file
        
        Args:
            excel_path (str): Path to the merged Excel file
            
        Returns:
            pandas.DataFrame: Processed data from the Excel file
        """
        try:
            # Read the Excel file
            df = pd.read_excel(excel_path)
            
            # Process dates and extract year and month
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            
            # Convert numeric columns to ensure proper handling
            numeric_columns = [
                "pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", 
                "Inorganic Phosphate (mg/L)", "Dissolved Oxygen (mg/L)", 
                "Temperature", "Chlorophyll-a (ug/L)"
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            return df
            
        except Exception as e:
            print(f"Error loading Excel file {excel_path}: {e}")
            # Return empty DataFrame if there's an error
            return pd.DataFrame()

    def load_coordinates(self, geojson_path):
        """
        Load station coordinates from GeoJSON file
        
        Args:
            geojson_path (str): Path to the GeoJSON file
            
        Returns:
            list: List of dictionaries with station information
        """
        try:
            gdf = gpd.read_file(geojson_path)
            stations = []
            
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'MultiPoint':
                    point = geom.geoms[0]
                elif geom.geom_type == 'Point':
                    point = geom
                else:
                    continue

                stations.append({
                    "id": row["id"],
                    "lat": point.y,
                    "lon": point.x,
                    "name": row.get("name", row["id"])  # Use name if available, else id
                })

            return stations
            
        except Exception as e:
            print(f"Error loading GeoJSON file {geojson_path}: {e}")
            return []

    def get_parameter_column(self, parameter):
        """
        Get the actual column name for a parameter
        
        Args:
            parameter (str): Parameter name as used in the UI
            
        Returns:
            str: Actual column name in the DataFrame
        """
        return self.parameter_mapping.get(parameter, parameter)

    def get_color_for_value(self, parameter, value):
        """
        Determine color based on parameter value and thresholds
        
        Args:
            parameter (str): Parameter column name
            value (float): Parameter value
            
        Returns:
            str: Color code for the given value
        """
        if parameter not in self.color_mappings:
            # Default to blue if no mapping exists
            return 'blue'
            
        thresholds = self.color_mappings[parameter]["thresholds"]
        colors = self.color_mappings[parameter]["colors"]
        
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return colors[i]
        
        # If value is higher than all thresholds
        return colors[-1]

    def get_radius_for_value(self, parameter, value):
        """
        Determine marker radius based on parameter value
        
        Args:
            parameter (str): Parameter column name
            value (float): Parameter value
            
        Returns:
            float: Radius for the circle marker
        """
        # Base radius
        base_radius = 8
        
        # Scale factor - different for each parameter
        scale_factors = {
            "Nitrate (mg/L)": 2,
            "Inorganic Phosphate (mg/L)": 15,
            "Dissolved Oxygen (mg/L)": 1,
            "pH (units)": 1,
            "Ammonia (mg/L)": 5,
            "Chlorophyll-a (ug/L)": 0.2,
            "Temperature": 0.3,
            "Phytoplankton": 0.0002
        }
        
        scale = scale_factors.get(parameter, 1)
        
        # Cap the radius to prevent extremely large circles
        return min(base_radius + (value * scale), 25)

    def generate_map(self, parameter, year, month, output_path="heatmapper/station_heatmap.html"):
        """
        Generate a heatmap for the specified parameter, year, and month
        
        Args:
            parameter (str): Parameter to visualize
            year (int): Year to filter data
            month (int): Month to filter data
            output_path (str): Path to save the HTML output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Map UI parameter to actual column name
            param_column = self.get_parameter_column(parameter)
            
            # Calculate map center based on average of station coordinates
            if not self.stations:
                print("[ERROR] No station coordinates available")
                return False
                
            center_lat = sum(s["lat"] for s in self.stations) / len(self.stations)
            center_lon = sum(s["lon"] for s in self.stations) / len(self.stations)

            # Create map
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=11,
                tiles='CartoDB positron',
                max_bounds=True
            )
            
            # Add title
            title_html = f'''
                <h3 align="center" style="font-size:16px">
                    <b>{parameter} Levels - {month}/{year}</b>
                </h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))

            # Track data availability
            data_available = False
            
            # Loop through stations to add markers
            for station in self.stations:
                # Get the station ID from GeoJSON
                sid = station["id"]
                
                # Map GeoJSON station ID to Excel station ID if needed
                excel_sid = self.station_id_mapping.get(sid, sid)
                
                # Filter data for this station, year, and month
                subset = self.data[
                    (self.data["Station"] == excel_sid) &
                    (self.data["Year"] == year) &
                    (self.data["Month"] == month)
                ]

                print(f"[INFO] Station: {sid} (Excel: {excel_sid}), Param: {param_column}, Year: {year}, Month: {month}")
                
                # Check if we have data for this station and parameter
                if not subset.empty and param_column in subset.columns:
                    # Get the parameter value
                    value = subset[param_column].values[0]
                    
                    if pd.notna(value):
                        data_available = True
                        print(f"[VALUE] {param_column}: {value}")
                        
                        # Determine color based on value
                        color = self.get_color_for_value(param_column, value)
                        
                        # Determine radius based on value
                        radius = self.get_radius_for_value(param_column, value)
                        
                        # Add marker to map
                        folium.CircleMarker(
                            location=[station["lat"], station["lon"]],
                            radius=radius,
                            color='black',
                            weight=1,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            popup=f"<b>Station: {station.get('name', sid)}</b><br>{parameter}: {value:.3f}"
                        ).add_to(m)
                else:
                    print(f"[WARN] No data for {sid} or parameter '{param_column}' not found.")
                    # Add a gray marker for stations with no data
                    folium.CircleMarker(
                        location=[station["lat"], station["lon"]],
                        radius=6,
                        color='gray',
                        weight=1,
                        fill=True,
                        fill_color='lightgray',
                        fill_opacity=0.5,
                        popup=f"<b>Station: {station.get('name', sid)}</b><br>No data available"
                    ).add_to(m)
            
            # Add legend
            if param_column in self.color_mappings:
                self._add_legend(m, param_column)
            
            # Save map
            m.save(output_path)
            
            if not data_available:
                print("[WARN] No data available for the selected parameters")
                self._create_no_data_map(output_path, parameter, year, month)
                return False
                
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to generate heatmap: {e}")
            self._create_error_map(output_path, str(e))
            return False
    
    def _add_legend(self, map_obj, parameter):
        """
        Add a legend to the map
        
        Args:
            map_obj (folium.Map): Map object to add legend to
            parameter (str): Parameter name
        """
        if parameter not in self.color_mappings:
            return
            
        thresholds = self.color_mappings[parameter]["thresholds"]
        colors = self.color_mappings[parameter]["colors"]
        
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: auto;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: white; padding: 10px;
                    opacity: 0.9">
        <div style="text-align: center; margin-bottom: 5px"><b>''' + parameter + '''</b></div>
        '''
        
        # Add legend items
        for i, color in enumerate(colors):
            if i == 0:
                label = f"< {thresholds[i]}"
            elif i == len(colors) - 1:
                label = f"> {thresholds[-1]}"
            else:
                label = f"{thresholds[i-1]} - {thresholds[i]}"
                
            legend_html += f'''
            <div>
                <span style="background-color: {color}; display: inline-block; width: 15px; height: 15px;"></span>
                <span style="padding-left: 5px;">{label}</span>
            </div>
            '''
            
        legend_html += '</div>'
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
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