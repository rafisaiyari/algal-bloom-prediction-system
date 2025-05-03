import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import numpy as np


class HeatmapByParameter:
    def __init__(self, excel_path=None, geojson_path=None):
        """Initialize the heatmap generator"""
        self.data = None
        self.stations = None
        self.station_id_mapping = {}  # Initialize empty mapping
        
        if excel_path:
            self.data = self.load_merged_excel(excel_path)
        if geojson_path:
            self.stations = self.load_coordinates(geojson_path)

    

    def load_coordinates(self, geojson_path):
        """Load station coordinates from GeoJSON file"""
        try:
            gdf = gpd.read_file(geojson_path)
            
            stations = []
            for _, row in gdf.iterrows():
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

            return stations
            
        except Exception as e:
            print(f"Error loading GeoJSON file {geojson_path}: {e}")
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
                .marker-pulse {
                    position: relative;
                }
                .marker-pulse:before {
                    content: '';
                    position: absolute;
                    width: 30px;
                    height: 30px;
                    left: -15px;
                    top: -15px;
                    background-color: rgba(255, 0, 0, 0.4);
                    border-radius: 50%;
                    animation: pulse 1.5s ease-out infinite;
                    z-index: -1;
                }
                @keyframes pulse {
                    0% {
                        transform: scale(0.1);
                        opacity: 0.8;
                    }
                    70% {
                        transform: scale(2);
                        opacity: 0.3;
                    }
                    100% {
                        transform: scale(3);
                        opacity: 0;
                    }
                }
            </style>
        """
        
        map_obj.get_root().header.add_child(folium.Element(pulse_css))

    def create_pulse_map(self, geojson_path):
        """Create a map with pulsing station markers and a heatmap"""
        # Load station data
        self.stations = self.load_coordinates(geojson_path)
        
        # Create map centered on Laguna Lake
        m = folium.Map(
            location=[14.35, 121.2], 
            zoom_start=11,
            tiles='CartoDB positron'
        )
        
        # List to collect coordinates for the heatmap
        heat_data = []

        # Add stations with pulse effect and custom intensity
        for station in self.stations:
            # Get a custom intensity (e.g., based on a parameter like ammonia)
            intensity = station.get('parameter_value', 1)  # Default to 1 if not available

            # Add pulsing marker for the station (fixed size)
            folium.Marker(
                location=[station['lat'], station['lon']],
                popup=f"Station {station['name']}",
                icon=folium.DivIcon(
                    html=f'<div class="pulse" style="width: 10px; height: 10px; border-radius: 50%; background: #1e88e5; border: 2px solid #1e88e5;"></div>'
                )
            ).add_to(m)
            
            # Add the station's coordinates and intensity to the heatmap data
            heat_data.append([station['lat'], station['lon'], intensity])

        # Add HeatMap to the map with collected coordinates and intensities
        # Set the radius to a fixed value to avoid scaling with zoom level
        HeatMap(heat_data, radius=20, blur=10, max_zoom=18, opacity=0.6).add_to(m)

        # Save map
        output_path = "heatmapper/laguna_stations_with_heatmap.html"
        m.save(output_path)
        return output_path

