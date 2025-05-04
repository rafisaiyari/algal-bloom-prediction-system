import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import numpy as np
from datetime import datetime


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

    def create_wind_direction_map(self, geojson_path, wind_data):
        """Create a map with wind direction arrows"""
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
            
            # Add wind direction arrows for each station
            for station in self.stations:
                station_id = station['id']
                
                # Check if we have wind data for this station
                if station_id in wind_data:
                    wind_direction = wind_data[station_id]
                    
                    # Calculate arrow endpoint based on wind direction
                    arrow_length = 0.01  # Adjust for visibility
                    
                    # Convert wind direction from meteorological to cartesian angle
                    # In meteorological convention, 0° is north, 90° is east
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
                    ).add_to(m)
            
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
                    Map generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Save map
            output_path = "heatmapper/laguna_wind_direction.html"
            m.save(output_path)
            return output_path
            
        except Exception as e:
            print(f"Error creating wind direction map: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_combined_map(self, geojson_path, lake_boundary_path=None, wind_data=None):
        """Create a combined map with station markers, wind direction, and heatmap constrained to lake boundary"""
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
            
            # Add lake boundary if provided
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
                        point = gpd.GeoSeries(
                            gpd.points_from_xy([station['lon']], [station['lat']]), 
                            crs=lake_gdf.crs
                        )
                        if any(point.within(geom) for geom in lake_gdf.geometry):
                            filtered_stations.append(station)
                            
                    print(f"Filtered stations: {len(filtered_stations)} of {len(self.stations)} are within lake boundary")
                except Exception as e:
                    print(f"Error processing lake boundary: {e}")
                    filtered_stations = self.stations
            else:
                filtered_stations = self.stations
            
            # Add pulsing CSS style
            self.add_pulse_style(m)
            
            # Create feature groups for different layers
            stations_group = folium.FeatureGroup(name="Stations")
            wind_group = folium.FeatureGroup(name="Wind Direction")
            heatmap_group = folium.FeatureGroup(name="Station Density")
            
            # List to collect coordinates for the heatmap - only use filtered stations
            heat_data = []
            
            # Add stations with pulse effect and wind directions
            for station in self.stations:  # Show all stations
                station_id = station['id']
                
                # Add pulsing marker for the station
                folium.Marker(
                    location=[station['lat'], station['lon']],
                    popup=f"<b>Station:</b> {station['name']}<br><b>ID:</b> {station['id']}",
                    icon=folium.DivIcon(
                        html=f'<div class="pulse"></div>'
                    )
                ).add_to(stations_group)
                
                # Only add to heatmap if in filtered stations
                if station in filtered_stations:
                    heat_data.append([station['lat'], station['lon']])
                
                # Add wind direction if data available
                if wind_data and station_id in wind_data:
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
            
            # Add HeatMap to the heatmap layer - simplified to avoid errors
            if heat_data:
                HeatMap(
                    heat_data, 
                    radius=20, 
                    blur=15, 
                    max_zoom=13
                ).add_to(heatmap_group)
            
            # Add all feature groups to the map
            stations_group.add_to(m)
            if wind_data:
                wind_group.add_to(m)
            heatmap_group.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add legend including lake boundary if provided
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
            
            if lake_boundary_path and os.path.exists(lake_boundary_path):
                legend_html += '''
                <p><span style="color:#3366ff;">▬</span> Lake Boundary</p>
                '''
                
            legend_html += '''
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
                Laguna Lake Monitoring Combined Map
                <div style="font-size: 12px; font-weight: normal; text-align: center;">
                    {len(self.stations)} stations • Map generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}
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