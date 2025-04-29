import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import Point
import os
import webbrowser

class HeatmapByParameter:
    def __init__(self, stations_geojson, csv_folder):
        self.stations_geojson = stations_geojson
        self.csv_folder = csv_folder
        self.stations = self.load_stations()

    def load_stations(self):
        gdf = gpd.read_file(self.stations_geojson)
        # Convert MultiPoints to Points
        def multipoint_to_point(geom):
            if geom.geom_type == 'MultiPoint':
                first_point = list(geom.geoms)[0].coords[0]
                return Point(first_point)
            return geom

        gdf['geometry'] = gdf['geometry'].apply(multipoint_to_point)
        gdf['Latitude'] = gdf.geometry.y
        gdf['Longitude'] = gdf.geometry.x
        return gdf

    def get_latest_value(self, station_id, parameter):
        # Build CSV filename based on station ID
        csv_file = f"Station_{station_id}.csv"  # You might adjust if naming differs slightly
        csv_path = os.path.join(self.csv_folder, csv_file)

        if not os.path.exists(csv_path):
            print(f"CSV for Station {station_id} not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            if parameter not in df.columns:
                print(f"Parameter {parameter} not found in {csv_path}")
                return None

            latest_value = df[parameter].dropna().iloc[-1]  # Get last non-NaN value
            return latest_value
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            return None

    def generate_map(self, parameter, output_html="station_heatmap.html"):
        m = folium.Map(
                location=[self.stations['Latitude'].mean(), self.stations['Longitude'].mean()],
                zoom_start=11,
                min_zoom=10,  # <- restrict minimum zoom out
                max_zoom=12,  # <- restrict maximum zoom in
                max_bounds=True  # <- prevent panning too far away
            )
        self.current_parameter = parameter

        for idx, row in self.stations.iterrows():
            station_id = row['id']
            lat = row['Latitude']
            lon = row['Longitude']

            latest_value = self.get_latest_value(station_id, parameter)

            if latest_value is not None:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=self.scale_radius(latest_value),
                    color='yellow',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    popup=f"Station {station_id}<br>{parameter}: {latest_value}"
                ).add_to(m)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=(self.scale_radius(latest_value)+20),
                    color='Yellow',
                    fill=True,
                    fill_color='Green',
                    fill_opacity=0.3,
                    popup=f"Station {station_id}<br>{parameter}: {latest_value}"
                ).add_to(m)

        m.save(output_html)
        print(f"Heatmap saved as {output_html}")
        #webbrowser.open(output_html)

    def scale_radius(self, value):
        # Smart scaling depending on parameter value
        if value <= 1:
            return 5 + (value * 5)  # Small values (like Ammonia)
        elif value <= 10:
            return 5 + (value * 2)   # Medium values (like pH, DO)
        else:
            return 5 + (value * 0.5) # Big values (like Chlorophyll-a)