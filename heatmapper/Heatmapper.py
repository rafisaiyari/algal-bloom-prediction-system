import geopandas as gpd
import pandas as pd
import folium
import os

class HeatmapByParameter:
    def __init__(self, csv_folder, geojson_path):
        self.data = self.load_station_csvs(csv_folder)
        self.stations = self.load_coordinates(geojson_path)

    def load_station_csvs(self, folder):
        all_data = []
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                full_station_name = filename.replace(".csv", "")
                parts = full_station_name.split("_")
                if len(parts) >= 2:
                    station_id = parts[0] + "_" + parts[1]  # e.g., Station_1
                else:
                    station_id = full_station_name

                path = os.path.join(folder, filename)
                df = pd.read_csv(path)

                df["Station"] = station_id
                df["Date"] = pd.to_datetime(df["Date"], format="%b-%y", errors='coerce')
                df["Year"] = df["Date"].dt.year
                df["Month"] = df["Date"].dt.month
                all_data.append(df)

        return pd.concat(all_data, ignore_index=True)

    def load_coordinates(self, geojson_path):
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
                "lon": point.x
            })

        return stations

    def generate_map(self, parameter, year, month, output_path="heatmapper/station_heatmap.html"):
        center_lat = sum([s["lat"] for s in self.stations]) / len(self.stations)
        center_lon = sum([s["lon"] for s in self.stations]) / len(self.stations)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, max_bounds=True)

        for station in self.stations:
            sid = station["id"]
            subset = self.data[
                (self.data["Station"] == sid) &
                (self.data["Year"] == year) &
                (self.data["Month"] == month)
            ]

            print(f"[INFO] Station: {sid}, Param: {parameter}, Year: {year}, Month: {month}")
            if not subset.empty and parameter in subset.columns:
                value = subset[parameter].values[0]
                print(f"[VALUE] {parameter}: {value}")
                if pd.notna(value):
                    folium.CircleMarker(
                        location=[station["lat"], station["lon"]],
                        radius=self.scale_radius(value),
                        color='blue',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.6,
                        popup=f"Station {sid}<br>{parameter}: {value:.3f}"
                    ).add_to(m)
            else:
                print(f"[WARN] No data for {sid} or parameter '{parameter}' not found.")

        m.save(output_path)

    def scale_radius(self, value):
        return 4 + min(20, value * 25)
