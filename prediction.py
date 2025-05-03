import customtkinter as ctk
import webview
import requests
from heatmapper.Heatmapper import HeatmapByParameter
import folium
from folium import plugins
import pandas as pd

class PredictionPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # File paths and coordinates
        self.geojson_path = "heatmapper/stations_final.geojson"
        self.laguna_coords = {"lat": 14.35, "lon": 121.2}
        
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
        
        self.generate_button = ctk.CTkButton(
            self.control_frame,
            text="Yes",
            width=100,
            height=30,
            command=self.generate_heatmap
        )
        self.generate_button.pack(pady=5)
        
        # Add geoheatmap button
        self.heatmap_button = ctk.CTkButton(
            self.control_frame,
            text="Show Heatmap",
            width=100,
            height=30,
            command=self.generate_geoheatmap
        )
        self.heatmap_button.pack(side="left", padx=10)
        
        # Create content frame for the main area
        self.content_frame = ctk.CTkFrame(self, fg_color="#F5F5F5")
        self.content_frame.pack(fill="both", expand=True)

    def generate_heatmap(self):
        """Generate heatmap based on selected months"""
        try:
            selected_month = self.month_var.get()
            print(f"Selected month: {selected_month}")
            
            # Create and show map regardless of month selection
            heatmap = HeatmapByParameter()
            output_path = heatmap.create_pulse_map(self.geojson_path)
            
            webview.create_window(
                "Laguna Lake Monitoring Stations",
                output_path,
                width=900,
                height=700
            )
            webview.start()
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")

    def show_station_map(self):
        """Display the station map with pulsing points"""
        try:
            heatmap = HeatmapByParameter()
            output_path = heatmap.create_pulse_map(self.geojson_path)
            
            webview.create_window(
                "Laguna Lake Monitoring Stations",
                output_path,
                width=900,
                height=700
            )
            webview.start()
            
        except Exception as e:
            print(f"Error showing map: {e}")

    def show_weather_data(self):
        """Display weather data from Open-Meteo API"""
        try:
            # Fetch weather data
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": self.laguna_coords["lat"],
                "longitude": self.laguna_coords["lon"],
                "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
                "timezone": "Asia/Manila"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Create weather window
            weather_html = self.create_weather_html(data)
            with open("weather.html", "w") as f:
                f.write(weather_html)
            
            webview.create_window(
                "Laguna Lake Weather",
                "weather.html",
                width=800,
                height=600
            )
            webview.start()
            
        except Exception as e:
            print(f"Error showing weather data: {e}")

    def create_weather_html(self, data):
        """Create HTML display for weather data"""
        return f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial; padding: 20px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .weather-card {{ 
                        border: 1px solid #ddd; 
                        padding: 20px;
                        margin: 10px;
                        border-radius: 8px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Weather Forecast for Laguna Lake</h2>
                    <div class="weather-card">
                        <h3>Current Conditions</h3>
                        <p>Temperature: {data['hourly']['temperature_2m'][0]}°C</p>
                        <p>Humidity: {data['hourly']['relative_humidity_2m'][0]}%</p>
                        <p>Wind Speed: {data['hourly']['wind_speed_10m'][0]} km/h</p>
                    </div>
                </div>
            </body>
        </html>
        """

    def analyze_data(self):
        """Analyze the selected parameter data"""
        selected_parameter = self.parameter_var.get()
        print(f"Analyzing {selected_parameter} data...")
        # Add your analysis logic here
        
    def export_results(self):
        """Export the analysis results"""
        print("Exporting results...")
        # Add your export logic here

    def generate_geoheatmap(self):
        """Generate Folium heatmap based on station data"""
        try:
            # Read station data from GeoJSON
            df = pd.read_json(self.geojson_path)
            
            # Create base map centered on Laguna Lake
            m = folium.Map(
                location=[14.35, 121.2],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Extract coordinates and create heatmap data
            heat_data = [[point['coordinates'][1], point['coordinates'][0]] 
                        for point in df['features'][0]['geometry']['coordinates']]
            
            # Add heatmap layer
            plugins.HeatMap(heat_data).add_to(m)
            
            # Save map
            heatmap_path = "heatmaps/geoheatmap.html"
            m.save(heatmap_path)
            
            # Display in webview
            webview.create_window(
                "Laguna Lake Heatmap",
                heatmap_path,
                width=900,
                height=700
            )
            webview.start()
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")

    def show(self):
        """Show the prediction page"""
        self.grid(row=0, column=0, sticky="nsew")