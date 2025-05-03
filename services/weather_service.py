import requests
from datetime import datetime, timedelta

class WeatherService:
    """Weather data service using Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.marine_url = "https://marine-api.open-meteo.com/v1"
        
        # Laguna Lake center coordinates
        self.lat = 14.35
        self.lon = 121.2

    def get_weather_forecast(self):
        """Get 7-day weather forecast"""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", 
                      "precipitation", "wind_speed_10m", "wind_direction_10m"],
            "timezone": "Asia/Manila"
        }
        
        response = requests.get(f"{self.base_url}/forecast", params=params)
        return response.json()

    def get_water_data(self):
        """Get marine and water conditions"""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": ["water_temperature", "wave_height", "wave_direction",
                      "wind_wave_height", "wind_wave_direction"],
            "timezone": "Asia/Manila"
        }
        
        response = requests.get(f"{self.marine_url}/marine", params=params)
        return response.json()