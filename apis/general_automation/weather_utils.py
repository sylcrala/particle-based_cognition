"""
overall weather utility for the toolkit, callable anywhere in the codebase
"""

import requests
from apis.api_registry import api

class WeatherUtils:
    def __init__(self):
        self.logger = api.get_api("logger")
        self.openmeteo_url = "https://open-meteo.com/v1/forecast"
        self.wttr_url = "https://wttr.in/"

    def get_weather_meteo(self, location=None):
        if location is None:
            location = "Vineland, New Jersey, USA"
        else:
            location = location
        try:
            location_api = api.get_api("location_utility_api")
            coords = location_api.get_coords(location)
            if not coords:
                self.logger.log(f"Could not get coordinates for location: {location}", "ERROR", "WeatherAPI", "get_weather_meteo")
                raise Exception(f"No coordinates found for location: {location}")
            else:
                lat, lon = coords.values()
        except Exception as e:
            self.logger.log(f"Error getting location coordinates: {e}", "ERROR", "WeatherAPI", "get_weather_meteo")
            print(e)
        
        lat, lon = coords.values()

        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "hourly": "temperature_2m,precipitation_probability",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 3
        }

        try:
            response = requests.get(self.openmeteo_url, params=params)

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.log(f"Failed to fetch weather data: {response.status_code}", "ERROR", "WeatherAPI", "get_weather_meteo")
                raise Exception(f"Weather API error: {response.status_code}")
        except Exception as e:
            self.logger.log(f"Error fetching weather data: {e}", "ERROR", "WeatherAPI", "get_weather_meteo")
            print(e)
            return None

    def get_weather_wttr(self, location=None):
        if location is None:
            location = "Vineland, New Jersey, USA"
        else:
            location = location
        """    
        try:
            location_api = api.get_api("location_utility_api")
            coords = location_api.get_coords(location)
            if not coords:
                self.logger.log(f"Could not get coordinates for location: {location}", "ERROR", "WeatherAPI", "get_weather_wttr")
                raise Exception(f"No coordinates found for location: {location}")
            else:
                lat, lon = coords.values()
        except Exception as e:
            self.logger.log(f"Error getting location coordinates: {e}", "ERROR", "WeatherAPI", "get_weather_wttr")
            print(e)
            return None
        """

        url = self.wttr_url + f"{location}?format=j1"

        try:
            response = requests.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.log(f"Failed to fetch weather data: {response.status_code}", "ERROR", "WeatherAPI", "get_weather_wttr")
                raise Exception(f"Weather API error: {response.status_code}")
        except Exception as e:
            self.logger.log(f"Error fetching weather data: {e}", "ERROR", "WeatherAPI", "get_weather")
            print(e)
            return None



# Register the API
api.register_api("weather_utils", WeatherUtils())
