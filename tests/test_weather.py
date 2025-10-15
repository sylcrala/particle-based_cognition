from apis.general_automation.weather_utils import WeatherUtils
from apis.general_automation.location_utils import LocationUtils
from shared_services import logging

weather_api = WeatherUtils()

print("testing openmeteo weather fetch...")
try:
    print(weather_api.get_weather_meteo())
except Exception as e:
    print(f"Error fetching openmeteo weather data: {e}")

print("testing wttr weather fetch...")
try:
    print(weather_api.get_weather_wttr())
except Exception as e:
    print(f"Error fetching wttr weather data: {e}")
