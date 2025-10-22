import geopy
from geopy.geocoders import Nominatim
from apis.api_registry import api


class LocationUtils:
    def __init__(self):
        self.logger = api.get_api("logger")
        self.geolocator = Nominatim(user_agent="local_kit_app")

    def get_coords(self, location):
        try:
            loc = self.geolocator.geocode(location)
            if loc:
                longitude = loc.longitude
                latitude = loc.latitude
                return {"latitude": latitude, "longitude": longitude}
            else:
                self.logger.log(f"Location not found: {location}", "ERROR", "LocationUtils", "get_coords")
                return None
        except Exception as e:
            self.logger.log(f"Error getting coordinates: {e}", "ERROR", "LocationUtils", "get_coords")
            return None
        
api.register_api("location_utility_api", LocationUtils())