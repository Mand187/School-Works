import math

class FlightCalc:
    def __init__(self, lat1, lon1, lat2, lon2):
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2
        self.r = 3958.8  # radius of earth

    def haversineFormula(self):
        phi1 = math.radians(self.lat1)
        phi2 = math.radians(self.lat2)
        delta_phi = math.radians(self.lat2 - self.lat1)
        delta_lambda = math.radians(self.lon2 - self.lon1)

        a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = self.r * c

        return distance

    def getFlightTime(self, speed=500):
        s = speed

        phi1 = math.radians(self.lat1)
        phi2 = math.radians(self.lat2)
        delta_phi = math.radians(self.lat2 - self.lat1)
        delta_lambda = math.radians(self.lon2 - self.lon1)

        a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = self.r * c

        time = distance / s

        return time

class Location: 
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude 

def getLocations():
    locations = [
        Location("Miami Beach, FL, USA", 25.793449, -80.139198),
        Location("Fargo, ND, USA", 46.877186, -96.789803),
        Location("Idaho City, ID, USA", 43.828850, -115.837860),
        Location("Northampton, MA, USA", 42.328674, -71.217133),
        Location("Newburyport, MA, USA", 42.810356, -70.893875),
        Location("New Bedford, MA, USA", 41.638409, -70.941208),
        Location("Medford, MA, USA", 42.419331, -71.119720),
        Location("Malden, MA, USA", 42.429752, -71.071022),
        Location("Leominster, MA, USA", 42.525482, -71.764183),
        Location("Lawrence, MA, USA", 42.701283, -71.175682)
    ]
    return locations

def main():
    locations = getLocations()

    # Print the list of locations and let the user choose two locations
    print("Available Locations:")
    for i, location in enumerate(locations):
        print(f"{i + 1}. {location.name}")

    try:
        # Get user input for the first location selection
        first_location_index = int(input("Select the first location (enter the corresponding number): ")) - 1
        first_location = locations[first_location_index]

        # Get user input for the second location selection
        second_location_index = int(input("Select the second location (enter the corresponding number): ")) - 1
        second_location = locations[second_location_index]

        print(f"\nSelected Locations: {first_location.name} and {second_location.name}")

        # Example: Using the selected locations for the FlightCalc instantiation
        flight_calc = FlightCalc(first_location.latitude, first_location.longitude, second_location.latitude, second_location.longitude)

        distance = flight_calc.haversineFormula()
        print(f"\nDistance between {first_location.name} and {second_location.name} is {distance:.2f} miles")

        flight_time = flight_calc.getFlightTime()
        print(f"Flight time: {flight_time:.2f} hours")

    except (ValueError, IndexError):
        print("Invalid input. Please enter valid numbers corresponding to locations.")
    except KeyboardInterrupt:
        print("\nUser interrupted the program.")
    finally:
        print("Exiting the program.")

if __name__ == "__main__":
    main()

