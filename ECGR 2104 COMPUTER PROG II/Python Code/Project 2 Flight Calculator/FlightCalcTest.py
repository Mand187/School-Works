import unittest
from FlightCalc import FlightCalc
from FlightCalc import Location


class TestFlightCalc(unittest.TestCase):
    def setUp(self):
        # Create a list with just two locations (Miami and Fargo)
        self.locations = [
            Location("Miami, FL, USA", 25.793449, -80.139198),
            Location("Fargo, ND, USA", 46.877186, -96.789803)
        ]

    def test_haversineFormula(self):
        miami, fargo = self.locations
        flight_calc = FlightCalc(miami.latitude, miami.longitude, fargo.latitude, fargo.longitude)
        distance = flight_calc.haversineFormula()
        self.assertAlmostEqual(distance, 1718.51, places=2)  # Adjust the expected value as needed

    def test_getFlightTime(self):
        miami, fargo = self.locations
        flight_calc = FlightCalc(miami.latitude, miami.longitude, fargo.latitude, fargo.longitude)
        flight_time = flight_calc.getFlightTime()
        self.assertAlmostEqual(flight_time, 3.44, places=2)  # Adjust the expected value as needed

if __name__ == '__main__':
    unittest.main()
