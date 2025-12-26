
import unittest
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date
from data.models import (
    Weather, Attraction, Hotel, Transportation, DayPlan, TripSummary,
    create_mock_weather, create_mock_attraction, create_mock_hotel
)

class TestModels(unittest.TestCase):
    
    def test_weather_model(self):
        weather = Weather(
            temperature=25.0,
            humidity=60,
            wind_speed=12.0,
            description="Sunny",
            feels_like=27.0,
            date="2024-01-01"
        )
        self.assertEqual(weather.temperature, 25.0)
        self.assertIn("Sunny", str(weather))

    def test_attraction_model(self):
        attraction = Attraction(
            name="Museum",
            type="cultural",
            price_level=2,
            rating=4.5,
            address="123 Main St",
            description="A great museum",
            location="City Center",
            estimated_cost=20.0,
            duration=120
        )
        self.assertEqual(attraction.name, "Museum")
        self.assertIn("Museum", str(attraction))

    def test_hotel_model(self):
        hotel = Hotel(
            name="Grand Hotel",
            rating=4.8,
            price_per_night=200.0,
            address="456 Hotel Lane",
            amenities=["Pool", "Gym"]
        )
        self.assertEqual(hotel.calculate_total_cost(3), 600.0)
        self.assertIn("Grand Hotel", str(hotel))

    def test_transportation_model(self):
        transport = Transportation(
            mode="Taxi",
            estimated_cost=30.0,
            duration=30
        )
        self.assertEqual(transport.mode, "Taxi")
        self.assertIn("Taxi", str(transport))

    def test_day_plan_model(self):
        weather = create_mock_weather()
        day_plan = DayPlan(
            day=1,
            date="2024-01-01",
            weather=weather,
            daily_cost=100.0
        )
        
        # Test default empty lists
        self.assertEqual(day_plan.attractions, [])
        self.assertEqual(day_plan.restaurants, [])
        
        # Test activity count
        attraction = create_mock_attraction()
        day_plan.attractions.append(attraction)
        self.assertEqual(day_plan.get_total_activities(), 1)
        
        self.assertIn("Day 1", str(day_plan))

    def test_trip_summary_model(self):
        trip_summary = TripSummary(
            destination="Paris",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            total_days=5,
            total_cost=1000.0,
            daily_budget=200.0,
            currency="USD",
            converted_total=1000.0,
            itinerary=[],
            hotels=[]
        )
        
        self.assertEqual(trip_summary.destination, "Paris")
        self.assertEqual(trip_summary.get_cost_per_person(2), 500.0)
        self.assertEqual(trip_summary.get_average_daily_cost(), 200.0)
        
        # Test post_init dictionaries
        self.assertEqual(trip_summary.trip_overview, {})
        self.assertEqual(trip_summary.travel_tips, [])
        
        self.assertIn("Trip to Paris", str(trip_summary))

    def test_helper_functions(self):
        weather = create_mock_weather()
        self.assertIsInstance(weather, Weather)
        
        attraction = create_mock_attraction()
        self.assertIsInstance(attraction, Attraction)
        
        hotel = create_mock_hotel()
        self.assertIsInstance(hotel, Hotel)

if __name__ == '__main__':
    unittest.main(verbosity=2)
