import mysql.connector
import os
import json
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    db_connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "newpassword"),
        connection_timeout=2
    )
    cursor = db_connection.cursor()

    
    db_name = os.getenv("MYSQL_DATABASE", "travel_agent")
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    cursor.execute(f"USE {db_name}")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS itineraries (
            id INT AUTO_INCREMENT PRIMARY KEY,
            origin VARCHAR(255),
            destination VARCHAR(255),
            duration INT,
            budget VARCHAR(255),
            interests TEXT,
            itinerary_data LONGTEXT,
            travel_dates VARCHAR(255),
            group_size INT,
            group_type VARCHAR(255),
            dietary_requirements TEXT,
            accessibility TEXT,
            pace VARCHAR(100),
            accommodation_preference VARCHAR(255),
            occasion VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return db_connection, cursor

def save_itinerary(origin, destination, duration, budget, interests, itinerary_data,
                   travel_dates="", group_size=2, group_type="Couple",
                   dietary_requirements=None, accessibility=None, pace="Moderate",
                   accommodation_preference="No preference", occasion=""):
    try:
        db_connection, cursor = get_db_connection()
        insert_query = """
            INSERT INTO itineraries (origin, destination, duration, budget, interests, itinerary_data,
                                     travel_dates, group_size, group_type, dietary_requirements,
                                     accessibility, pace, accommodation_preference, occasion)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            origin, 
            destination, 
            duration, 
            budget, 
            json.dumps(interests), 
            json.dumps(itinerary_data),
            travel_dates or "",
            group_size or 2,
            group_type or "Couple",
            json.dumps(dietary_requirements or []),
            json.dumps(accessibility or []),
            pace or "Moderate",
            accommodation_preference or "No preference",
            occasion or ""
        ))
        db_connection.commit()
        cursor.close()
        db_connection.close()
    except Exception as db_err:
        print(f"Database error: {str(db_err)}")
