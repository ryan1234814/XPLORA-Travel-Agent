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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return db_connection, cursor

def save_itinerary(origin, destination, duration, budget, interests, itinerary_data):
    try:
        db_connection, cursor = get_db_connection()
        insert_query = """
            INSERT INTO itineraries (origin, destination, duration, budget, interests, itinerary_data)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            origin, 
            destination, 
            duration, 
            budget, 
            json.dumps(interests), 
            json.dumps(itinerary_data)
        ))
        db_connection.commit()
        cursor.close()
        db_connection.close()
    except Exception as db_err:
        print(f"Database error: {str(db_err)}")
