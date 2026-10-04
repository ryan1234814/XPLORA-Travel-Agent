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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trip_ratings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            destination VARCHAR(255) NOT NULL,
            origin VARCHAR(255),
            rating INT NOT NULL,
            feedback TEXT,
            trip_title VARCHAR(512),
            helpful_votes INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_destination (destination),
            INDEX idx_rating (rating)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS place_reviews_cache (
            id INT AUTO_INCREMENT PRIMARY KEY,
            query VARCHAR(512) NOT NULL,
            place_id VARCHAR(255),
            place_name VARCHAR(512),
            rating DOUBLE,
            user_ratings_total INT,
            photos JSON,
            reviews JSON,
            google_maps_url TEXT,
            raw_data LONGTEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_query (query),
            INDEX idx_place_id (place_id)
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


def save_trip_rating(destination, rating, feedback="", origin="", trip_title=""):
    try:
        db_connection, cursor = get_db_connection()
        cursor.execute(
            "INSERT INTO trip_ratings (destination, origin, rating, feedback, trip_title) VALUES (%s,%s,%s,%s,%s)",
            (destination, origin, int(rating), feedback or "", trip_title or "")
        )
        db_connection.commit()
        cursor.close()
        db_connection.close()
        return True
    except Exception as e:
        print(f"save_trip_rating error: {e}")
        return False


def get_trip_ratings(destination=None, limit=20):
    try:
        db_connection, cursor = get_db_connection()
        if destination:
            cursor.execute("SELECT destination, rating, feedback, trip_title, helpful_votes, created_at FROM trip_ratings WHERE destination=%s ORDER BY created_at DESC LIMIT %s", (destination, limit))
        else:
            cursor.execute("SELECT destination, rating, feedback, trip_title, helpful_votes, created_at FROM trip_ratings ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        result = [dict(zip(cols, r)) for r in rows]
        cursor.close()
        db_connection.close()
        return result
    except Exception as e:
        print(f"get_trip_ratings error: {e}")
        return []


def get_rating_stats(destination=None):
    try:
        db_connection, cursor = get_db_connection()
        if destination:
            cursor.execute("SELECT COUNT(*) as cnt, AVG(rating) as avg_rating FROM trip_ratings WHERE destination=%s", (destination,))
        else:
            cursor.execute("SELECT COUNT(*) as cnt, AVG(rating) as avg_rating FROM trip_ratings")
        row = cursor.fetchone()
        cursor.execute("SELECT rating, COUNT(*) as c FROM trip_ratings WHERE destination=%s GROUP BY rating" if destination else "SELECT rating, COUNT(*) as c FROM trip_ratings GROUP BY rating", (destination,) if destination else ())
        dist = cursor.fetchall()
        cursor.close()
        db_connection.close()
        cnt = row[0] if row else 0
        avg = float(row[1]) if row and row[1] else 0.0
        buckets = {int(r[0]): int(r[1]) for r in dist} if dist else {}
        return {"count": cnt, "avg": round(avg, 2) if cnt else 0, "distribution": buckets}
    except Exception as e:
        print(f"get_rating_stats error: {e}")
        return {"count": 0, "avg": 0, "distribution": {}}


def get_place_cache(query):
    try:
        db_connection, cursor = get_db_connection()
        cursor.execute("SELECT place_id, place_name, rating, user_ratings_total, photos, reviews, google_maps_url, raw_data FROM place_reviews_cache WHERE query=%s LIMIT 1", (query,))
        row = cursor.fetchone()
        cursor.close()
        db_connection.close()
        if not row:
            return None
        import json as _json
        return {
            "place_id": row[0], "place_name": row[1], "rating": row[2], "user_ratings_total": row[3],
            "photos": _json.loads(row[4]) if row[4] else [], "reviews": _json.loads(row[5]) if row[5] else [],
            "google_maps_url": row[6], "raw": _json.loads(row[7]) if row[7] else {}
        }
    except Exception as e:
        print(f"get_place_cache error: {e}")
        return None


def set_place_cache(query, data):
    try:
        import json as _json
        db_connection, cursor = get_db_connection()
        cursor.execute("""
            INSERT INTO place_reviews_cache (query, place_id, place_name, rating, user_ratings_total, photos, reviews, google_maps_url, raw_data)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE place_id=VALUES(place_id), place_name=VALUES(place_name), rating=VALUES(rating),
              user_ratings_total=VALUES(user_ratings_total), photos=VALUES(photos), reviews=VALUES(reviews),
              google_maps_url=VALUES(google_maps_url), raw_data=VALUES(raw_data)
        """, (query, data.get("place_id"), data.get("place_name"), data.get("rating"), data.get("user_ratings_total"),
              _json.dumps(data.get("photos", [])), _json.dumps(data.get("reviews", [])), data.get("google_maps_url"), _json.dumps(data.get("raw", {}))))
        db_connection.commit()
        cursor.close()
        db_connection.close()
    except Exception as e:
        print(f"set_place_cache error: {e}")
