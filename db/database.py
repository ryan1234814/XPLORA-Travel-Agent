"""Neon (serverless Postgres) persistence layer.

Replaces the old MySQL implementation. Connection settings come from a single
`DATABASE_URL` env var — use the *pooled* Neon endpoint, which looks like:

    postgresql://user:pass@ep-xxx-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require

Neon requires TLS, so `sslmode=require` must stay in the string.
"""

import atexit
import os
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal

from dotenv import load_dotenv
from psycopg import errors as pgerr
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

load_dotenv()

# Postgres types come back as datetime/Decimal; the API serialises responses as
# plain JSON, so every row is normalised before it leaves this module.
_SCALAR = (str, int, float, bool, bytes, bytearray, Decimal, datetime, date, type(None))


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, _SCALAR):
        return value
    return str(value)


_pool: ConnectionPool | None = None

# The schema is created on demand so a misconfigured DATABASE_URL can never stop
# the health check endpoint from answering.
_schema_ready = False


def _database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set — copy the pooled connection string from the "
            "Neon console (https://console.neon.tech) into .env"
        )
    return url


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_database_url(),
            min_size=1,
            max_size=5,
            open=True,
            timeout=15,
            max_idle=300,
            kwargs={"row_factory": dict_row, "autocommit": True, "prepare_threshold": 1},
        )
        # Without this the pool's threads are joined during interpreter
        # finalisation and short-lived scripts print a shutdown error.
        atexit.register(_pool.close)
    return _pool


@contextmanager
def _db():
    conn = get_pool().getconn(timeout=15)
    try:
        yield conn
    finally:
        get_pool().putconn(conn)


def execute(query, params=()):
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount


def query_one(query, params=()):
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return _json_safe(row) if row else None


def query_all(query, params=()):
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [_json_safe(dict(r)) for r in cur.fetchall()]


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id             BIGSERIAL PRIMARY KEY,
    email          TEXT NOT NULL,
    email_lower    TEXT GENERATED ALWAYS AS (lower(email)) STORED UNIQUE,
    password_hash  TEXT NOT NULL,
    display_name   TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS itineraries (
    id                       BIGSERIAL PRIMARY KEY,
    user_id                  BIGINT REFERENCES users(id) ON DELETE CASCADE,
    origin                   TEXT,
    destination              TEXT,
    duration                 INT,
    budget                   TEXT,
    interests                JSONB,
    itinerary_data           JSONB,
    travel_dates             TEXT,
    group_size               INT,
    group_type               TEXT,
    dietary_requirements     JSONB,
    accessibility            JSONB,
    pace                     TEXT,
    accommodation_preference TEXT,
    occasion                 TEXT,
    title                    TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_itineraries_user ON itineraries (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS trip_ratings (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT REFERENCES users(id) ON DELETE SET NULL,
    destination  TEXT NOT NULL,
    origin       TEXT,
    rating       INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    feedback     TEXT,
    trip_title   TEXT,
    helpful_votes INT NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ratings_destination ON trip_ratings (destination);
CREATE INDEX IF NOT EXISTS idx_ratings_rating ON trip_ratings (rating);
CREATE INDEX IF NOT EXISTS idx_ratings_user ON trip_ratings (user_id);

CREATE TABLE IF NOT EXISTS place_reviews_cache (
    id                BIGSERIAL PRIMARY KEY,
    query             TEXT NOT NULL UNIQUE,
    place_id          TEXT,
    place_name        TEXT,
    rating            DOUBLE PRECISION,
    user_ratings_total INT,
    photos            JSONB,
    reviews           JSONB,
    google_maps_url   TEXT,
    raw_data          JSONB,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cache_place_id ON place_reviews_cache (place_id);
"""


def ensure_schema() -> None:
    """Create the tables on first use. Safe to call repeatedly."""
    global _schema_ready
    if _schema_ready:
        return
    with _db() as conn:
        conn.execute(SCHEMA)
    _schema_ready = True


def bootstrap() -> bool:
    """Best-effort schema creation at startup. Returns success; never raises."""
    try:
        ensure_schema()
        return True
    except Exception as e:
        print(f"[DB] Neon bootstrap deferred: {e}")
        return False


# ─────────────────────────── users ───────────────────────────

_USER_PUBLIC = "id, email, display_name, created_at"


def create_user(email, password_hash, display_name=None):
    ensure_schema()
    try:
        return query_one(
            f"""INSERT INTO users (email, password_hash, display_name)
                VALUES (%s, %s, %s)
                RETURNING {_USER_PUBLIC}""",
            (email.strip(), password_hash, (display_name or "").strip() or None),
        )
    except pgerr.UniqueViolation:
        # email_lower is UNIQUE — an existing account is the caller's problem to
        # phrase (409), not a 500.
        return None


def get_user_by_email(email):
    ensure_schema()
    return query_one(
        f"SELECT {_USER_PUBLIC} FROM users WHERE email_lower = lower(%s)",
        (email.strip(),),
    )


def get_user_by_id(user_id):
    return query_one(f"SELECT {_USER_PUBLIC} FROM users WHERE id = %s", (user_id,))


def get_password_hash(email):
    """Internal helper for the sign-in path: returns the row with the hash."""
    return query_one(
        "SELECT id, email, display_name, created_at, password_hash "
        "FROM users WHERE email_lower = lower(%s)",
        (email.strip(),),
    )


def verify_credentials(email, password):
    """Returns (user, reason).

    user is None when authentication fails; reason is then 'unknown_email',
    'bad_password' or 'db_error' so the caller can phrase the error and the
    timing side channels stay neutral (hash() runs for unknown emails too).
    """
    from backend.auth import verify_password

    try:
        row = get_password_hash(email)
    except Exception as e:
        print(f"verify_credentials db error: {e}")
        return None, "db_error"
    if not row:
        return None, "unknown_email"
    if not verify_password(password, row["password_hash"]):
        return None, "bad_password"
    return {k: v for k, v in row.items() if k != "password_hash"}, "ok"


# ───────────────────────── itineraries ─────────────────────────

# Heavy JSONB payload excluded from list queries: a dashboard with hundreds of
# trips should not drag megabytes of itinerary text over the wire.
_SUMMARY_COLS = """
    id, user_id, origin, destination, duration, budget, interests, travel_dates,
    group_size, group_type, dietary_requirements, accessibility, pace,
    accommodation_preference, occasion, title, created_at
"""


def save_itinerary(origin, destination, duration, budget, interests, itinerary_data,
                   travel_dates="", group_size=2, group_type="Couple",
                   dietary_requirements=None, accessibility=None, pace="Moderate",
                   accommodation_preference="No preference", occasion="",
                   user_id=None, title=""):
    try:
        ensure_schema()
        row = query_one(
            """
            INSERT INTO itineraries (user_id, origin, destination, duration, budget, interests,
                                     itinerary_data, travel_dates, group_size, group_type,
                                     dietary_requirements, accessibility, pace,
                                     accommodation_preference, occasion, title)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (user_id, origin, destination, duration, budget,
             Jsonb(interests or []), Jsonb(itinerary_data or {}), travel_dates or "",
             group_size or 2, group_type or "Couple",
             Jsonb(dietary_requirements or []), Jsonb(accessibility or []),
             pace or "Moderate", accommodation_preference or "No preference",
             occasion or "", (title or "")[:200] or None),
        )
        return row["id"] if row else None
    except Exception as db_err:
        print(f"Database error: {str(db_err)}")
        return None


def list_user_itineraries(user_id, limit=50):
    try:
        ensure_schema()
        return query_all(
            f"SELECT {_SUMMARY_COLS} FROM itineraries WHERE user_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (user_id, min(int(limit or 50), 100)),
        )
    except Exception as e:
        print(f"list_user_itineraries error: {e}")
        return []


def get_itinerary(itinerary_id, user_id=None):
    """Full itinerary. When user_id is given, access is restricted to that owner."""
    try:
        ensure_schema()
        if user_id is None:
            return query_one(
                "SELECT * FROM itineraries WHERE id = %s", (itinerary_id,)
            )
        return query_one(
            "SELECT * FROM itineraries WHERE id = %s AND user_id = %s",
            (itinerary_id, user_id),
        )
    except Exception as e:
        print(f"get_itinerary error: {e}")
        return None


def delete_itinerary(itinerary_id, user_id):
    try:
        affected = execute(
            "DELETE FROM itineraries WHERE id = %s AND user_id = %s",
            (itinerary_id, user_id),
        )
        return affected > 0
    except Exception as e:
        print(f"delete_itinerary error: {e}")
        return False


# ───────────────────────── trip ratings ─────────────────────────

def save_trip_rating(destination, rating, feedback="", origin="", trip_title="", user_id=None):
    try:
        ensure_schema()
        execute(
            "INSERT INTO trip_ratings (destination, origin, rating, feedback, trip_title, user_id) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (destination, origin, int(rating), feedback or "", trip_title or "", user_id),
        )
        return True
    except Exception as e:
        print(f"save_trip_rating error: {e}")
        return False


def get_trip_ratings(destination=None, limit=20):
    try:
        cols = "destination, origin, rating, feedback, trip_title, helpful_votes, created_at"
        if destination:
            return query_all(
                f"SELECT {cols} FROM trip_ratings WHERE destination = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (destination, min(int(limit or 20), 50)),
            )
        return query_all(
            f"SELECT {cols} FROM trip_ratings ORDER BY created_at DESC LIMIT %s",
            (min(int(limit or 20), 50),),
        )
    except Exception as e:
        print(f"get_trip_ratings error: {e}")
        return []


def get_rating_stats(destination=None):
    try:
        if destination:
            agg = query_one(
                "SELECT COUNT(*) AS cnt, AVG(rating) AS avg_rating FROM trip_ratings "
                "WHERE destination = %s",
                (destination,),
            )
            dist = query_all(
                "SELECT rating, COUNT(*) AS c FROM trip_ratings WHERE destination = %s "
                "GROUP BY rating",
                (destination,),
            )
        else:
            agg = query_one(
                "SELECT COUNT(*) AS cnt, AVG(rating) AS avg_rating FROM trip_ratings"
            )
            dist = query_all("SELECT rating, COUNT(*) AS c FROM trip_ratings GROUP BY rating")
        cnt = int(agg["cnt"]) if agg else 0
        avg = float(agg["avg_rating"]) if agg and agg["avg_rating"] is not None else 0.0
        return {
            "count": cnt,
            "avg": round(avg, 2) if cnt else 0,
            "distribution": {int(r["rating"]): int(r["c"]) for r in (dist or [])},
        }
    except Exception as e:
        print(f"get_rating_stats error: {e}")
        return {"count": 0, "avg": 0, "distribution": {}}


# ─────────────────────── place review cache ───────────────────────

def get_place_cache(query_str):
    try:
        ensure_schema()
        row = query_one(
            "SELECT place_id, place_name, rating, user_ratings_total, photos, reviews, "
            "google_maps_url, raw_data FROM place_reviews_cache WHERE query = %s LIMIT 1",
            (query_str,),
        )
        if not row:
            return None
        return {
            "place_id": row["place_id"],
            "place_name": row["place_name"],
            "rating": row["rating"],
            "user_ratings_total": row["user_ratings_total"],
            "photos": row["photos"] or [],
            "reviews": row["reviews"] or [],
            "google_maps_url": row["google_maps_url"],
            "raw": row["raw_data"] or {},
        }
    except Exception as e:
        print(f"get_place_cache error: {e}")
        return None


def set_place_cache(query_str, data):
    try:
        ensure_schema()
        execute(
            """
            INSERT INTO place_reviews_cache
                (query, place_id, place_name, rating, user_ratings_total,
                 photos, reviews, google_maps_url, raw_data)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (query) DO UPDATE SET
                place_id = EXCLUDED.place_id,
                place_name = EXCLUDED.place_name,
                rating = EXCLUDED.rating,
                user_ratings_total = EXCLUDED.user_ratings_total,
                photos = EXCLUDED.photos,
                reviews = EXCLUDED.reviews,
                google_maps_url = EXCLUDED.google_maps_url,
                raw_data = EXCLUDED.raw_data,
                updated_at = now()
            """,
            (query_str, data.get("place_id"), data.get("place_name"), data.get("rating"),
             data.get("user_ratings_total"), Jsonb(data.get("photos") or []),
             Jsonb(data.get("reviews") or []), data.get("google_maps_url"),
             Jsonb(data.get("raw") or {})),
        )
    except pgerr.UniqueViolation:
        # Two concurrent misses on the same query: the row now exists, so the
        # cached value is already good — nothing to do.
        print(f"set_place_cache race resolved for {query_str!r}")
    except Exception as e:
        print(f"set_place_cache error: {e}")
