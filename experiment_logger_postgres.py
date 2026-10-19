import os
import time
import random
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import errors as pg_errors
from experiment_logger_base import BaseExperimentDB, BaseRunLogger

# ---------------------------------------------------------------------------
# Connection config (read from environment / password file)
# ---------------------------------------------------------------------------

POSTGRES_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.environ.get('POSTGRES_PORT', '5432'))
POSTGRES_USER = os.environ.get('POSTGRES_USER', 'alirezas')
POSTGRES_DB = os.environ.get('POSTGRES_DB', 'mydb')


def get_postgres_password():
    password_file = "/lustre/work/soh/alirezas/PLATO/DB/config/postgres-password"
    try:
        with open(password_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return os.environ.get('POSTGRES_PASSWORD', 'password')


POSTGRES_PASSWORD = get_postgres_password()
DB_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def convert_sqlite_to_postgres(sql_script: str) -> str:
    """Convert SQLite DDL to PostgreSQL-compatible DDL."""
    conversions = [
        ("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY"),
        ("DATETIME DEFAULT CURRENT_TIMESTAMP", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("REAL", "DECIMAL"),
        ("PRAGMA foreign_keys = ON;", "-- PRAGMA foreign_keys = ON; (not needed in PostgreSQL)"),
    ]
    result = sql_script
    for old, new in conversions:
        result = result.replace(old, new)
    return result


def _connect_with_retry(max_retries=12, base_delay=5.0, max_delay=120.0, **kwargs):
    """Connect to postgres with exponential backoff retry on 'too many clients'."""
    for attempt in range(max_retries):
        try:
            return psycopg2.connect(**kwargs)
        except psycopg2.OperationalError as e:
            if "too many clients" in str(e) and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 5)
                print(f"[postgres] Too many clients (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {delay:.1f}s…")
                time.sleep(delay)
            else:
                raise


def init_db(schema_path: str = "database_postgres.sql"):
    """Create the PostgreSQL database and apply schema if not already present."""
    try:
        conn = _connect_with_retry(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD,
            database='postgres',
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{POSTGRES_DB}'"
            )
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{POSTGRES_DB}"')
                print(f"Created database {POSTGRES_DB}")
        conn.close()

        conn = _connect_with_retry(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
        )
        if os.path.exists(schema_path):
            try:
                with conn.cursor() as cur, open(schema_path, 'r') as f:
                    cur.execute(convert_sqlite_to_postgres(f.read()))
                conn.commit()
                print("Schema applied successfully")
            except Exception as schema_err:
                conn.rollback()
                if "already exists" in str(schema_err) or "DuplicateTable" in type(schema_err).__name__:
                    print("Schema already exists, skipping.")
                else:
                    print(f"Schema warning (non-fatal): {schema_err}")
        else:
            print(f"Warning: Schema file {schema_path} not found")
        conn.close()
        print("DB has been created/initialized.")
    except Exception as e:
        print(f"DB initialization failed: {e}")
        raise


# ---------------------------------------------------------------------------
# ExperimentDB — Postgres backend
# ---------------------------------------------------------------------------

class ExperimentDB(BaseExperimentDB):

    def __init__(self, url: str = DB_URL):
        super().__init__(url)

    def _init_db(self):
        init_db()

    def _after_log_execution(self):
        self.db.commit()


RunLogger = BaseRunLogger
