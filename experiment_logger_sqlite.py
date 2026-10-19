import os
import sqlite3
import dataset
from experiment_logger_base import BaseExperimentDB, BaseRunLogger

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB")
DB_FILE = "results.db"
DB_PATH = os.path.join(DB_DIR, DB_FILE)
SCHEMA_PATH = os.path.join(DB_DIR, "database.sql")
DB_URL = f"sqlite:///{DB_PATH}"


def init_db(path: str = DB_PATH, schema_path: str = SCHEMA_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(schema_path):
            print(f"Warning: Schema file {schema_path} not found, creating minimal schema")
            return
        with sqlite3.connect(path) as conn, open(schema_path, "r") as f:
            conn.executescript(f.read())
        print("DB has been created/initialized.")
    except Exception as e:
        print(f"DB initialization failed: {e}")
        raise


class ExperimentDB(BaseExperimentDB):

    def __init__(self, url: str = DB_URL):
        super().__init__(url)

    def _init_db(self):
        init_db()


RunLogger = BaseRunLogger


# ---------------------------------------------------------------------------
# Legacy module-level helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def log_eval(experiment_meta, eval_stats, eval_runs, convergence_results=None):
    """Log experiment + eval results + eval runs to DB."""
    init_db()
    db = dataset.connect(DB_URL)

    experiment_id = db["experiments"].insert(experiment_meta)
    eval_id = db["eval_results"].insert({
        'experiment_id': experiment_id,
        'episode': None,
        **eval_stats,
    })
    for i, run in enumerate(eval_runs):
        db["eval_runs"].insert({'eval_id': eval_id, 'run_index': i, **run})
    if convergence_results:
        for agent, stats in convergence_results.items():
            db["convergence_results"].insert({
                'eval_id': eval_id,
                'agent_name': agent,
                **stats,
            })
    return eval_id
