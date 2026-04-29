# data_collector.py

from datetime import datetime
from ml_engine.db_client import insert_feedback


def build_feedback_record(user_id, predicted, actual, error):
    return {
        "user_id": user_id,
        "predicted": float(predicted),
        "actual": float(actual),
        "error": float(error),
        "created_at": datetime.utcnow().isoformat()
    }


def store_feedback(record):
    return insert_feedback(record)
