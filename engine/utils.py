"""
utils.py – Shared utility functions used across the engine and interface layers.
"""

from datetime import date, datetime, timedelta


def period_to_start_date(period: str) -> date:
    """Convert a period string to the earliest date to include."""
    today = datetime.now().date()
    mapping = {
        "1mo": today - timedelta(days=30),
        "1y": today - timedelta(days=365),
        "2y": today - timedelta(days=730),
        "5y": today - timedelta(days=1825),
        "max": date(1900, 1, 1),
    }
    return mapping.get(period, today - timedelta(days=365))
