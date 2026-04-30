"""
routes/settings.py – User settings endpoints (persistent JSON storage).
"""

import json
from pathlib import Path

from fastapi import APIRouter

from web.backend.schemas import UserSettings

router = APIRouter(prefix="/api/settings", tags=["settings"])

SETTINGS_FILE = Path("data/settings.json")


def _load() -> UserSettings:
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text())
        return UserSettings(**data)
    return UserSettings()


def _save(settings: UserSettings) -> None:
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings.model_dump(), indent=2))


@router.get("", response_model=UserSettings)
def get_settings() -> UserSettings:
    return _load()


@router.put("", response_model=UserSettings)
def update_settings(settings: UserSettings) -> UserSettings:
    _save(settings)
    return settings


@router.patch("", response_model=UserSettings)
def patch_settings(updates: dict) -> UserSettings:  # type: ignore[type-arg]
    current = _load()
    current_dict = current.model_dump()
    current_dict.update(updates)
    updated = UserSettings(**current_dict)
    _save(updated)
    return updated
