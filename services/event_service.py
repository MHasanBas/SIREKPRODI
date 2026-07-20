import json
import os
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

EVENTS_PATH = os.path.join("configs", "events.json")
TZ_JKT = ZoneInfo("Asia/Jakarta")

def _now_iso() -> str:
    return datetime.now(TZ_JKT).isoformat(timespec="seconds")

def load_events() -> list:
    """Load all events from JSON file."""
    if not os.path.exists(EVENTS_PATH):
        return []
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []

def save_events(events: list) -> None:
    """Save all events to JSON file."""
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    with open(EVENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

def get_event_by_id(event_id: str) -> dict | None:
    """Get a single event by ID."""
    for evt in load_events():
        if evt.get("id") == event_id:
            return evt
    return None

def create_event(data: dict) -> dict:
    """Create a new event and save it."""
    events = load_events()
    tipe = data.get("tipe_sasaran", "sekolah")
    sekolah = [s.strip() for s in data.get("sekolah", []) if s.strip()] if tipe == "sekolah" else []
    prodi = [p.strip() for p in data.get("prodi", []) if p.strip()] if tipe == "prodi" else []
    
    new_event = {
        "id": f"evt_{uuid.uuid4().hex[:12]}",
        "nama": data.get("nama", "").strip(),
        "tanggal": data.get("tanggal", ""),
        "lokasi": data.get("lokasi", "").strip(),
        "keterangan": data.get("keterangan", "").strip(),
        "tipe_sasaran": tipe,
        "sekolah": sekolah,
        "prodi": prodi,
        "dibuat_pada": _now_iso(),
        "diperbarui_pada": _now_iso(),
    }
    events.append(new_event)
    save_events(events)
    return new_event

def update_event(event_id: str, data: dict) -> dict | None:
    """Update an existing event."""
    events = load_events()
    for i, evt in enumerate(events):
        if evt.get("id") == event_id:
            tipe = data.get("tipe_sasaran", evt.get("tipe_sasaran", "sekolah"))
            sekolah = [s.strip() for s in data.get("sekolah", []) if s.strip()] if tipe == "sekolah" else []
            prodi = [p.strip() for p in data.get("prodi", []) if p.strip()] if tipe == "prodi" else []
            
            events[i] = {
                **evt,
                "nama": data.get("nama", evt.get("nama", "")).strip(),
                "tanggal": data.get("tanggal", evt.get("tanggal", "")),
                "lokasi": data.get("lokasi", evt.get("lokasi", "")).strip(),
                "keterangan": data.get("keterangan", evt.get("keterangan", "")).strip(),
                "tipe_sasaran": tipe,
                "sekolah": sekolah,
                "prodi": prodi,
                "diperbarui_pada": _now_iso(),
            }
            save_events(events)
            return events[i]
    return None

def delete_event(event_id: str) -> bool:
    """Delete an event by ID."""
    events = load_events()
    original_count = len(events)
    events = [evt for evt in events if evt.get("id") != event_id]
    if len(events) < original_count:
        save_events(events)
        return True
    return False
