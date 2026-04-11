import os
import pickle
import pandas as pd
from collections import OrderedDict
from threading import RLock
from utils.helpers import _ensure_prestasi_columns

_CACHE_LOCK = RLock()
_DF_CACHE = {}  # {model: {"mtime": float, "df": DataFrame, "prestasi_ready": bool}}
_DASH_CACHE = OrderedDict()  # {(model, mtime, selected_prodi, tampilkan): context_dict}
_DASH_CACHE_MAX = 12

def _load_df_kmeans_cached(active_model: str) -> pd.DataFrame:
    model_path = f"models/{active_model}/hasil_kmeans_3cluster.pkl"
    mtime = os.path.getmtime(model_path)

    with _CACHE_LOCK:
        cached = _DF_CACHE.get(active_model)
        if cached and cached.get("mtime") == mtime:
            return cached["df"]

    with open(model_path, "rb") as f:
        hasil_kmeans = pickle.load(f)
        df_kmeans = hasil_kmeans["data"] if isinstance(hasil_kmeans, dict) else hasil_kmeans

    # STRIP SEMUA KOLOM untuk buang spasi tak terlihat
    df_kmeans.columns = df_kmeans.columns.str.strip()

    with _CACHE_LOCK:
        _DF_CACHE[active_model] = {"mtime": mtime, "df": df_kmeans, "prestasi_ready": False}
    return df_kmeans

def _ensure_prestasi_cached(active_model: str, expected_mtime: float) -> None:
    # Hitung kolom prestasi hanya saat dibutuhkan (dashboard).
    with _CACHE_LOCK:
        entry = _DF_CACHE.get(active_model)
        if not entry or entry.get("mtime") != expected_mtime:
            return
        if entry.get("prestasi_ready"):
            return
        df_ref = entry["df"]

    _ensure_prestasi_columns(df_ref)

    with _CACHE_LOCK:
        entry = _DF_CACHE.get(active_model)
        if entry and entry.get("mtime") == expected_mtime:
            entry["prestasi_ready"] = True

def _dash_cache_get(key):
    with _CACHE_LOCK:
        value = _DASH_CACHE.get(key)
        if value is not None:
            # refresh LRU order
            _DASH_CACHE.move_to_end(key)
        return value

def _dash_cache_set(key, value):
    with _CACHE_LOCK:
        _DASH_CACHE[key] = value
        _DASH_CACHE.move_to_end(key)
        while len(_DASH_CACHE) > _DASH_CACHE_MAX:
            _DASH_CACHE.popitem(last=False)
