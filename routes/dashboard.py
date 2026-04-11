import os
import tempfile
import pickle
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file
import pandas as pd

from services.model_service import load_active_model_name
from utils.cache import (
    _load_df_kmeans_cached, 
    _ensure_prestasi_cached, 
    _dash_cache_get, 
    _dash_cache_set
)
from services.dashboard_service import process_dashboard_data

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    tampilkan = request.args.get('tampilkan', 'top10')  # default: top10
    selected_prodi = request.args.get('prodi', 'all')

    active_model = load_active_model_name()
    model_path = f"models/{active_model}/hasil_kmeans_3cluster.pkl"
    if not os.path.exists(model_path):
        flash("❌ File hasil clustering tidak ditemukan. Silakan upload ulang data.")
        return redirect(url_for('upload.upload'))

    try:
        df_kmeans_full = _load_df_kmeans_cached(active_model)
        model_mtime = os.path.getmtime(model_path)
    except Exception:
        flash("❌ Gagal memuat file hasil clustering. Silakan upload ulang data.")
        return redirect(url_for('upload.upload'))

    _ensure_prestasi_cached(active_model, model_mtime)

    prodi_column = None
    for kandidat in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
        if kandidat in df_kmeans_full.columns:
            prodi_column = kandidat
            break

    prodi_list = sorted(df_kmeans_full[prodi_column].dropna().unique().tolist()) if prodi_column else []

    cache_key = (active_model, model_mtime, selected_prodi, tampilkan)
    cached_ctx = _dash_cache_get(cache_key)
    if cached_ctx is not None:
        return render_template("dashboard.html", **cached_ctx)

    df_kmeans = df_kmeans_full
    if prodi_column and selected_prodi != 'all':
        df_kmeans = df_kmeans_full[df_kmeans_full[prodi_column] == selected_prodi].copy()
        if df_kmeans.empty:
            flash(f"⚠️ Data untuk prodi {selected_prodi} tidak ditemukan.", "warning")
            return redirect(url_for('dashboard.dashboard'))

    rekom_prodi_per_sekolah, rekom_sekolah_per_prodi, kmeans_data = process_dashboard_data(
        df_kmeans, prodi_column, tampilkan
    )

    ctx = {
        "kmeans_data": kmeans_data,
        "prodi_list": prodi_list,
        "selected_prodi": selected_prodi,
        "rekom_prodi_per_sekolah": rekom_prodi_per_sekolah,
        "rekom_sekolah_per_prodi": rekom_sekolah_per_prodi,
        "prodi_column_available": bool(prodi_column),
    }
    _dash_cache_set(cache_key, ctx)
    return render_template("dashboard.html", **ctx)

@dashboard_bp.route('/data_cluster')
def data_cluster():
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    active_model = load_active_model_name()
    model_path = f"models/{active_model}/hasil_kmeans_3cluster.pkl"
    if not os.path.exists(model_path):
        flash("Silakan upload data terlebih dahulu untuk diproses.", "warning")
        return redirect(url_for('upload.upload'))

    try:
        df_kmeans = _load_df_kmeans_cached(active_model).copy()
    except Exception as e:
        flash(f"Gagal memuat model aktif ({active_model}): {e}", "danger")
        return redirect(url_for('upload.upload'))

    selected_prodi = request.args.get('prodi', 'all')
    limit_rows = request.args.get('limit', '500')
    search_sekolah = request.args.get('search', '').strip()
    selected_cluster = request.args.get('cluster', 'all')
    
    prodi_column = 'PROGRAM STUDI' if 'PROGRAM STUDI' in df_kmeans.columns else None
    prodi_list = sorted(df_kmeans[prodi_column].dropna().unique().astype(str)) if prodi_column else []

    if prodi_column and selected_prodi != 'all':
        df_kmeans = df_kmeans[df_kmeans[prodi_column] == selected_prodi]

    sekolah_list = []
    if 'ASAL SEKOLAH' in df_kmeans.columns:
        sekolah_list = sorted(df_kmeans['ASAL SEKOLAH'].dropna().unique().astype(str))

    if search_sekolah and 'ASAL SEKOLAH' in df_kmeans.columns:
        df_kmeans = df_kmeans[df_kmeans['ASAL SEKOLAH'].str.contains(search_sekolah, case=False, na=False)]

    cluster_options = []
    if 'Cluster' in df_kmeans.columns:
        try:
            cluster_options = sorted(df_kmeans['Cluster'].dropna().astype(int).unique().tolist())
        except Exception:
            cluster_options = sorted(df_kmeans['Cluster'].dropna().unique().tolist())

    if selected_cluster != 'all' and 'Cluster' in df_kmeans.columns:
        try:
            cluster_id = int(selected_cluster)
            df_kmeans = df_kmeans[df_kmeans['Cluster'] == cluster_id]
        except ValueError:
            selected_cluster = 'all'

    if limit_rows != 'all':
        try:
            lim = int(limit_rows)
            df_kmeans = df_kmeans.head(lim)
        except ValueError:
            pass

    records = df_kmeans.to_dict(orient='records')
    
    return render_template(
        "data_cluster.html",
        records=records,
        prodi_list=prodi_list,
        selected_prodi=selected_prodi,
        cluster_options=cluster_options,
        selected_cluster=selected_cluster,
        limit_rows=limit_rows,
        search_sekolah=search_sekolah,
        sekolah_list=sekolah_list
    )

@dashboard_bp.route('/download_cluster')
def download_cluster():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    active_flag_path = "models/active_model.txt"
    if not os.path.exists(active_flag_path):
        return redirect(url_for('upload.upload'))
        
    with open(active_flag_path, "r") as f:
        active_model = f.read().strip()
        
    try:
        model_path = f"models/{active_model}/hasil_kmeans_3cluster.pkl"
        with open(model_path, 'rb') as f:
            data_dict = pickle.load(f)
            df = data_dict['data']
    except Exception:
        return redirect(url_for('dashboard.data_cluster'))

    selected_prodi = request.args.get('prodi', 'all')
    selected_cluster = request.args.get('cluster', 'all')
    search_sekolah = request.args.get('search', '').strip()
    
    if 'PROGRAM STUDI' in df.columns and selected_prodi != 'all':
        df = df[df['PROGRAM STUDI'] == selected_prodi]

    if search_sekolah and 'ASAL SEKOLAH' in df.columns:
        df = df[df['ASAL SEKOLAH'].str.contains(search_sekolah, case=False, na=False)]

    if selected_cluster != 'all' and 'Cluster' in df.columns:
        try:
            cluster_id = int(selected_cluster)
            df = df[df['Cluster'] == cluster_id]
        except ValueError:
            pass

    import tempfile
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    df.to_excel(path, index=False)
    return send_file(path, as_attachment=True, download_name="Data_Cluster_SIREKPRODI.xlsx")
