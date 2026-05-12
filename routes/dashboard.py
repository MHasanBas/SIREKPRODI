import os
import tempfile
import pickle
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file
import pandas as pd
import re

from services.model_service import load_active_model_name
from utils.cache import (
    _load_df_kmeans_cached, 
    _ensure_prestasi_cached, 
    _dash_cache_get, 
    _dash_cache_set
)
from services.dashboard_service import process_dashboard_data

dashboard_bp = Blueprint('dashboard', __name__)


def _get_prodi_column(df: pd.DataFrame):
    for kandidat in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
        if kandidat in df.columns:
            return kandidat
    return None


def _compute_label_to_cluster_map(df: pd.DataFrame):
    """Map label A/B/C -> numeric cluster id based on ranking used in dashboard_service."""
    if 'Cluster' not in df.columns:
        return {}

    cluster_summary = df.groupby('Cluster').agg(
        total_mahasiswa=('NILAI KESELURUHAN', 'count'),
        ipk_rata2=('NILAI KESELURUHAN', 'mean'),
        deviasi=('NILAI KESELURUHAN', 'std')
    ).reset_index()

    def _safe_norm(s: pd.Series):
        denom = (s.max() - s.min())
        if denom == 0:
            return s * 0
        return (s - s.min()) / denom

    cluster_summary['mahasiswa_norm'] = _safe_norm(cluster_summary['total_mahasiswa'])
    cluster_summary['ipk_norm'] = _safe_norm(cluster_summary['ipk_rata2'])
    cluster_summary['deviasi_norm'] = _safe_norm(cluster_summary['deviasi'])
    cluster_summary['ranking'] = (
        0.4 * cluster_summary['mahasiswa_norm'] +
        0.3 * cluster_summary['ipk_norm'] +
        0.3 * (1 - cluster_summary['deviasi_norm'])
    )

    ordered_clusters = cluster_summary.sort_values(by='ranking', ascending=False)['Cluster'].tolist()
    labels = ['A', 'B', 'C']
    if ordered_clusters:
        ordered_clusters = ordered_clusters + [ordered_clusters[-1]] * (len(labels) - len(ordered_clusters))

    return {label: int(ordered_clusters[i]) for i, label in enumerate(labels) if i < len(ordered_clusters)}

@dashboard_bp.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    tampilkan = request.args.get('tampilkan', 'top10')
    selected_prodi = request.args.get('prodi', 'all')

    active_model = load_active_model_name()
    model_path = f"models/{active_model}/hasil_kmeans_3cluster.pkl"
    if not os.path.exists(model_path):
        flash(" File hasil clustering tidak ditemukan. Silakan upload ulang data.")
        return redirect(url_for('upload.upload'))

    try:
        df_kmeans_full = _load_df_kmeans_cached(active_model)
        model_mtime = os.path.getmtime(model_path)
    except Exception:
        flash(" Gagal memuat file hasil clustering. Silakan upload ulang data.")
        return redirect(url_for('upload.upload'))

    _ensure_prestasi_cached(active_model, model_mtime)

    prodi_column = _get_prodi_column(df_kmeans_full)

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
        "total_sekolah": df_kmeans_full['ASAL SEKOLAH'].nunique() if 'ASAL SEKOLAH' in df_kmeans_full.columns else 0,
        "active_model": active_model,
        "data_rekomendasi": rekom_prodi_per_sekolah,
        "total_alumni": len(df_kmeans_full)
    }
    _dash_cache_set(cache_key, ctx)
    return render_template("dashboard.html", **ctx)


@dashboard_bp.route('/api/mahasiswa_detail')
def api_mahasiswa_detail():
    if 'user' not in session:
        return {"ok": False, "error": "unauthorized"}, 401

    active_model = load_active_model_name()
    try:
        df = _load_df_kmeans_cached(active_model).copy()
    except Exception as e:
        return {"ok": False, "error": f"Gagal memuat data: {e}"}, 500

    prodi_column = _get_prodi_column(df)
    sekolah = (request.args.get('sekolah') or '').strip()
    prodi = (request.args.get('prodi') or '').strip()
    cluster_label = (request.args.get('cluster') or 'all').strip().upper()

    if not sekolah:
        return {"ok": False, "error": "Parameter sekolah wajib diisi."}, 400
    if not prodi_column:
        return {"ok": False, "error": "Kolom prodi tidak ditemukan di dataset."}, 400

    if prodi.lower() == 'all' or not prodi:
        df = df[(df['ASAL SEKOLAH'] == sekolah)]
    else:
        df = df[(df['ASAL SEKOLAH'] == sekolah) & (df[prodi_column].astype(str) == prodi)]

    label_to_cluster = _compute_label_to_cluster_map(df) if not df.empty else {}
    if cluster_label in ('A', 'B', 'C') and 'Cluster' in df.columns:
        cluster_id = label_to_cluster.get(cluster_label)
        if cluster_id is not None:
            df = df[df['Cluster'] == cluster_id]

    try:
        limit = int(request.args.get('limit', '200'))
    except ValueError:
        limit = 200
    limit = max(1, min(limit, 1000))
    df = df.head(limit)

    preferred_cols = [
        'NIM', 'NAMA', 'JENIS KELAMIN', 'TAHUN MASUK',
        prodi_column, 'ASAL SEKOLAH',
        'NILAI KESELURUHAN', 'POIN_AKADEMIK', 'POIN_NON_AKADEMIK',
        'TEKS_AKADEMIK', 'TEKS_NON_AKADEMIK',
        'Cluster'
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        cols = df.columns.tolist()

    data = df[cols].to_dict(orient='records')
    return {
        "ok": True,
        "count": int(len(data)),
        "columns": cols,
        "rows": data,
        "label_to_cluster": label_to_cluster,
    }

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
    selected_label = request.args.get('label', 'all')
    
    prodi_column = 'PROGRAM STUDI' if 'PROGRAM STUDI' in df_kmeans.columns else None
    prodi_list = sorted(df_kmeans[prodi_column].dropna().unique().astype(str)) if prodi_column else []

    if prodi_column and selected_prodi != 'all':
        df_kmeans = df_kmeans[df_kmeans[prodi_column] == selected_prodi]

    df_kmeans_for_stats = df_kmeans.copy()

    sekolah_list = []
    if 'ASAL SEKOLAH' in df_kmeans.columns:
        sekolah_list = sorted(df_kmeans['ASAL SEKOLAH'].dropna().unique().astype(str))

    if search_sekolah and 'ASAL SEKOLAH' in df_kmeans.columns:
        escaped_search = re.escape(search_sekolah)
        pattern = r'\b' + escaped_search + r'\b'
        df_kmeans = df_kmeans[df_kmeans['ASAL SEKOLAH'].str.contains(pattern, case=False, na=False, regex=True)]

    from services.dashboard_service import get_crosscheck_data
    crosscheck_records = get_crosscheck_data(df_kmeans_for_stats)
    if prodi_column:
        crosscheck_map = {str(r.get('ASAL SEKOLAH', '')) + '_' + str(r.get(prodi_column, '')): r for r in crosscheck_records}
    else:
        crosscheck_map = {str(r.get('ASAL SEKOLAH', '')): r for r in crosscheck_records}

    label_to_cluster = _compute_label_to_cluster_map(df_kmeans_for_stats)
    cluster_to_label = {v: k for k, v in label_to_cluster.items()}

    cluster_options = []
    if 'Cluster' in df_kmeans.columns:
        try:
            cluster_options = sorted(df_kmeans['Cluster'].dropna().astype(int).unique().tolist())
        except Exception:
            cluster_options = sorted(df_kmeans['Cluster'].dropna().unique().tolist())

    if selected_label != 'all' and 'Cluster' in df_kmeans.columns:
        try:
            target_cluster_id = label_to_cluster.get(selected_label)
            if target_cluster_id is not None:
                df_kmeans = df_kmeans[df_kmeans['Cluster'] == target_cluster_id]
        except Exception:
            selected_label = 'all'

    # Compute stats for each label to show in cards
    algo_stats = {}
    for lab, cid in label_to_cluster.items():
        sub = df_kmeans_for_stats[df_kmeans_for_stats['Cluster'] == cid]
        algo_stats[lab] = {
            'total_mahasiswa': int(len(sub)),
            'ipk_mean': float(sub['NILAI KESELURUHAN'].mean()) if not sub.empty else 0.0
        }

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
        crosscheck_map=crosscheck_map,
        cluster_to_label=cluster_to_label,
        prodi_list=prodi_list,
        selected_prodi=selected_prodi,
        cluster_options=cluster_options,
        selected_label=selected_label,
        algo_stats=algo_stats,
        labels=['A', 'B', 'C'],
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
        escaped_search = re.escape(search_sekolah)
        pattern = r'\b' + escaped_search + r'\b'
        df = df[df['ASAL SEKOLAH'].str.contains(pattern, case=False, na=False, regex=True)]

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

@dashboard_bp.route('/download_rekomendasi')
def download_rekomendasi():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    active_model = load_active_model_name()
    tampilkan = request.args.get('tampilkan', 'top10')
    selected_prodi = request.args.get('prodi', 'all')
    
    try:
        df_kmeans_full = _load_df_kmeans_cached(active_model)
    except Exception:
        flash("Gagal memuat file hasil clustering.", "error")
        return redirect(url_for('dashboard.dashboard'))
        
    prodi_column = _get_prodi_column(df_kmeans_full)
    df_kmeans = df_kmeans_full
    if prodi_column and selected_prodi != 'all':
        df_kmeans = df_kmeans_full[df_kmeans_full[prodi_column] == selected_prodi].copy()
        
    rekom_prodi_per_sekolah, _, _ = process_dashboard_data(df_kmeans, prodi_column, tampilkan)
    
    # Flatten data for Excel
    flat_data = []
    for row in rekom_prodi_per_sekolah:
        sekolah = row['sekolah']
        kota = row['kota']
        for p in row['top_prodi']:
            flat_data.append({
                'Asal Sekolah': sekolah,
                'Kota': kota,
                'Program Studi Rekomendasi': p['prodi'],
                'Rata-rata IPK': p['ipk'],
                'Jumlah Mahasiswa': p['mahasiswa'],
                'Poin Akademik': p.get('poin_akademik', 0),
                'Poin Non-Akademik': p.get('poin_non_akademik', 0)
            })
            
    df_out = pd.DataFrame(flat_data)
    
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    df_out.to_excel(path, index=False)
    
    filename = f"Rekomendasi_Promosi_{selected_prodi}.xlsx" if selected_prodi != 'all' else "Rekomendasi_Promosi_Semua.xlsx"
    return send_file(path, as_attachment=True, download_name=filename)
