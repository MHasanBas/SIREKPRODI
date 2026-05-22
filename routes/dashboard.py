import os
import tempfile
import pickle
import json
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
from utils.helpers import format_datetime_jakarta
from services.dashboard_service import (
    IPK_TINGGI_THRESHOLD,
    build_cluster_academic_summary,
    ordered_academic_clusters,
    process_dashboard_data,
)

dashboard_bp = Blueprint('dashboard', __name__)


def _load_model_meta(model_name):
    if not model_name:
        return {}
    meta_path = os.path.join("models", model_name, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        meta["uploaded_at_display"] = format_datetime_jakarta(meta.get("uploaded_at"))
        return meta
    except Exception:
        return {}


def _get_prodi_column(df: pd.DataFrame):
    for kandidat in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
        if kandidat in df.columns:
            return kandidat
    return None


def _compute_label_to_cluster_map(df: pd.DataFrame):
    """Map label A/B/C -> numeric cluster id based on academic ranking used in dashboard_service."""
    if 'Cluster' not in df.columns:
        return {}

    ordered_clusters = ordered_academic_clusters(df)
    labels = ['A', 'B', 'C']
    if ordered_clusters:
        ordered_clusters = ordered_clusters + [ordered_clusters[-1]] * (len(labels) - len(ordered_clusters))

    return {label: int(ordered_clusters[i]) for i, label in enumerate(labels) if i < len(ordered_clusters)}


def _filter_school_search(df: pd.DataFrame, search_value: str) -> pd.DataFrame:
    if not search_value or 'ASAL SEKOLAH' not in df.columns:
        return df

    terms = [term for term in re.split(r'\s+', search_value.strip()) if term]
    if not terms:
        return df

    school_series = df['ASAL SEKOLAH'].astype(str)
    mask = pd.Series(True, index=df.index)
    for term in terms:
        escaped_term = re.escape(term)
        if term.isdigit():
            pattern = rf'(?<!\d){escaped_term}(?!\d)'
        else:
            pattern = escaped_term
        mask &= school_series.str.contains(pattern, case=False, na=False, regex=True)
    return df[mask]

@dashboard_bp.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    tampilkan = request.args.get('tampilkan', 'top10')
    selected_prodi = request.args.get('prodi', 'all')

    active_model = load_active_model_name()
    active_model_meta = _load_model_meta(active_model)
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
            flash(f"Data untuk prodi {selected_prodi} tidak ditemukan.", "warning")
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
        "active_model_meta": active_model_meta,
        "data_rekomendasi": rekom_prodi_per_sekolah,
        "total_alumni": len(df_kmeans_full),
        "filtered_total": len(df_kmeans),
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
        df_full_for_map = df.copy() # Simpan copy untuk mapping prioritas
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

    # Hitung label_to_cluster berdasarkan prodi secara global (bukan per sekolah) agar konsisten
    if prodi_column and prodi and prodi.lower() != 'all':
        df_prodi = df_full_for_map[df_full_for_map[prodi_column].astype(str) == prodi]
        label_to_cluster = _compute_label_to_cluster_map(df_prodi) if not df_prodi.empty else {}
    else:
        label_to_cluster = _compute_label_to_cluster_map(df_full_for_map) if not df_full_for_map.empty else {}
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
    active_model_meta = _load_model_meta(active_model)
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
    limit_rows = request.args.get('limit', '100')
    page_param = request.args.get('page', '1')
    search_sekolah = request.args.get('search', '').strip()
    selected_cluster = request.args.get('cluster', 'all')
    selected_label = request.args.get('label', 'all')
    selected_status = request.args.get('status', 'all')
    try:
        per_page = int(limit_rows)
    except (TypeError, ValueError):
        per_page = 100
        limit_rows = '100'
    per_page = min(max(per_page, 25), 500)
    limit_rows = str(per_page)

    try:
        current_page = int(page_param)
    except (TypeError, ValueError):
        current_page = 1
    current_page = max(current_page, 1)
    
    prodi_column = _get_prodi_column(df_kmeans)
    prodi_list = sorted(df_kmeans[prodi_column].dropna().unique().astype(str)) if prodi_column else []

    # === Hitung pemetaan label cluster berdasarkan prodi yang dipilih (lokal) agar konsisten ===
    if prodi_column and selected_prodi != 'all':
        df_kmeans_prodi = df_kmeans[df_kmeans[prodi_column] == selected_prodi]
        label_to_cluster = _compute_label_to_cluster_map(df_kmeans_prodi)
    else:
        label_to_cluster = _compute_label_to_cluster_map(df_kmeans)
    cluster_to_label = {v: k for k, v in label_to_cluster.items()}

    from services.dashboard_service import get_crosscheck_data
    crosscheck_records = get_crosscheck_data(df_kmeans)
    if prodi_column:
        crosscheck_map = {str(r.get('ASAL SEKOLAH', '')) + '_' + str(r.get(prodi_column, '')): r for r in crosscheck_records}
    else:
        crosscheck_map = {str(r.get('ASAL SEKOLAH', '')): r for r in crosscheck_records}

    # === Filter prodi dan pencarian sekolah baru dijalankan setelah pemetaan global siap ===
    if prodi_column and selected_prodi != 'all':
        df_kmeans = df_kmeans[df_kmeans[prodi_column] == selected_prodi]

    df_kmeans_for_stats = df_kmeans.copy()

    sekolah_list = []
    if 'ASAL SEKOLAH' in df_kmeans.columns:
        sekolah_list = sorted(df_kmeans['ASAL SEKOLAH'].dropna().unique().astype(str))

    df_kmeans = _filter_school_search(df_kmeans, search_sekolah)

    # Filter by Lolos / Gagal status
    if selected_status != 'all' and not df_kmeans.empty:
        def match_status(row):
            key = str(row.get('ASAL SEKOLAH', '')) + '_' + str(row.get(prodi_column, '')) if prodi_column else str(row.get('ASAL SEKOLAH', ''))
            stat = crosscheck_map.get(key, {})
            is_lolos = (stat.get('status', '') == 'Masuk Rekomendasi')
            if selected_status == 'lolos':
                return is_lolos
            elif selected_status == 'gagal':
                return not is_lolos
            return True
            
        mask = df_kmeans.apply(match_status, axis=1)
        df_kmeans = df_kmeans[mask]

    if selected_cluster == 'all' and selected_label != 'all':
        mapped_cluster = label_to_cluster.get(selected_label)
        if mapped_cluster is not None:
            selected_cluster = str(mapped_cluster)

    cluster_options = []
    if 'Cluster' in df_kmeans_for_stats.columns:
        try:
            cluster_options = sorted(df_kmeans_for_stats['Cluster'].dropna().astype(int).unique().tolist())
        except Exception:
            cluster_options = sorted(df_kmeans_for_stats['Cluster'].dropna().unique().tolist())

    if selected_cluster != 'all' and 'Cluster' in df_kmeans.columns:
        try:
            cluster_id = int(selected_cluster)
            df_kmeans = df_kmeans[df_kmeans['Cluster'] == cluster_id]
        except Exception:
            selected_cluster = 'all'

    # Compute stats for each academic priority label to preserve interpretation context.
    algo_stats = {}
    academic_summary = build_cluster_academic_summary(df_kmeans_for_stats)
    for lab, cid in label_to_cluster.items():
        sub_summary = academic_summary[academic_summary['Cluster'] == cid]
        if sub_summary.empty:
            algo_stats[lab] = {
                'total_mahasiswa': 0,
                'ipk_mean': 0.0,
                'ipk_median': 0.0,
                'ipk_min': 0.0,
                'ipk_max': 0.0,
                'ipk_tinggi_count': 0,
                'ipk_tinggi_pct': 0.0,
                'ipk_tinggi_threshold': IPK_TINGGI_THRESHOLD,
            }
        else:
            row = sub_summary.iloc[0]
            algo_stats[lab] = {
                'total_mahasiswa': int(row['total_mahasiswa']),
                'ipk_mean': float(row['ipk_rata2']),
                'ipk_median': float(row['ipk_median']),
                'ipk_min': float(row['ipk_min']),
                'ipk_max': float(row['ipk_max']),
                'ipk_tinggi_count': int(row['ipk_tinggi_count']),
                'ipk_tinggi_pct': float(row['ipk_tinggi_pct']),
                'ipk_tinggi_threshold': IPK_TINGGI_THRESHOLD,
            }

    raw_cluster_summary = []
    if 'Cluster' in df_kmeans_for_stats.columns and 'NILAI KESELURUHAN' in df_kmeans_for_stats.columns:
        cluster_summary_df = df_kmeans_for_stats.groupby('Cluster').agg(
            total_mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk_mean=('NILAI KESELURUHAN', 'mean'),
            ipk_median=('NILAI KESELURUHAN', 'median'),
            ipk_min=('NILAI KESELURUHAN', 'min'),
            ipk_max=('NILAI KESELURUHAN', 'max'),
        ).reset_index()
        high_counts = df_kmeans_for_stats.assign(
            ipk_tinggi=df_kmeans_for_stats['NILAI KESELURUHAN'] >= IPK_TINGGI_THRESHOLD
        ).groupby('Cluster')['ipk_tinggi'].sum()
        top_prodi_map = {}
        if prodi_column:
            for cid, grp in df_kmeans_for_stats.groupby('Cluster'):
                top_prodi_map[cid] = ", ".join(grp[prodi_column].value_counts().head(3).index.astype(str).tolist())

        for _, row in cluster_summary_df.sort_values('Cluster').iterrows():
            cid = row['Cluster']
            total = int(row['total_mahasiswa'])
            ipk_tinggi_count = int(high_counts.get(cid, 0))
            raw_cluster_summary.append({
                'cluster': int(cid) if pd.notna(cid) else cid,
                'prioritas': cluster_to_label.get(int(cid) if pd.notna(cid) else cid, '-'),
                'total_mahasiswa': total,
                'ipk_mean': float(row['ipk_mean']),
                'ipk_median': float(row['ipk_median']),
                'ipk_min': float(row['ipk_min']),
                'ipk_max': float(row['ipk_max']),
                'ipk_tinggi_count': ipk_tinggi_count,
                'ipk_tinggi_pct': (ipk_tinggi_count / total * 100) if total else 0.0,
                'top_prodi': top_prodi_map.get(cid, '-'),
            })

    selected_cluster_stats = None
    if selected_cluster != 'all':
        selected_cluster_stats = next(
            (row for row in raw_cluster_summary if str(row.get('cluster')) == str(selected_cluster)),
            None
        )

    filtered_total_before_limit = len(df_kmeans)
    total_pages = max(1, (filtered_total_before_limit + per_page - 1) // per_page)
    current_page = min(current_page, total_pages)
    page_start = (current_page - 1) * per_page
    page_end = page_start + per_page
    df_kmeans = df_kmeans.iloc[page_start:page_end]

    records = df_kmeans.to_dict(orient='records')
    displayed_total = len(df_kmeans)
    
    return render_template(
        "data_cluster.html",
        records=records,
        crosscheck_map=crosscheck_map,
        cluster_to_label=cluster_to_label,
        prodi_list=prodi_list,
        selected_prodi=selected_prodi,
        cluster_options=cluster_options,
        selected_cluster=selected_cluster,
        selected_label=selected_label,
        algo_stats=algo_stats,
        raw_cluster_summary=raw_cluster_summary,
        selected_cluster_stats=selected_cluster_stats,
        ipk_tinggi_threshold=IPK_TINGGI_THRESHOLD,
        labels=['A', 'B', 'C'],
        limit_rows=limit_rows,
        search_sekolah=search_sekolah,
        sekolah_list=sekolah_list,
        active_model=active_model,
        active_model_meta=active_model_meta,
        selected_status=selected_status,
        filtered_total=filtered_total_before_limit,
        displayed_total=displayed_total,
        total_rows=len(df_kmeans_for_stats),
        current_page=current_page,
        total_pages=total_pages,
        per_page=per_page,
        page_start=page_start + 1 if filtered_total_before_limit else 0,
        page_end=min(page_end, filtered_total_before_limit),
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
    selected_label = request.args.get('label', 'all')
    search_sekolah = request.args.get('search', '').strip()
    selected_status = request.args.get('status', 'all')
    
    # === FIX: Simpan copy global untuk perhitungan crosscheck & mapping prioritas yang konsisten ===
    df_full = df.copy()
    
    # === Hitung pemetaan label cluster berdasarkan prodi yang dipilih (lokal) agar konsisten ===
    prodi_column = _get_prodi_column(df)

    if prodi_column and selected_prodi != 'all':
        df_prodi = df[df[prodi_column] == selected_prodi]
        label_to_cluster = _compute_label_to_cluster_map(df_prodi)
    else:
        label_to_cluster = _compute_label_to_cluster_map(df_full)
    cluster_to_label = {v: k for k, v in label_to_cluster.items()}

    if prodi_column and selected_prodi != 'all':
        df = df[df[prodi_column] == selected_prodi]

    df = _filter_school_search(df, search_sekolah)

    # Filter by Lolos / Gagal status
    if selected_status != 'all' and not df.empty:
        from services.dashboard_service import get_crosscheck_data
        crosscheck_records = get_crosscheck_data(df_full)
        if prodi_column:
            crosscheck_map = {str(r.get('ASAL SEKOLAH', '')) + '_' + str(r.get(prodi_column, '')): r for r in crosscheck_records}
        else:
            crosscheck_map = {str(r.get('ASAL SEKOLAH', '')): r for r in crosscheck_records}

        def match_status(row):
            key = str(row.get('ASAL SEKOLAH', '')) + '_' + str(row.get(prodi_column, '')) if prodi_column else str(row.get('ASAL SEKOLAH', ''))
            stat = crosscheck_map.get(key, {})
            is_lolos = (stat.get('status', '') == 'Masuk Rekomendasi')
            if selected_status == 'lolos':
                return is_lolos
            elif selected_status == 'gagal':
                return not is_lolos
            return True
            
        mask = df.apply(match_status, axis=1)
        df = df[mask]

    if 'Cluster' in df.columns:
        df = df.copy()
        df['PRIORITAS AKADEMIK'] = df['Cluster'].map(cluster_to_label).fillna('-')

    if selected_label != 'all' and 'Cluster' in df.columns:
        target_cluster_id = label_to_cluster.get(selected_label)
        if target_cluster_id is not None:
            df = df[df['Cluster'] == target_cluster_id]
    elif selected_cluster != 'all' and 'Cluster' in df.columns:
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
        model_path = f"models/{active_model}/hasil_kmeans_3cluster.pkl"
        _ensure_prestasi_cached(active_model, os.path.getmtime(model_path))
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
