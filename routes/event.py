import os
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd

from services.event_service import load_events, create_event, update_event, delete_event, get_event_by_id
from services.model_service import load_active_model_name
from utils.cache import _load_df_kmeans_cached

event_bp = Blueprint('event', __name__)

def _get_sekolah_list() -> list[str]:
    """Retrieve list of schools from active model data."""
    try:
        active_model = load_active_model_name()
        if not active_model:
            return []
        df = _load_df_kmeans_cached(active_model)
        if 'ASAL SEKOLAH' in df.columns:
            return sorted(df['ASAL SEKOLAH'].dropna().unique().astype(str).tolist())
    except Exception:
        pass
    return []

def _get_prodi_list() -> list[str]:
    """Retrieve list of study programs (prodi) from active model data."""
    try:
        active_model = load_active_model_name()
        if not active_model:
            return []
        df = _load_df_kmeans_cached(active_model)
        for col in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
            if col in df.columns:
                return sorted(df[col].dropna().unique().astype(str).tolist())
    except Exception:
        pass
    return []

@event_bp.route('/event')
def event_list():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    events = load_events()
    sekolah_list = _get_sekolah_list()
    prodi_list = _get_prodi_list()
    
    active_model = load_active_model_name()
    df = None
    prodi_col = None
    if active_model:
        try:
            df = _load_df_kmeans_cached(active_model)
            for col in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
                if col in df.columns:
                    prodi_col = col
                    break
        except Exception:
            pass

    for evt in events:
        if 'tipe_sasaran' not in evt:
            if evt.get('sekolah') and not evt.get('prodi'):
                evt['tipe_sasaran'] = 'sekolah'
            elif evt.get('prodi') and not evt.get('sekolah'):
                evt['tipe_sasaran'] = 'prodi'
            else:
                evt['tipe_sasaran'] = 'sekolah'

        evt['rekomendasi_prodi'] = []
        evt['rekomendasi_sekolah'] = []

        if df is not None and prodi_col:
            if evt['tipe_sasaran'] == 'sekolah' and evt.get('sekolah'):
                sub_df = df[df['ASAL SEKOLAH'].isin(evt['sekolah'])]
                if not sub_df.empty:
                    agg = sub_df.groupby(prodi_col).agg(
                        mahasiswa=('NILAI KESELURUHAN', 'count'),
                        ipk_mean=('NILAI KESELURUHAN', 'mean')
                    ).fillna(0).reset_index()
                    max_mhs = agg['mahasiswa'].max() or 1
                    agg['skor'] = 0.7 * agg['ipk_mean'] + 0.3 * (agg['mahasiswa'] / max_mhs * 4.0)
                    agg = agg.sort_values('skor', ascending=False).head(3)
                    for _, row in agg.iterrows():
                        evt['rekomendasi_prodi'].append({
                            'prodi': str(row[prodi_col]),
                            'ipk': round(float(row['ipk_mean']), 2),
                            'mahasiswa': int(row['mahasiswa'])
                        })
            elif evt['tipe_sasaran'] == 'prodi' and evt.get('prodi'):
                sub_df = df[df[prodi_col].isin(evt['prodi'])]
                if not sub_df.empty:
                    agg = sub_df.groupby('ASAL SEKOLAH').agg(
                        mahasiswa=('NILAI KESELURUHAN', 'count'),
                        ipk_mean=('NILAI KESELURUHAN', 'mean'),
                        kota=('KOTA SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                    ).fillna(0).reset_index()
                    max_mhs = agg['mahasiswa'].max() or 1
                    agg['skor'] = 0.7 * agg['ipk_mean'] + 0.3 * (agg['mahasiswa'] / max_mhs * 4.0)
                    agg = agg.sort_values('skor', ascending=False).head(3)
                    for _, row in agg.iterrows():
                        evt['rekomendasi_sekolah'].append({
                            'sekolah': str(row['ASAL SEKOLAH']),
                            'kota': str(row['kota']),
                            'ipk': round(float(row['ipk_mean']), 2),
                            'mahasiswa': int(row['mahasiswa'])
                        })
    
    events_sorted = sorted(
        events,
        key=lambda e: (e.get('tanggal', ''), e.get('dibuat_pada', '')),
        reverse=True
    )
    
    return render_template(
        'event.html',
        events=events_sorted,
        sekolah_list=sekolah_list,
        prodi_list=prodi_list,
        total_events=len(events)
    )

@event_bp.route('/event/tambah', methods=['POST'])
def event_tambah():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    nama = request.form.get('nama', '').strip()
    if not nama:
        flash('Nama event wajib diisi.', 'error')
        return redirect(url_for('event.event_list'))
        
    tipe_sasaran = request.form.get('tipe_sasaran', 'sekolah')
    sekolah_list = request.form.getlist('sekolah')
    prodi_list = request.form.getlist('prodi')
    
    data = {
        'nama': nama,
        'tanggal': request.form.get('tanggal', ''),
        'lokasi': request.form.get('lokasi', ''),
        'keterangan': request.form.get('keterangan', ''),
        'tipe_sasaran': tipe_sasaran,
        'sekolah': sekolah_list,
        'prodi': prodi_list
    }
    
    create_event(data)
    flash(f'Event "{nama}" berhasil ditambahkan.', 'success')
    return redirect(url_for('event.event_list'))

@event_bp.route('/event/edit/<event_id>', methods=['POST'])
def event_edit(event_id):
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    nama = request.form.get('nama', '').strip()
    if not nama:
        flash('Nama event wajib diisi.', 'error')
        return redirect(url_for('event.event_list'))
        
    tipe_sasaran = request.form.get('tipe_sasaran', 'sekolah')
    sekolah_list = request.form.getlist('sekolah')
    prodi_list = request.form.getlist('prodi')
    
    data = {
        'nama': nama,
        'tanggal': request.form.get('tanggal', ''),
        'lokasi': request.form.get('lokasi', ''),
        'keterangan': request.form.get('keterangan', ''),
        'tipe_sasaran': tipe_sasaran,
        'sekolah': sekolah_list,
        'prodi': prodi_list
    }
    
    result = update_event(event_id, data)
    if result:
        flash(f'Event "{nama}" berhasil diperbarui.', 'success')
    else:
        flash('Event tidak ditemukan.', 'error')
    return redirect(url_for('event.event_list'))

@event_bp.route('/event/hapus/<event_id>', methods=['POST'])
def event_hapus(event_id):
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    evt = get_event_by_id(event_id)
    if evt:
        nama = evt.get('nama', 'Event')
        delete_event(event_id)
        flash(f'Event "{nama}" berhasil dihapus.', 'success')
    else:
        flash('Event tidak ditemukan.', 'error')
    return redirect(url_for('event.event_list'))


@event_bp.route('/api/event/recommendations', methods=['POST'])
def api_event_recommendations():
    if 'user' not in session:
        return jsonify({'ok': False, 'error': 'unauthorized'}), 401
    
    req_data = request.get_json() or {}
    selected_sekolah = req_data.get('sekolah', [])
    selected_prodi = req_data.get('prodi', [])
    
    active_model = load_active_model_name()
    if not active_model:
        return jsonify({'ok': True, 'prodi_recommendations': [], 'sekolah_recommendations': []})
        
    try:
        df = _load_df_kmeans_cached(active_model)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
        
    prodi_col = None
    for col in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
        if col in df.columns:
            prodi_col = col
            break
            
    prodi_recom = []
    sekolah_recom = []
    
    # 1. If sekolah selected, find recommended prodis
    if selected_sekolah and prodi_col:
        sub_df = df[df['ASAL SEKOLAH'].isin(selected_sekolah)]
        if not sub_df.empty:
            agg = sub_df.groupby(prodi_col).agg(
                mahasiswa=('NILAI KESELURUHAN', 'count'),
                ipk_mean=('NILAI KESELURUHAN', 'mean'),
                poin_akademik=('POIN_AKADEMIK', 'sum'),
                poin_non_akademik=('POIN_NON_AKADEMIK', 'sum')
            ).fillna(0).reset_index()
            
            max_mhs = agg['mahasiswa'].max() or 1
            agg['skor'] = 0.7 * agg['ipk_mean'] + 0.3 * (agg['mahasiswa'] / max_mhs * 4.0)
            agg = agg.sort_values('skor', ascending=False)
            
            for _, row in agg.iterrows():
                prodi_recom.append({
                    'prodi': str(row[prodi_col]),
                    'mahasiswa': int(row['mahasiswa']),
                    'ipk': round(float(row['ipk_mean']), 2),
                    'poin_akademik': int(row['poin_akademik']),
                    'poin_non_akademik': int(row['poin_non_akademik']),
                    'skor': round(float(row['skor']), 2)
                })
                
    # 2. If prodi selected, find recommended schools
    if selected_prodi and prodi_col:
        sub_df = df[df[prodi_col].isin(selected_prodi)]
        if not sub_df.empty:
            agg = sub_df.groupby('ASAL SEKOLAH').agg(
                mahasiswa=('NILAI KESELURUHAN', 'count'),
                ipk_mean=('NILAI KESELURUHAN', 'mean'),
                poin_akademik=('POIN_AKADEMIK', 'sum'),
                poin_non_akademik=('POIN_NON_AKADEMIK', 'sum'),
                kota=('KOTA SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                provinsi=('PROVINSI SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            ).fillna(0).reset_index()
            
            max_mhs = agg['mahasiswa'].max() or 1
            agg['skor'] = 0.7 * agg['ipk_mean'] + 0.3 * (agg['mahasiswa'] / max_mhs * 4.0)
            agg = agg.sort_values('skor', ascending=False)
            
            for _, row in agg.iterrows():
                sekolah_recom.append({
                    'sekolah': str(row['ASAL SEKOLAH']),
                    'kota': str(row['kota']),
                    'provinsi': str(row['provinsi']),
                    'mahasiswa': int(row['mahasiswa']),
                    'ipk': round(float(row['ipk_mean']), 2),
                    'poin_akademik': int(row['poin_akademik']),
                    'poin_non_akademik': int(row['poin_non_akademik']),
                    'skor': round(float(row['skor']), 2)
                })
                
    return jsonify({
        'ok': True,
        'prodi_recommendations': prodi_recom,
        'sekolah_recommendations': sekolah_recom
    })
