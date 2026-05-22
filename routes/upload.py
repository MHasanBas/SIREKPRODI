import os
import pickle
import json
import math
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
from clustering.preprocessing import proses_upload_data
from clustering.kmeans_module import DEFAULT_TRAINING_RUNS, jalankan_kmeans
from services.model_service import get_next_model_folder, load_active_model_name
from utils.helpers import format_datetime_jakarta

upload_bp = Blueprint('upload', __name__)
UPLOAD_PROGRESS = {}
UPLOAD_PROGRESS_LOCK = threading.Lock()


def _is_ajax_upload():
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def _set_upload_progress(job_id, percent, message, run_current=0, run_total=DEFAULT_TRAINING_RUNS, status="running"):
    if not job_id:
        return
    payload = {
        "percent": max(0, min(100, int(percent))),
        "message": message,
        "run_current": int(run_current or 0),
        "run_total": int(run_total or DEFAULT_TRAINING_RUNS),
        "status": status,
        "updated_at": time.time(),
    }
    with UPLOAD_PROGRESS_LOCK:
        UPLOAD_PROGRESS[job_id] = payload


def _upload_error(message, status_code=400):
    _set_upload_progress(request.form.get("upload_job_id"), 100, message, 0, DEFAULT_TRAINING_RUNS, "failed")
    if _is_ajax_upload():
        return jsonify({"success": False, "error": message}), status_code
    flash(message, "error")
    return redirect(url_for('upload.upload'))


@upload_bp.route('/upload/progress/<job_id>')
def upload_progress(job_id):
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    with UPLOAD_PROGRESS_LOCK:
        payload = UPLOAD_PROGRESS.get(job_id)
    if not payload:
        payload = {
            "percent": 0,
            "message": "Menunggu proses upload dimulai...",
            "run_current": 0,
            "run_total": DEFAULT_TRAINING_RUNS,
            "status": "pending",
        }
    return jsonify({"success": True, **payload})


def _load_model_dbi(model_name):
    if not model_name:
        return None

    meta_path = os.path.join("models", model_name, "meta.json")
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        dbi = meta.get("dbi_after_ga")
        if dbi is None:
            return None
        dbi = float(dbi)
        return dbi if math.isfinite(dbi) else None
    except Exception:
        return None


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

@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        job_id = request.form.get('upload_job_id')
        ajax_upload = _is_ajax_upload()
        _set_upload_progress(job_id, 1, "Memvalidasi form upload...", 0, DEFAULT_TRAINING_RUNS)

        nama_data = request.form.get('nama_data')
        files = [f for f in request.files.getlist('files') if f.filename]
         
       
        if not nama_data or nama_data.strip() == "":
            return _upload_error("Nama data tidak boleh kosong.")
        
   
        if len(files) < 1:
            return _upload_error("Minimal 1 file .xlsx harus diunggah (format lama butuh pasangan data/nilai, format baru cukup satu file).")


        for f in files:
            if not f.filename.endswith('.xlsx'):
                return _upload_error(f"Format file tidak didukung: {f.filename}. Harus .xlsx")

        # Bersihkan folder upload
        _set_upload_progress(job_id, 5, "Membersihkan folder upload...", 0, DEFAULT_TRAINING_RUNS)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            os.remove(os.path.join(upload_folder, filename))

         # Simpan file Excel
        _set_upload_progress(job_id, 10, "Menyimpan file Excel...", 0, DEFAULT_TRAINING_RUNS)
        for f in files:
            if f.filename.endswith('.xlsx'):
                f.save(os.path.join(upload_folder, secure_filename(f.filename)))
        
        # Proses dan gabungkan dengan data lama (jika ada)
        _set_upload_progress(job_id, 15, "Membersihkan dan menggabungkan data...", 0, DEFAULT_TRAINING_RUNS)
        df_baru = proses_upload_data(upload_folder)
        if df_baru.empty:
            return _upload_error("Data tidak valid atau kolom wajib tidak ditemukan. Periksa kembali file yang diunggah.")

        active_model = load_active_model_name()
        timpa_data = request.form.get('timpa_data')
        
        df_prev = None
        if not timpa_data:
            prev_path = os.path.join("models", active_model, "data_gabungan_clean.pkl")
            if os.path.exists(prev_path):
                try:
                    with open(prev_path, "rb") as f:
                        df_prev = pickle.load(f)
                    print(f"Data lama ditemukan: {prev_path}, baris: {df_prev.shape[0]}")
                except Exception as e:
                    print("Gagal memuat data lama, lanjut pakai data baru saja:", e)

        if df_prev is not None and not df_prev.empty:
            df_baru = pd.concat([df_prev, df_baru], ignore_index=True, sort=False)
            print(f"Data gabungan (lama+baru): {df_baru.shape[0]} baris")
        else:
            print(f"Ter-reset, Data baru saja: {df_baru.shape[0]} baris")

        # Proses dan simpan sebagai model baru (tidak menimpa model aktif)
        model_folder = get_next_model_folder()
        os.makedirs(model_folder, exist_ok=True)

        # Tulis meta.json dulu agar jalankan_kmeans() bisa menambahkan metrik ke dalamnya
        meta = {
            "nama_data": nama_data,
            "uploaded_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(timespec="seconds"),
            "k_selection": "manual_k_3",
        }
        with open(os.path.join(model_folder, "meta.json"), "w") as f:
            json.dump(meta, f)

        # Jalankan clustering (akan menambahkan silhouette, dbi, elbow ke meta.json)
        def training_progress(run_current, run_total, state):
            if state == "running":
                percent = 20 + ((run_current - 1) / run_total) * 70
                message = f"Training model K-Means + GA: run {run_current}/{run_total}..."
            else:
                percent = 20 + (run_current / run_total) * 70
                message = f"Training run {run_current}/{run_total} selesai."
            _set_upload_progress(job_id, percent, message, run_current, run_total)

        _set_upload_progress(job_id, 20, "Memulai training model...", 0, DEFAULT_TRAINING_RUNS)
        jalankan_kmeans(df_baru, n_clusters=3, save_path=model_folder, progress_callback=training_progress)

        _set_upload_progress(job_id, 92, "Menyimpan data bersih dan hasil model...", DEFAULT_TRAINING_RUNS, DEFAULT_TRAINING_RUNS)
        with open(os.path.join(model_folder, "data_gabungan_clean.pkl"), "wb") as f:
            pickle.dump(df_baru, f)

        # Cek isi .pkl
        try:
            with open(os.path.join(model_folder, 'hasil_kmeans_3cluster.pkl'), 'rb') as f:
                kmeans_test = pickle.load(f)
            print("hasil_kmeans_3cluster.pkl berhasil dibaca")
        except Exception as e:
            print("ERROR membaca hasil_kmeans_3cluster.pkl:", e)
            return _upload_error("Upload selesai, tetapi model tidak diaktifkan karena file hasil clustering gagal dibaca.", 500)

        new_model_name = os.path.basename(model_folder)
        with open("models/active_model.txt", "w") as f:
            f.write(new_model_name)
        _set_upload_progress(job_id, 100, f"Upload selesai. {new_model_name} aktif sebagai model terbaru.", DEFAULT_TRAINING_RUNS, DEFAULT_TRAINING_RUNS, "completed")

        if ajax_upload:
            return jsonify({
                "success": True,
                "message": f"Upload selesai. {new_model_name} otomatis aktif sebagai model terbaru.",
                "redirect_url": url_for('dashboard.dashboard'),
                "model_name": new_model_name,
            })

        flash(f"Upload selesai. {new_model_name} otomatis aktif sebagai model terbaru.", "success")
        return redirect(url_for('dashboard.dashboard'))

    active_model = load_active_model_name()
    return render_template(
        'upload.html',
        active_model=active_model,
        active_model_meta=_load_model_meta(active_model),
        default_training_runs=DEFAULT_TRAINING_RUNS,
    )
