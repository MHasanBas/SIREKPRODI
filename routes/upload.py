import os
import pickle
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app
from werkzeug.utils import secure_filename
import pandas as pd
from clustering.preprocessing import proses_upload_data
from clustering.kmeans_module import jalankan_kmeans
from services.model_service import get_next_model_folder, load_active_model_name

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        nama_data = request.form.get('nama_data')
        files = [f for f in request.files.getlist('files') if f.filename]
         
       
        if not nama_data or nama_data.strip() == "":
            flash(" Nama data tidak boleh kosong.", "error")
            return redirect(url_for('upload.upload'))
        
   
        if len(files) < 1:
            flash(" Minimal 1 file .xlsx harus diunggah (format lama butuh pasangan data/nilai, format baru cukup satu file).", "error")
            return redirect(url_for('upload.upload'))


        for f in files:
            if not f.filename.endswith('.xlsx'):
                flash(f"Format file tidak didukung: {f.filename}. Harus .xlsx", "error")
                return redirect(url_for('upload.upload'))

        # Bersihkan folder upload
        upload_folder = current_app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            os.remove(os.path.join(upload_folder, filename))

         # Simpan file Excel
        for f in files:
            if f.filename.endswith('.xlsx'):
                f.save(os.path.join(upload_folder, secure_filename(f.filename)))
        
        # Proses dan gabungkan dengan data lama (jika ada)
        df_baru = proses_upload_data(upload_folder)
        if df_baru.empty:
            flash("Data tidak valid atau kolom wajib tidak ditemukan. Periksa kembali file yang diunggah.", "error")
            return redirect(url_for('upload.upload'))

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
        jalankan_kmeans(df_baru, save_path=model_folder)

        with open(os.path.join(model_folder, "data_gabungan_clean.pkl"), "wb") as f:
            pickle.dump(df_baru, f)
        

        meta = {
            "nama_data": nama_data,
            # Simpan waktu upload agar bisa ditampilkan di halaman Pelatihan Model.
            "uploaded_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(timespec="seconds"),
        }
        with open(os.path.join(model_folder, "meta.json"), "w") as f:
            json.dump(meta, f)

        # Cek isi .pkl
        try:
            with open(os.path.join(model_folder, 'hasil_kmeans_3cluster.pkl'), 'rb') as f:
                kmeans_test = pickle.load(f)
            print("hasil_kmeans_3cluster.pkl berhasil dibaca")
        except Exception as e:
            print("ERROR membaca hasil_kmeans_3cluster.pkl:", e)

        # Otomatis terapkan model baru
        with open("models/active_model.txt", "w") as f:
            f.write(os.path.basename(model_folder))
    
        flash(f"Data berhasil diupload. Model baru dibuat dan diaktifkan: {os.path.basename(model_folder)}.")
        return redirect(url_for('dashboard.dashboard'))

    return render_template('upload.html')
