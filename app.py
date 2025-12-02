from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
from clustering.preprocessing import proses_upload_data
from clustering.kmeans_module import jalankan_kmeans
from clustering.fcm_module import jalankan_fcm
import pickle

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Buat folder yang diperlukan
for folder in ['uploads', 'models/model_utama']:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['user'] = 'admin'
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah!')
    return render_template('login.html')

def get_next_model_folder():
    base_dir = "models"
    existing = [d for d in os.listdir(base_dir) if d.startswith("model_") and d != "model_utama"]
    next_id = len(existing) + 1
    folder_path = os.path.join(base_dir, f"model_{next_id}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        nama_data = request.form.get('nama_data')
        files = [f for f in request.files.getlist('files') if f.filename]
         
        # === ✅ VALIDASI: Nama data tidak boleh kosong ===
        if not nama_data or nama_data.strip() == "":
            flash("❌ Nama data tidak boleh kosong.", "error")
            return redirect(url_for('upload'))
        
        # === ✅ VALIDASI: Minimal 1 file ===
        if len(files) < 1:
            flash("❌ Minimal 1 file .xlsx harus diunggah (format lama butuh pasangan data/nilai, format baru cukup satu file).", "error")
            return redirect(url_for('upload'))

        # === ✅ VALIDASI: Format file harus .xlsx
        for f in files:
            if not f.filename.endswith('.xlsx'):
                flash(f"❌ Format file tidak didukung: {f.filename}. Harus .xlsx", "error")
                return redirect(url_for('upload'))

        # Bersihkan folder upload
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

         # Simpan file Excel
       
        for f in files:
            if f.filename.endswith('.xlsx'):
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        
        # Proses dan buat folder model baru
        df_baru = proses_upload_data(app.config['UPLOAD_FOLDER'])
        if df_baru.empty:
            flash("❌ Data tidak valid atau kolom wajib tidak ditemukan. Periksa kembali file yang diunggah.", "error")
            return redirect(url_for('upload'))
        model_folder = get_next_model_folder()
        
        jalankan_kmeans(df_baru, save_path=model_folder)
        jalankan_fcm(df_baru, save_path=model_folder)

        with open(os.path.join(model_folder, "data_gabungan_clean.pkl"), "wb") as f:
            pickle.dump(df_baru, f)
        
        # 🔥 Simpan nama data ke file meta.json
        import json
        meta = {"nama_data": nama_data}
        with open(os.path.join(model_folder, "meta.json"), "w") as f:
            json.dump(meta, f)

        # Update model aktif
        with open("models/active_model.txt", "w") as f:
            f.write(os.path.basename(model_folder))
            
        # Cek isi .pkl
        try:
            with open(os.path.join(model_folder, 'hasil_kmeans_3cluster.pkl'), 'rb') as f:
                kmeans_test = pickle.load(f)
            print("✅ hasil_kmeans_3cluster.pkl berhasil dibaca")
            print("🔑 Key tersedia:", list(kmeans_test.keys()) if isinstance(kmeans_test, dict) else "Bukan dict")
        except Exception as e:
            print("❌ ERROR membaca hasil_kmeans_3cluster.pkl:", e)

        try:
            with open(os.path.join(model_folder, 'hasil_fcm_3cluster.pkl'), 'rb') as f:
                fcm_test = pickle.load(f)
            print("✅ hasil_fcm_3cluster.pkl berhasil dibaca")
            print("🔑 Key tersedia:", list(fcm_test.keys()) if isinstance(fcm_test, dict) else "Bukan dict")
        except Exception as e:
            print("❌ ERROR membaca hasil_fcm_3cluster.pkl:", e)
    
        flash("✅ Data berhasil diupload dan model berhasil diretrain.")
        return redirect(url_for('dashboard'))

    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    tampilkan = request.args.get('tampilkan', 'top10')  # default: top10
    selected_prodi = request.args.get('prodi', 'all')

    active_flag_path = "models/active_model.txt"
    active_model = "model_utama"
    if os.path.exists(active_flag_path):
        with open(active_flag_path, "r") as f:
            active_model = f.read().strip()

    try:
        with open(f"models/{active_model}/hasil_kmeans_3cluster.pkl", "rb") as f:
            hasil_kmeans = pickle.load(f)
            df_kmeans = hasil_kmeans["data"] if isinstance(hasil_kmeans, dict) else hasil_kmeans
            # STRIP SEMUA KOLOM untuk buang spasi tak terlihat
            df_kmeans.columns = df_kmeans.columns.str.strip()
            print("📋 Kolom df_kmeans:", df_kmeans.columns.tolist())
    except FileNotFoundError:
        flash("❌ File hasil clustering tidak ditemukan. Silakan upload ulang data.")
        return redirect(url_for('upload'))

    prodi_column = None
    for kandidat in ['PROGRAM STUDI', 'PROGRAM_STUDI', 'PRODI']:
        if kandidat in df_kmeans.columns:
            prodi_column = kandidat
            break

    prodi_list = sorted(df_kmeans[prodi_column].dropna().unique().tolist()) if prodi_column else []

    if prodi_column and selected_prodi != 'all':
        df_kmeans_filtered = df_kmeans[df_kmeans[prodi_column] == selected_prodi]

        if df_kmeans_filtered.empty:
            flash(f"⚠️ Data untuk prodi {selected_prodi} tidak ditemukan.", "warning")
            return redirect(url_for('dashboard'))

        df_kmeans = df_kmeans_filtered

    def _norm(series):
        if series.max() == series.min():
            return pd.Series(0, index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    rekom_prodi_per_sekolah = []
    rekom_sekolah_per_prodi = []

    if prodi_column:
        # Sekolah unggulan berdasarkan ipk/mahasiswa/deviasi
        sekolah_stats = df_kmeans.groupby('ASAL SEKOLAH').agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std')
        ).fillna(0).reset_index()
        sekolah_stats['skor'] = (
            0.4 * _norm(sekolah_stats['mahasiswa']) +
            0.4 * _norm(sekolah_stats['ipk']) +
            0.2 * (1 - _norm(sekolah_stats['deviasi']))
        )

        info_sekolah = df_kmeans.groupby('ASAL SEKOLAH').agg(
            kota=('KOTA SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            provinsi=('PROVINSI SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        )

        sekolah_unggulan = sekolah_stats.sort_values('skor', ascending=False).head(10)

        for _, row in sekolah_unggulan.iterrows():
            nama_sekolah = row['ASAL SEKOLAH']
            df_school = df_kmeans[df_kmeans['ASAL SEKOLAH'] == nama_sekolah]
            prodi_stats = df_school.groupby(prodi_column).agg(
                mahasiswa=('NILAI KESELURUHAN', 'count'),
                ipk=('NILAI KESELURUHAN', 'mean'),
                deviasi=('NILAI KESELURUHAN', 'std')
            ).fillna(0).reset_index()
            prodi_stats = prodi_stats.rename(columns={prodi_column: 'prodi'})
            prodi_stats['skor'] = (
                0.45 * _norm(prodi_stats['mahasiswa']) +
                0.4 * _norm(prodi_stats['ipk']) +
                0.15 * (1 - _norm(prodi_stats['deviasi']))
            )
            top_prodi = prodi_stats.sort_values('skor', ascending=False).head(3).to_dict(orient='records')
            info = info_sekolah.loc[nama_sekolah]
            rekom_prodi_per_sekolah.append({
                "sekolah": nama_sekolah,
                "kota": info['kota'],
                "provinsi": info['provinsi'],
                "ipk_mean": round(row['ipk'], 2),
                "mahasiswa": int(row['mahasiswa']),
                "top_prodi": top_prodi
            })

        # Sekolah sasaran per prodi
        for prodi in sorted(df_kmeans[prodi_column].dropna().unique()):
            df_prodi = df_kmeans[df_kmeans[prodi_column] == prodi]
            sekolah_stats_prodi = df_prodi.groupby('ASAL SEKOLAH').agg(
                mahasiswa=('NILAI KESELURUHAN', 'count'),
                ipk=('NILAI KESELURUHAN', 'mean'),
                deviasi=('NILAI KESELURUHAN', 'std'),
                kota=('KOTA SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                provinsi=('PROVINSI SEKOLAH', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            ).fillna(0).reset_index()
            sekolah_stats_prodi = sekolah_stats_prodi.rename(columns={'ASAL SEKOLAH': 'sekolah'})
            sekolah_stats_prodi['skor'] = (
                0.45 * _norm(sekolah_stats_prodi['mahasiswa']) +
                0.4 * _norm(sekolah_stats_prodi['ipk']) +
                0.15 * (1 - _norm(sekolah_stats_prodi['deviasi']))
            )
            top_sekolah = sekolah_stats_prodi.sort_values('skor', ascending=False).head(5).to_dict(orient='records')
            rekom_sekolah_per_prodi.append({
                "prodi": prodi,
                "top_sekolah": top_sekolah,
                "total_mahasiswa": int(sekolah_stats_prodi['mahasiswa'].sum()),
                "ipk_mean": round(df_prodi['NILAI KESELURUHAN'].mean(), 2)
            })

    def siapkan_data(df, nama_algo, tampilkan='top10'):
        jumlah_mhs_per_sekolah = df.groupby('ASAL SEKOLAH')['NILAI KESELURUHAN'].count().reset_index(name='jumlah_mahasiswa')

        df = df.merge(jumlah_mhs_per_sekolah, on='ASAL SEKOLAH', how='left')

        df_dev = df.groupby(['Cluster', 'ASAL SEKOLAH']).agg(
            mahasiswa=('NILAI KESELURUHAN', 'count'),
            ipk=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std')
        ).reset_index()

        df_info_sekolah = df[['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH']].drop_duplicates()
        df_dev = df_dev.merge(df_info_sekolah, on='ASAL SEKOLAH', how='left')

        cluster_summary = df.groupby('Cluster').agg(
            total_mahasiswa=('jumlah_mahasiswa', 'sum'),
            ipk_rata2=('NILAI KESELURUHAN', 'mean'),
            deviasi=('NILAI KESELURUHAN', 'std')
        ).reset_index()

        cluster_summary['mahasiswa_norm'] = (cluster_summary['total_mahasiswa'] - cluster_summary['total_mahasiswa'].min()) / (cluster_summary['total_mahasiswa'].max() - cluster_summary['total_mahasiswa'].min())
        cluster_summary['ipk_norm'] = (cluster_summary['ipk_rata2'] - cluster_summary['ipk_rata2'].min()) / (cluster_summary['ipk_rata2'].max() - cluster_summary['ipk_rata2'].min())
        cluster_summary['deviasi_norm'] = (cluster_summary['deviasi'] - cluster_summary['deviasi'].min()) / (cluster_summary['deviasi'].max() - cluster_summary['deviasi'].min())
        cluster_summary['ranking'] = (
            0.4 * cluster_summary['mahasiswa_norm'] +
            0.3 * cluster_summary['ipk_norm'] +
            0.3 * (1 - cluster_summary['deviasi_norm'])
        )

        ordered_clusters = cluster_summary.sort_values(by='ranking', ascending=False)['Cluster'].tolist()
        label_cluster = ['A', 'B', 'C']
        if ordered_clusters:
            ordered_clusters = ordered_clusters + [ordered_clusters[-1]] * (len(label_cluster) - len(ordered_clusters))
        result = {'algoritma': nama_algo}
        
        summary_dict = {}
        for idx, label in enumerate(label_cluster):
            cluster_id = ordered_clusters[idx]
            cluster_data = cluster_summary[cluster_summary['Cluster'] == cluster_id].iloc[0]
            summary_dict[label] = {
                'ipk': cluster_data['ipk_rata2'],
                'mahasiswa': cluster_data['total_mahasiswa'],
                'deviasi': cluster_data['deviasi']
            }
        
        def bandingkan(c1, c2, key):
            return "lebih tinggi" if c1[key] > c2[key] else "lebih rendah" if c1[key] < c2[key] else "sama"

        deskripsi_cluster = {}
        deskripsi_cluster['A'] = (
            f"Jumlah mahasiswa dari tiap sekolah dengan tingkat keberlanjutan studi di Polinema {bandingkan(summary_dict['A'], summary_dict['B'], 'mahasiswa')} dari cluster B "
            f"dan {bandingkan(summary_dict['A'], summary_dict['C'], 'mahasiswa')} dari cluster C. "
            f"IPK rata-rata {bandingkan(summary_dict['A'], summary_dict['B'], 'ipk')} dari cluster B "
            f"dan {bandingkan(summary_dict['A'], summary_dict['C'], 'ipk')} dari cluster C. "
            f"Standar deviasi {bandingkan(summary_dict['A'], summary_dict['B'], 'deviasi')} dari cluster B "
            f"dan {bandingkan(summary_dict['A'], summary_dict['C'], 'deviasi')} dari cluster C."
        )

        deskripsi_cluster['B'] = (
            f"Jumlah mahasiswa dari tiap sekolah dengan tingkat keberlanjutan studi di Polinema {bandingkan(summary_dict['B'], summary_dict['A'], 'mahasiswa')} dari cluster A "
            f"dan {bandingkan(summary_dict['B'], summary_dict['C'], 'mahasiswa')} dari cluster C. "
            f"IPK rata-rata {bandingkan(summary_dict['B'], summary_dict['A'], 'ipk')} dari cluster A "
            f"dan {bandingkan(summary_dict['B'], summary_dict['C'], 'ipk')} dari cluster C. "
            f"Standar deviasi {bandingkan(summary_dict['B'], summary_dict['A'], 'deviasi')} dari cluster A "
            f"dan {bandingkan(summary_dict['B'], summary_dict['C'], 'deviasi')} dari cluster C."
        )

        deskripsi_cluster['C'] = (
            f"Jumlah mahasiswa dari tiap sekolah dengan tingkat keberlanjutan studi di Polinema {bandingkan(summary_dict['C'], summary_dict['A'], 'mahasiswa')} dari cluster A "
            f"dan {bandingkan(summary_dict['C'], summary_dict['B'], 'mahasiswa')} dari cluster B. "
            f"IPK rata-rata {bandingkan(summary_dict['C'], summary_dict['A'], 'ipk')} dari cluster A "
            f"dan {bandingkan(summary_dict['C'], summary_dict['B'], 'ipk')} dari cluster B. "
            f"Standar deviasi {bandingkan(summary_dict['C'], summary_dict['A'], 'deviasi')} dari cluster A "
            f"dan {bandingkan(summary_dict['C'], summary_dict['B'], 'deviasi')} dari cluster B."
        )

        # Tambahkan ke result
        result['deskripsi_cluster'] = deskripsi_cluster

        for idx, label in enumerate(label_cluster):
            cluster_id = ordered_clusters[idx]
            df_final = df[df['Cluster'] == cluster_id].copy()
            df_sub = df_dev[df_dev['Cluster'] == cluster_id].copy()

            # # Hitung global min-max dari seluruh data (bukan hanya dari df_sub)
            # global_min_mhs = df_dev['mahasiswa'].min()
            # global_max_mhs = df_dev['mahasiswa'].max()
            # global_min_ipk = df_dev['ipk'].min()
            # global_max_ipk = df_dev['ipk'].max()
            # global_min_dev = df_dev['deviasi'].min()
            # global_max_dev = df_dev['deviasi'].max()

            # # Terapkan ke df_sub (masing-masing sekolah di cluster)
            # df_sub['mahasiswa_norm'] = (df_sub['mahasiswa'] - global_min_mhs) / (global_max_mhs - global_min_mhs)
            # df_sub['ipk_norm'] = (df_sub['ipk'] - global_min_ipk) / (global_max_ipk - global_min_ipk)
            # df_sub['deviasi_norm'] = (df_sub['deviasi'] - global_min_dev) / (global_max_dev - global_min_dev)

            # # Hitung skor gabungan dengan bobot yang sudah kamu tetapkan
            # df_sub['skor'] = 0.4 * df_sub['mahasiswa_norm'] + 0.3 * df_sub['ipk_norm'] + 0.3 * (1 - df_sub['deviasi_norm'])

            df_sub['skor'] = (
                0.4 * df_sub['mahasiswa'] +
                0.3 * df_sub['ipk'] -
                0.3 * df_sub['deviasi']
            ).round(3)


            df_sorted = df_sub.sort_values(by='skor', ascending=False)
            if tampilkan == 'semua':
                tabel_rekomendasi = df_sorted[
                    ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'mahasiswa', 'ipk', 'deviasi']
                ].to_dict(orient='records')
            else:
                tabel_rekomendasi = df_sorted.head(10)[
                    ['ASAL SEKOLAH', 'KOTA SEKOLAH', 'PROVINSI SEKOLAH', 'mahasiswa', 'ipk', 'deviasi']
                ].to_dict(orient='records')


            bins = [2.0, 2.5, 3.0, 3.5, 4.0]
            labels = ['2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0']
            df_final.loc[:, 'ipk_bin'] = pd.cut(df_final['NILAI KESELURUHAN'], bins=bins, labels=labels, include_lowest=True)
            distribusi_raw = df_final['ipk_bin'].value_counts().sort_index().reset_index(name='count')
            distribusi_raw.columns = ['jumlah', 'count']
            distribusi_ipk = [
                {"jumlah": str(row['jumlah']), "count": int(row['count']) if pd.notnull(row['count']) else 0}
                for _, row in distribusi_raw.iterrows()
            ]

            line_top10 = df_final['KOTA SEKOLAH'].value_counts().head(10)
            line_chart = [{"kota": k, "jumlah": int(v)} for k, v in line_top10.items()]

            top_row = df_sub.sort_values(by='skor', ascending=False).iloc[0]
            result[label] = {
                'total_mahasiswa': int(df_final.shape[0]),
                'ipk_mean': round(df_final['NILAI KESELURUHAN'].mean(), 2),
                'kota_terbaik': top_row['KOTA SEKOLAH'],
                'sekolah_terbaik': top_row['ASAL SEKOLAH'],
                'tabel_rekomendasi': tabel_rekomendasi,
                'distribusi_ipk': distribusi_ipk,
                'mahasiswa_per_kota_top10': line_chart,
                'deskripsi': deskripsi_cluster[label]
            }

        return result

    kmeans_data = siapkan_data(df_kmeans, "K-Means", tampilkan)


    return render_template(
        "dashboard.html",
        kmeans_data=kmeans_data,
        prodi_list=prodi_list,
        selected_prodi=selected_prodi,
        rekom_prodi_per_sekolah=rekom_prodi_per_sekolah,
        rekom_sekolah_per_prodi=rekom_sekolah_per_prodi,
        prodi_column_available=bool(prodi_column)
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/pelatihan_model')
def pelatihan_model():
    import json
    
    model_list = []
    active_model = "model_utama"
    if os.path.exists("models/active_model.txt"):
        with open("models/active_model.txt") as f:
            active_model = f.read().strip()

    for folder in os.listdir("models"):
        if folder.startswith("model_"):
            model_path = os.path.join("models", folder)

            # 🔥 Ambil nama_data dari meta.json (kalau ada)
            meta_path = os.path.join(model_path, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                nama_data = meta.get("nama_data", "-")
            else:
                nama_data = "-"

            status = "Aktif" if folder == active_model else ""
            model_list.append({
                "nama_model": folder,
                "nama_data": nama_data,
                "status": status
            })

    return render_template("pelatihan_model.html", model_list=model_list)

@app.route('/terapkan_model/<nama_model>')
def terapkan_model(nama_model):
    active_flag_path = "models/active_model.txt"
    with open(active_flag_path, "w") as f:
        f.write(nama_model)
    flash(f"✅ Model {nama_model} sekarang diterapkan sebagai model aktif.")
    return redirect(url_for('pelatihan_model'))

@app.route('/hapus_model/<nama_model>')
def hapus_model(nama_model):
    active_flag_path = "models/active_model.txt"

    # Cegah penghapusan model aktif
    if os.path.exists(active_flag_path):
        with open(active_flag_path, "r") as f:
            active_model = f.read().strip()
        if nama_model == active_model:
            flash("❌ Tidak bisa menghapus model yang sedang aktif.", "error")
            return redirect(url_for('pelatihan_model'))

    # Hapus folder model
    model_path = os.path.join("models", nama_model)
    if os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
        flash(f"🗑️ Model {nama_model} berhasil dihapus.", "success")
    else:
        flash("⚠️ Model tidak ditemukan.", "warning")

    return redirect(url_for('pelatihan_model'))

@app.route('/preview_model/<nama_model>')
def preview_model(nama_model):
    model_path = os.path.join("models", nama_model)
    data_path = os.path.join(model_path, "data_gabungan_clean.pkl")
    limit_param = request.args.get("limit", "20")

    if not os.path.exists(data_path):
        return jsonify({"success": False, "error": "data_gabungan_clean.pkl tidak ditemukan."}), 404

    try:
        with open(data_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and "data" in loaded:
            df = loaded["data"]
        else:
            try:
                df = pd.DataFrame(loaded)
            except Exception:
                return jsonify({"success": False, "error": "Format data tidak dikenali untuk preview."}), 500

        limit = None if limit_param == "all" else int(limit_param)
        preview_df = df if limit is None else df.head(limit)
        preview_json = preview_df.copy()
        preview_json = preview_json.where(pd.notnull(preview_json), None)
        preview_json = preview_json.applymap(lambda x: x if isinstance(x, (str, int, float, bool, type(None))) else str(x))
        return jsonify({
            "success": True,
            "total_rows": int(df.shape[0]),
            "columns": list(preview_df.columns),
            "rows": preview_json.to_dict(orient="records"),
            "limit_used": "all" if limit is None else limit
        })
    except ValueError:
        return jsonify({"success": False, "error": "Parameter limit tidak valid."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/update_model_meta', methods=['POST'])
def update_model_meta():
    data = request.get_json(silent=True) or {}
    nama_model = data.get("nama_model")
    nama_data_baru = data.get("nama_data")

    if not nama_model or not nama_data_baru:
        return jsonify({"success": False, "error": "nama_model dan nama_data wajib diisi."}), 400

    model_path = os.path.join("models", nama_model)
    if not os.path.exists(model_path):
        return jsonify({"success": False, "error": "Model tidak ditemukan."}), 404

    meta_path = os.path.join(model_path, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
    meta["nama_data"] = nama_data_baru
    with open(meta_path, "w") as f:
        import json
        json.dump(meta, f)

    return jsonify({"success": True, "message": "Nama data berhasil diperbarui."})

if __name__ == '__main__':
    app.run(debug=True)
