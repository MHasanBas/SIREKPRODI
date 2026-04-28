import os
import json
import pickle
import shutil
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from services.model_service import load_active_model_name

model_bp = Blueprint('model', __name__)

@model_bp.route('/pelatihan_model')
def pelatihan_model():
    model_list = []
    active_model = load_active_model_name()

    for folder in os.listdir("models"):
        if folder.startswith("model_"):
            model_path = os.path.join("models", folder)


            meta_path = os.path.join(model_path, "meta.json")
            pkl_path = os.path.join(model_path, "hasil_kmeans_3cluster.pkl")
            uploaded_at = "-"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                nama_data = meta.get("nama_data", "-")
                uploaded_at = meta.get("uploaded_at") or "-"
            else:
                nama_data = "-"

            if uploaded_at not in ("-", "", None):
                # Normalisasi jika tersimpan sebagai ISO string.
                try:
                    dt = datetime.fromisoformat(str(uploaded_at))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("Asia/Jakarta"))
                    uploaded_at = dt.astimezone(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

            if uploaded_at == "-":
                # Fallback untuk model lama: pakai mtime meta.json lalu pkl.
                ts = None
                if os.path.exists(meta_path):
                    ts = os.path.getmtime(meta_path)
                elif os.path.exists(pkl_path):
                    ts = os.path.getmtime(pkl_path)
                if ts is not None:
                    uploaded_at = datetime.fromtimestamp(ts, ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

            status = "Aktif" if folder == active_model else ""
            model_list.append({
                "nama_model": folder,
                "nama_data": nama_data,
                "uploaded_at": uploaded_at,
                "status": status
            })

    return render_template("pelatihan_model.html", model_list=model_list)

@model_bp.route('/terapkan_model/<nama_model>')
def terapkan_model(nama_model):
    active_flag_path = "models/active_model.txt"
    with open(active_flag_path, "w") as f:
        f.write(nama_model)
    flash(f" Model {nama_model} sekarang diterapkan sebagai model aktif.")
    return redirect(url_for('model.pelatihan_model'))

@model_bp.route('/hapus_model/<nama_model>')
def hapus_model(nama_model):
    active_flag_path = "models/active_model.txt"

    # Cegah penghapusan model aktif
    if os.path.exists(active_flag_path):
        with open(active_flag_path, "r") as f:
            active_model = f.read().strip()
        if nama_model == active_model:
            flash(" Tidak bisa menghapus model yang sedang aktif.", "error")
            return redirect(url_for('model.pelatihan_model'))

    # Hapus folder model
    model_path = os.path.join("models", nama_model)
    if os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
        flash(f"Model {nama_model} berhasil dihapus.", "success")
    else:
        flash(" Model tidak ditemukan.", "warning")

    return redirect(url_for('model.pelatihan_model'))

@model_bp.route('/preview_model/<nama_model>')
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

@model_bp.route('/update_model_meta', methods=['POST'])
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
        with open(meta_path) as f:
            meta = json.load(f)
    meta["nama_data"] = nama_data_baru
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return jsonify({"success": True, "message": "Nama data berhasil diperbarui."})
