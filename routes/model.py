import os
import json
import pickle
import subprocess
import shutil
import math
import uuid
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from clustering.kmeans_module import _build_feature_matrix
from services.model_service import load_active_model_name
from services.training_config_service import load_active_training_config, save_active_training_config
from utils.helpers import format_datetime_jakarta
from tune_ga_ui_worker import JOB_STATE_DIR, job_state_path

model_bp = Blueprint('model', __name__)


def _latest_tuning_result():
    base = "outputs/ga_tuning"
    if not os.path.isdir(base):
        return None
    run_dirs = sorted(
        os.path.join(base, name)
        for name in os.listdir(base)
        if name.startswith("run_") and os.path.isdir(os.path.join(base, name))
    )
    for run_dir in reversed(run_dirs):
        summary_path = os.path.join(run_dir, "ga_tuning_summary.csv")
        manifest_path = os.path.join(run_dir, "ga_tuning_manifest.json")
        if not os.path.exists(summary_path):
            continue
        try:
            summary = pd.read_csv(summary_path)
            best = summary.iloc[0].to_dict() if not summary.empty else {}
            manifest = {}
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
            return {
                "run_dir": run_dir,
                "summary_path": summary_path,
                "manifest_path": manifest_path,
                "n_executions": manifest.get("n_executions"),
                "created_at": manifest.get("created_at"),
                "best": best,
            }
        except Exception:
            continue
    return None


def _latest_tuning_best_config():
    latest = _latest_tuning_result()
    if not latest or not latest.get("best"):
        return None
    best = latest["best"]
    try:
        mutation_rate = best.get("mutation_rate")
        return {
            "population_size": int(best["population_size"]),
            "generations": int(best["generations"]),
            "mutation_rate": float(mutation_rate) if mutation_rate is not None else float(best["late_mutation_rate"]),
            "early_mutation_rate": float(best["early_mutation_rate"]),
            "mid_mutation_rate": float(best["mid_mutation_rate"]),
            "late_mutation_rate": float(best["late_mutation_rate"]),
            "max_stagnant": int(best["max_stagnant"]),
            "hyperparameter_source": f'{os.path.basename(latest["run_dir"])}_rank_{int(best["rank"])}',
            "tuning_run_dir": latest["run_dir"],
            "tuning_created_at": latest.get("created_at"),
        }
    except Exception:
        return None


def _read_tuning_job(job_id):
    path = job_state_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _running_tuning_job():
    if not os.path.isdir(JOB_STATE_DIR):
        return None
    for name in sorted(os.listdir(JOB_STATE_DIR), reverse=True):
        if not name.endswith(".json"):
            continue
        job_id = name[:-5]
        job = _read_tuning_job(job_id)
        if not job:
            continue
        if job.get("status") == "running":
            pid = job.get("pid")
            try:
                if pid:
                    os.kill(int(pid), 0)
                return job_id, job
            except Exception:
                job["status"] = "failed"
                job["message"] = job.get("message") or "Proses tuning terhenti."
                with open(job_state_path(job_id), "w", encoding="utf-8") as f:
                    json.dump(job, f, indent=2)
    return None


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


def _build_pca_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    PCA harus tetap bisa jalan meskipun sebagian kolom training tidak lengkap.
    Coba pakai builder training utama dulu; jika gagal, fallback ke builder yang
    lebih toleran dengan kolom yang tersedia.
    """
    try:
        return _build_feature_matrix(df)
    except Exception:
        fallback_df = df.copy()

        excluded_columns = {
            "ASAL SEKOLAH", "SEKOLAH", "NAMA SEKOLAH", "SCHOOL NAME",
            "NPSN", "NPSN SEKOLAH", "CLUSTER", "DBI_SCORE", "SILHOUETTE_SCORE",
            "Distance_to_Assigned_Centroid",
        }
        categorical_candidates = [
            "JURUSAN SEKOLAH",
            "PROVINSI SEKOLAH",
        ]
        categorical_cols = [col for col in categorical_candidates if col in fallback_df.columns]

        numeric_candidates = ["NILAI KESELURUHAN", "IPK"]
        numeric_cols = []
        for col in [c for c in numeric_candidates if c in fallback_df.columns]:
            col_upper = str(col).upper().strip()
            if (
                col in excluded_columns or
                col_upper in excluded_columns or
                "NPSN" in col_upper or
                col_upper.startswith("DISTANCE_TO_") or
                col_upper.startswith("MEMBERSHIP_")
            ):
                continue
            numeric_cols.append(col)

        numeric_features = fallback_df[numeric_cols].copy()
        numeric_features = numeric_features.apply(pd.to_numeric, errors="coerce")
        numeric_features = numeric_features.replace([float("inf"), float("-inf")], pd.NA)
        if not numeric_features.empty:
            numeric_features = numeric_features.fillna(numeric_features.median(numeric_only=True)).fillna(0)

        categorical_features = pd.DataFrame(index=fallback_df.index)
        if categorical_cols:
            cat_source = fallback_df[categorical_cols].copy().fillna("UNKNOWN").astype(str)
            cat_source = cat_source.apply(lambda s: s.str.strip().str.upper().replace("", "UNKNOWN"))
            categorical_features = pd.get_dummies(cat_source, columns=categorical_cols, dtype=float)

        feature_df = pd.concat([numeric_features, categorical_features], axis=1)
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        if feature_df.empty:
            raise ValueError("Tidak ada fitur yang bisa dipakai untuk PCA.")
        return feature_df

@model_bp.route('/pelatihan_model')
def pelatihan_model():
    model_list = []
    active_model = load_active_model_name()
    latest_tuning = _latest_tuning_result()
    active_training_config = load_active_training_config()
    latest_tuning_best_config = _latest_tuning_best_config()
    current_tuning = None
    running_job = _running_tuning_job()
    if running_job:
        current_tuning = {"job_id": running_job[0], **running_job[1]}

    for folder in os.listdir("models"):
        if folder.startswith("model_"):
            model_path = os.path.join("models", folder)


            meta_path = os.path.join(model_path, "meta.json")
            pkl_path = os.path.join(model_path, "hasil_kmeans_3cluster.pkl")
            uploaded_at = "-"
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                nama_data = meta.get("nama_data", "-")
                uploaded_at = meta.get("uploaded_at") or "-"
            else:
                nama_data = "-"

            if uploaded_at not in ("-", "", None):
                uploaded_at = format_datetime_jakarta(uploaded_at)

            if uploaded_at == "-":
                # Fallback untuk model lama: pakai mtime meta.json lalu pkl.
                ts = None
                if os.path.exists(meta_path):
                    ts = os.path.getmtime(meta_path)
                elif os.path.exists(pkl_path):
                    ts = os.path.getmtime(pkl_path)
                if ts is not None:
                    uploaded_at = datetime.fromtimestamp(ts, ZoneInfo("Asia/Jakarta")).strftime("%d/%m/%Y %H:%M:%S WIB")

            # Baca metrik evaluasi dari meta.json
            silhouette      = meta.get("silhouette_score")
            dbi             = meta.get("dbi_score")
            dbi_before_ga   = meta.get("dbi_before_ga")
            dbi_after_ga    = meta.get("dbi_after_ga")
            dbi_improvement = meta.get("dbi_improvement_pct")
            ch_score        = meta.get("ch_score")
            n_clusters_used = meta.get("n_clusters_used")
            elbow_data      = meta.get("elbow_data")
            avg_dbi         = meta.get("avg_dbi_after_ga_runs", meta.get("avg_dbi_after_ga_3_runs"))
            avg_silhouette  = meta.get("avg_silhouette_runs", meta.get("avg_silhouette_3_runs"))
            training_runs   = meta.get("training_runs")
            random_seed     = meta.get("random_seed")
            population_size = meta.get("population_size")
            generations     = meta.get("generations")
            mutation_rate   = meta.get("mutation_rate")
            early_mutation_rate = meta.get("early_mutation_rate")
            mid_mutation_rate   = meta.get("mid_mutation_rate")
            late_mutation_rate  = meta.get("late_mutation_rate")
            max_stagnant        = meta.get("max_stagnant")
            hyperparameter_source = meta.get("hyperparameter_source")
            is_improved     = (
                dbi_before_ga is not None and
                dbi_after_ga is not None and
                dbi_after_ga < dbi_before_ga
            )

            status = "Aktif" if folder == active_model else ""
            model_list.append({
                "nama_model":       folder,
                "nama_data":        nama_data,
                "uploaded_at":      uploaded_at,
                "status":           status,
                "silhouette_score": silhouette,
                "dbi_score":        dbi,
                "dbi_before_ga":    dbi_before_ga,
                "dbi_after_ga":     dbi_after_ga,
                "dbi_improvement_pct": dbi_improvement,
                "ch_score":         ch_score,
                "n_clusters_used":  n_clusters_used,
                "elbow_data":       elbow_data,
                "avg_dbi_after_ga_3_runs": avg_dbi,
                "avg_silhouette_3_runs": avg_silhouette,
                "avg_dbi_after_ga_runs": avg_dbi,
                "avg_silhouette_runs": avg_silhouette,
                "training_runs": training_runs,
                "random_seed": random_seed,
                "population_size": population_size,
                "generations": generations,
                "mutation_rate": mutation_rate,
                "early_mutation_rate": early_mutation_rate,
                "mid_mutation_rate": mid_mutation_rate,
                "late_mutation_rate": late_mutation_rate,
                "max_stagnant": max_stagnant,
                "hyperparameter_source": hyperparameter_source,
                "is_improved_baseline": is_improved,
            })

    model_list.sort(
        key=lambda item: (
            item["status"] == "Aktif",
            item["uploaded_at"] if item["uploaded_at"] not in ("-", None, "") else "",
        ),
        reverse=True,
    )

    return render_template(
        "pelatihan_model.html",
        model_list=model_list,
        active_model=active_model,
        latest_tuning=latest_tuning,
        latest_tuning_best_config=latest_tuning_best_config,
        active_training_config=active_training_config,
        current_tuning=current_tuning,
    )


@model_bp.route('/tuning/start', methods=['POST'])
def start_tuning():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    running_job = _running_tuning_job()
    if running_job:
        existing_id, job = running_job
        return jsonify({
            "success": True,
            "job_id": existing_id,
            "message": "Tuning sedang berjalan.",
            **job,
        })

    active_model = load_active_model_name()
    data_path = os.path.join("models", active_model or "", "data_gabungan_clean.pkl")
    if not active_model or not os.path.exists(data_path):
        return jsonify({"success": False, "error": "Data model aktif tidak ditemukan."}), 400

    job_id = uuid.uuid4().hex
    os.makedirs(JOB_STATE_DIR, exist_ok=True)
    initial_state = {
        "job_id": job_id,
        "status": "pending",
        "percent": 0,
        "message": "Menyiapkan tuning hyperparameter...",
        "completed_runs": 0,
        "total_runs": 405,
    }
    with open(job_state_path(job_id), "w", encoding="utf-8") as f:
        json.dump(initial_state, f, indent=2)

    log_path = os.path.join(JOB_STATE_DIR, f"{job_id}.log")
    with open(log_path, "ab") as log_file:
        subprocess.Popen(
            ["python3", "tune_ga_ui_worker.py", "--job-id", job_id],
            cwd=os.getcwd(),
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    return jsonify({"success": True, "job_id": job_id, "message": "Tuning dimulai."})


@model_bp.route('/tuning/progress/<job_id>')
def tuning_progress(job_id):
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    job = _read_tuning_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job tuning tidak ditemukan."}), 404
    return jsonify({"success": True, "job_id": job_id, **job})


@model_bp.route('/tuning/apply-best', methods=['POST'])
def apply_best_tuning():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    running_job = _running_tuning_job()
    if running_job:
        return jsonify({"success": False, "error": "Tuning masih berjalan. Terapkan setelah job selesai."}), 400

    best_config = _latest_tuning_best_config()
    if not best_config:
        return jsonify({"success": False, "error": "Hasil tuning terbaik belum tersedia."}), 400

    saved_config = save_active_training_config(best_config)
    return jsonify({
        "success": True,
        "message": "Konfigurasi tuning terbaik berhasil diterapkan sebagai default training.",
        "config": saved_config,
        "tuning_run_dir": best_config.get("tuning_run_dir"),
    })

@model_bp.route('/terapkan_model/<nama_model>')
def terapkan_model(nama_model):
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    active_flag_path = "models/active_model.txt"
    model_path = os.path.join("models", nama_model, "hasil_kmeans_3cluster.pkl")
    if not os.path.exists(model_path):
        flash(" Model tidak dapat diterapkan karena file hasil clustering tidak ditemukan.", "error")
        return redirect(url_for('model.pelatihan_model'))

    with open(active_flag_path, "w") as f:
        f.write(nama_model)
    flash(f" Model {nama_model} sekarang diterapkan sebagai model aktif secara manual.")
    return redirect(url_for('model.pelatihan_model'))

@model_bp.route('/hapus_model/<nama_model>')
def hapus_model(nama_model):
    if 'user' not in session:
        return redirect(url_for('auth.login'))

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
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

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

@model_bp.route('/pca_data/<nama_model>')
def pca_data(nama_model):
    """Serve PCA 2D scatter data for cluster visualization, optionally filtered by prodi."""
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    prodi = request.args.get('prodi', 'Global')
    pkl_path = os.path.join("models", nama_model, "hasil_kmeans_3cluster.pkl")

    if not os.path.exists(pkl_path):
        return jsonify({"success": False, "error": "File data model tidak ditemukan."}), 404

    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        df_all = d.get('data') if isinstance(d, dict) else d

        prodi_list = []
        if 'PROGRAM STUDI' in df_all.columns:
            prodi_list = sorted(df_all['PROGRAM STUDI'].dropna().unique().astype(str).tolist())

        df = df_all.copy()
        if prodi != 'Global' and 'PROGRAM STUDI' in df.columns:
            df = df[df['PROGRAM STUDI'] == prodi]
        df = df.reset_index(drop=True)

        if len(df) < 2:
            return jsonify({"success": False, "error": f"Data untuk {prodi} terlalu sedikit (< 2)."})

        feature_df = _build_pca_feature_matrix(df)
        if feature_df.shape[1] < 2:
            return jsonify({"success": False, "error": "Fitur tidak mencukupi untuk PCA."})

        n_comp = min(2, feature_df.shape[1])
        
        numeric_cols = [col for col in ["NILAI KESELURUHAN", "IPK"] if col in feature_df.columns]
        categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]
        
        scaled_parts = []
        if numeric_cols:
            scaled_numeric = StandardScaler().fit_transform(feature_df[numeric_cols])
            scaled_parts.append(pd.DataFrame(scaled_numeric, index=feature_df.index, columns=numeric_cols))
        if categorical_cols:
            weighted_categorical = feature_df[categorical_cols] * 0.1
            scaled_parts.append(weighted_categorical)
            
        if scaled_parts:
            feature_df_scaled = pd.concat(scaled_parts, axis=1)
        else:
            feature_df_scaled = feature_df
            
        scaled = feature_df_scaled.to_numpy()
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        pca = PCA(n_components=n_comp, random_state=42)
        coords = pca.fit_transform(scaled)

        if coords.shape[1] == 1:
            coords = np.hstack([coords, np.zeros((len(coords), 1))])

        # Ambil sampel titik secara proporsional per cluster agar cluster kecil tetap terlihat.
        max_points = min(2000, len(df))
        if 'Cluster' in df.columns and len(df) > max_points:
            sampled_indices = []
            grouped = df.groupby('Cluster').indices
            remaining = max_points
            cluster_items = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
            for pos, (_, idxs) in enumerate(cluster_items):
                clusters_left = len(cluster_items) - pos
                quota = max(1, int(round(max_points * (len(idxs) / len(df)))))
                quota = min(quota, len(idxs), remaining - max(0, clusters_left - 1))
                chosen = np.random.choice(idxs, size=quota, replace=False).tolist()
                sampled_indices.extend(chosen)
                remaining = max_points - len(sampled_indices)
                if remaining <= 0:
                    break
            if len(sampled_indices) < max_points:
                leftover = list(set(df.index.tolist()) - set(sampled_indices))
                sampled_indices.extend(np.random.choice(leftover, size=min(max_points - len(sampled_indices), len(leftover)), replace=False).tolist())
        else:
            sampled_indices = list(range(len(df))) if len(df) <= max_points else np.random.choice(len(df), size=max_points, replace=False).tolist()

        points = []
        for idx in sampled_indices:
            row = df.iloc[idx]
            points.append({
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "cluster": int(row['Cluster']) if 'Cluster' in df.columns else 0,
                "prodi": str(row.get('PROGRAM STUDI', '')),
                "sekolah": str(row.get('ASAL SEKOLAH', '')),
                "nilai": float(row.get('NILAI KESELURUHAN')) if pd.notnull(row.get('NILAI KESELURUHAN')) else None,
            })

        variance = pca.explained_variance_ratio_
        total_variance = float(sum(variance))
        if total_variance >= 0.5:
            quality_band = "strong"
            quality_label = "Representatif"
        elif total_variance >= 0.2:
            quality_band = "moderate"
            quality_label = "Cukup terbatas"
        else:
            quality_band = "weak"
            quality_label = "Sangat terbatas"

        component_names = [f"PC{i + 1}" for i in range(n_comp)]
        feature_importance = []
        for comp_idx in range(n_comp):
            loading_pairs = []
            for feature_name, loading in zip(feature_df.columns, pca.components_[comp_idx]):
                loading_pairs.append({
                    "feature": str(feature_name),
                    "loading": round(float(loading), 4),
                    "magnitude": round(float(abs(loading)), 4),
                    "component": component_names[comp_idx],
                })
            feature_importance.extend(
                sorted(loading_pairs, key=lambda item: item["magnitude"], reverse=True)[:5]
            )

        return jsonify({
            "success": True,
            "prodi_list": prodi_list,
            "points": points,
            "variance_ratio": [float(v) for v in variance],
            "total_variance_explained": total_variance,
            "n_sample": len(points),
            "n_total": len(df),
            "feature_count": int(feature_df.shape[1]),
            "quality_band": quality_band,
            "quality_label": quality_label,
            "top_features": feature_importance,
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Gagal menghitung PCA dinamis: {e}"}), 500

@model_bp.route('/comparison_data/<nama_model>')
def comparison_data(nama_model):
    """Serve per-prodi DBI comparison data for the preview modal."""
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    meta_path = os.path.join("models", nama_model, "meta.json")
    if not os.path.exists(meta_path):
        return jsonify({"success": False, "error": "meta.json tidak ditemukan."}), 404
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        rows = meta.get("dbi_comparison_per_prodi")
        if not rows:
            return jsonify({"success": False, "error": "Data perbandingan per-prodi belum tersedia. Upload ulang data untuk menghasilkan perbandingan."}), 404
        return jsonify({
            "success": True,
            "rows": rows,
            "dbi_before_ga": meta.get("dbi_before_ga"),
            "dbi_after_ga":  meta.get("dbi_after_ga"),
            "dbi_improvement_pct": meta.get("dbi_improvement_pct"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@model_bp.route('/update_model_meta', methods=['POST'])
def update_model_meta():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

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
