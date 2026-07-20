import argparse
import itertools
import json
import os
from argparse import Namespace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from tune_ga_hyperparams import (
    apply_preset, build_manifest, load_dataframe, make_output_dir,
    parse_float_list, parse_int_list, run_tuning, export_thesis_excel,
)


JOB_STATE_DIR = os.path.join("outputs", "ga_tuning", "jobs")


def ensure_job_dir():
    os.makedirs(JOB_STATE_DIR, exist_ok=True)


def job_state_path(job_id):
    return os.path.join(JOB_STATE_DIR, f"{job_id}.json")


def write_state(job_id, payload):
    ensure_job_dir()
    payload["updated_at"] = datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(timespec="seconds")
    with open(job_state_path(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ── Konfigurasi Grid Tuning (ubah di sini untuk mengubah tombol UI) ──────────
UI_TUNING_CONFIG = {
    "seeds":               "42,43,44,45,46",
    "pop_sizes":           "20,30,50",
    "generations":         "25,50",
    "early_mutation_rates": "0.5,0.3",
    "mid_mutation_rates":   "0.3,0.25",
    "late_mutation_rates":  "0.1,0.15",
    "max_stagnants":        "5,10",
}




def compute_total_runs(config):
    """Hitung total run dari konfigurasi grid secara dinamis."""
    seeds       = parse_int_list(config["seeds"])
    pop_sizes   = parse_int_list(config["pop_sizes"])
    generations = parse_int_list(config["generations"])
    e_muts      = parse_float_list(config["early_mutation_rates"])
    m_muts      = parse_float_list(config["mid_mutation_rates"])
    l_muts      = parse_float_list(config["late_mutation_rates"])
    stagnants   = parse_int_list(config["max_stagnants"])
    combos = list(itertools.product(pop_sizes, generations, e_muts, m_muts, l_muts, stagnants))
    return len(combos) * len(seeds)


def build_args():
    cfg = UI_TUNING_CONFIG
    return apply_preset(Namespace(
        data_pkl=None,
        active_model_file="models/active_model.txt",
        n_clusters=3,
        preset="balanced",
        seeds=cfg["seeds"],
        pop_sizes=cfg["pop_sizes"],
        generations=cfg["generations"],
        early_mutation_rates=cfg["early_mutation_rates"],
        mid_mutation_rates=cfg["mid_mutation_rates"],
        late_mutation_rates=cfg["late_mutation_rates"],
        mutation_rates=None,
        max_stagnants=cfg["max_stagnants"],
        output_root="outputs/ga_tuning",
        verbose=False,
    ))


def main():
    parser = argparse.ArgumentParser(description="Background worker untuk tuning GA dari UI.")
    parser.add_argument("--job-id", required=True)
    args_cli = parser.parse_args()
    job_id = args_cli.job_id

    try:
        args = build_args()
        df, data_path = load_dataframe(args)
        output_dir = make_output_dir(args.output_root)
        total_runs_estimate = compute_total_runs(UI_TUNING_CONFIG)
        write_state(job_id, {
            "job_id": job_id,
            "status": "running",
            "percent": 1,
            "message": "Memulai tuning hyperparameter GA...",
            "output_dir": output_dir,
            "data_path": data_path,
            "completed_runs": 0,
            "total_runs": total_runs_estimate,
            "pid": os.getpid(),
        })

        def progress(completed, total, info):
            percent = max(1, min(98, int((completed / total) * 100)))
            write_state(job_id, {
                "job_id": job_id,
                "status": "running",
                "percent": percent,
                "message": (
                    f"Eksperimen {completed}/{total}: "
                    f"DBI={info.get('dbi_after_ga')}, Silhouette={info.get('silhouette_score')}"
                ),
                "output_dir": output_dir,
                "data_path": data_path,
                "completed_runs": completed,
                "total_runs": total,
                "current_combo": info.get("label"),
                "pid": os.getpid(),
            })

        per_run_df, summary_df = run_tuning(df, args, output_dir, progress_callback=progress)

        per_run_csv = os.path.join(output_dir, "ga_tuning_per_run.csv")
        summary_csv = os.path.join(output_dir, "ga_tuning_summary.csv")
        summary_xlsx = os.path.join(output_dir, "ga_tuning_results.xlsx")
        manifest_json = os.path.join(output_dir, "ga_tuning_manifest.json")

        per_run_df.to_csv(per_run_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        export_thesis_excel(summary_df, per_run_df, summary_xlsx, args, data_path, len(df))

        manifest = build_manifest(args, data_path, output_dir, len(df), per_run_df, summary_df)
        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        write_state(job_id, {
            "job_id": job_id,
            "status": "completed",
            "percent": 100,
            "message": "Tuning selesai.",
            "output_dir": output_dir,
            "data_path": data_path,
            "completed_runs": len(per_run_df),
            "total_runs": len(per_run_df),
            "summary_path": summary_csv,
            "excel_path": summary_xlsx,
            "manifest_path": manifest_json,
            "best": best,
            "pid": os.getpid(),
        })
    except Exception as exc:
        write_state(job_id, {
            "job_id": job_id,
            "status": "failed",
            "percent": 100,
            "message": f"Tuning gagal: {exc}",
            "pid": os.getpid(),
        })
        raise


if __name__ == "__main__":
    main()
