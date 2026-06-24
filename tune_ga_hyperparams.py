import argparse
import contextlib
import itertools
import io
import json
import math
import os
import pickle
import statistics
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from clustering.kmeans_module import _run_kmeans_single


DEFAULT_ACTIVE_MODEL_FILE = "models/active_model.txt"
DEFAULT_OUTPUT_ROOT = "outputs/ga_tuning"
TUNING_PRESETS = {
    "quick": {
        "seeds": "42,43",
        "pop_sizes": "10,20",
        "generations": "10",
        "early_mutation_rates": "0.2,0.28",
        "mid_mutation_rates": "0.1,0.14",
        "late_mutation_rates": "0.01,0.05",
        "max_stagnants": "8",
    },
    "balanced": {
        "seeds": "42,43,44",
        "pop_sizes": "10,20",
        "generations": "10,25",
        "early_mutation_rates": "0.2,0.28",
        "mid_mutation_rates": "0.1,0.14",
        "late_mutation_rates": "0.01,0.05",
        "max_stagnants": "8",
    },
    "full": {
        "seeds": "42,43,44,45,46",
        "pop_sizes": "10,20,30",
        "generations": "10,25,50",
        "early_mutation_rates": "0.2,0.28,0.35",
        "mid_mutation_rates": "0.1,0.14,0.2",
        "late_mutation_rates": "0.01,0.05,0.1",
        "max_stagnants": "5,8,12",
    },
}


def parse_int_list(raw_value):
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_float_list(raw_value):
    return [float(part.strip()) for part in raw_value.split(",") if part.strip()]


def load_dataframe(args):
    if args.data_pkl:
        data_path = args.data_pkl
    else:
        with open(args.active_model_file, "r", encoding="utf-8") as f:
            active_model = f.read().strip()
        if not active_model:
            raise ValueError(f"File model aktif kosong: {args.active_model_file}")
        data_path = os.path.join("models", active_model, "data_gabungan_clean.pkl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data tidak ditemukan: {data_path}")

    with open(data_path, "rb") as f:
        payload = pickle.load(f)

    df = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Payload pada {data_path} bukan DataFrame yang valid.")

    return df.copy(), data_path


def make_output_dir(output_root):
    timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def safe_mean(values):
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return None
    return float(statistics.mean(valid))


def safe_std(values):
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if len(valid) < 2:
        return 0.0 if valid else None
    return float(statistics.pstdev(valid))


def rounded(value, digits=4):
    if value is None:
        return None
    return round(value, digits)


def validate_rates(label, values):
    invalid = [value for value in values if value < 0 or value > 1]
    if invalid:
        raise ValueError(f"{label} harus berada pada rentang 0..1. Nilai tidak valid: {invalid}")


def apply_preset(args):
    if (
        args.mutation_rates
        and args.early_mutation_rates is None
        and args.mid_mutation_rates is None
        and args.late_mutation_rates is None
    ):
        # Classic GA tuning mode: one mutation_rate value is applied to all phases.
        args.early_mutation_rates = args.mutation_rates
        args.mid_mutation_rates = args.mutation_rates
        args.late_mutation_rates = args.mutation_rates
        args.mutation_mode = "single_rate_all_phases"
    else:
        args.mutation_mode = "phase_schedule"

    preset = TUNING_PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.mutation_rates and args.mutation_mode != "single_rate_all_phases":
        args.late_mutation_rates = args.mutation_rates
    return args


def rank_summary(df_summary):
    if df_summary.empty:
        return df_summary

    ranked = df_summary.sort_values(
        by=[
            "mean_dbi_after_ga",
            "mean_silhouette",
            "mean_dbi_improvement_pct",
            "mean_ch_score",
            "mean_runtime_seconds",
        ],
        ascending=[True, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def run_tuning(df, args, output_dir, progress_callback=None):
    seeds = parse_int_list(args.seeds)
    pop_sizes = parse_int_list(args.pop_sizes)
    generations_list = parse_int_list(args.generations)
    early_mutation_rates = parse_float_list(args.early_mutation_rates)
    mid_mutation_rates = parse_float_list(args.mid_mutation_rates)
    late_mutation_rates = parse_float_list(args.late_mutation_rates)
    max_stagnants = parse_int_list(args.max_stagnants)

    validate_rates("early_mutation_rates", early_mutation_rates)
    validate_rates("mid_mutation_rates", mid_mutation_rates)
    validate_rates("late_mutation_rates", late_mutation_rates)

    if args.mutation_mode == "single_rate_all_phases":
        mutation_schedules = [(rate, rate, rate) for rate in late_mutation_rates]
    else:
        mutation_schedules = list(itertools.product(
            early_mutation_rates,
            mid_mutation_rates,
            late_mutation_rates,
        ))

    combinations = [
        (pop_size, generations, early_rate, mid_rate, late_rate, max_stagnant)
        for pop_size, generations, (early_rate, mid_rate, late_rate), max_stagnant
        in itertools.product(pop_sizes, generations_list, mutation_schedules, max_stagnants)
    ]
    total_runs = len(combinations) * len(seeds)

    print(f"Dataset: {len(df)} baris")
    print(f"Mode mutation rate: {args.mutation_mode}")
    print(f"Kombinasi hyperparameter: {len(combinations)}")
    print(f"Jumlah seed per kombinasi: {len(seeds)}")
    print(f"Total eksekusi: {total_runs}")
    print(f"Output directory: {output_dir}")

    per_run_rows = []
    summary_rows = []
    completed_runs = 0

    for combo_index, (
        pop_size,
        generations,
        early_mutation_rate,
        mid_mutation_rate,
        late_mutation_rate,
        max_stagnant,
    ) in enumerate(combinations, start=1):
        combo_label = (
            f"pop={pop_size}, gen={generations}, "
            f"mut=({early_mutation_rate}/{mid_mutation_rate}/{late_mutation_rate}), "
            f"stagnant={max_stagnant}"
        )
        print(f"\n[{combo_index}/{len(combinations)}] Menjalankan {combo_label}")

        combo_results = []
        combo_runtimes = []

        for seed in seeds:
            start = time.perf_counter()
            run_kwargs = {
                "df": df,
                "n_clusters": args.n_clusters,
                "random_seed": seed,
                "ga_pop_size": pop_size,
                "ga_generations": generations,
                "ga_mutation_rate": late_mutation_rate,
                "ga_max_stagnant": max_stagnant,
                "ga_early_mutation_rate": early_mutation_rate,
                "ga_mid_mutation_rate": mid_mutation_rate,
                "ga_late_mutation_rate": late_mutation_rate,
            }
            if args.verbose:
                result = _run_kmeans_single(**run_kwargs)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = _run_kmeans_single(**run_kwargs)
            runtime_seconds = time.perf_counter() - start
            combo_results.append(result)
            combo_runtimes.append(runtime_seconds)
            completed_runs += 1

            per_run_rows.append({
                "population_size": pop_size,
                "generations": generations,
                "early_mutation_rate": early_mutation_rate,
                "mid_mutation_rate": mid_mutation_rate,
                "late_mutation_rate": late_mutation_rate,
                "mutation_rate": late_mutation_rate,
                "max_stagnant": max_stagnant,
                "seed": seed,
                "runtime_seconds": round(runtime_seconds, 4),
                "dbi_before_ga": result.get("dbi_before_ga"),
                "dbi_after_ga": result.get("dbi_after_ga"),
                "dbi_improvement_pct": result.get("dbi_improvement_pct"),
                "silhouette_score": result.get("silhouette_score"),
                "ch_score": result.get("ch_score"),
                "total_inertia": result.get("total_inertia"),
                "final_k": result.get("final_k"),
                "n_clusters_used": result.get("n_clusters_used"),
            })
            print(
                f"  Seed {seed} selesai "
                f"({completed_runs}/{total_runs}) | "
                f"DBI={result.get('dbi_after_ga')} | "
                f"Silhouette={result.get('silhouette_score')} | "
                f"{runtime_seconds:.2f}s"
            )
            if progress_callback:
                progress_callback(
                    completed_runs,
                    total_runs,
                    {
                        "combo_index": combo_index,
                        "combo_total": len(combinations),
                        "seed": seed,
                        "dbi_after_ga": result.get("dbi_after_ga"),
                        "silhouette_score": result.get("silhouette_score"),
                        "runtime_seconds": runtime_seconds,
                        "label": combo_label,
                    },
                )

        summary_rows.append({
            "population_size": pop_size,
            "generations": generations,
            "early_mutation_rate": early_mutation_rate,
            "mid_mutation_rate": mid_mutation_rate,
            "late_mutation_rate": late_mutation_rate,
            "mutation_rate": late_mutation_rate,
            "max_stagnant": max_stagnant,
            "n_runs": len(combo_results),
            "mean_runtime_seconds": rounded(safe_mean(combo_runtimes), 4),
            "std_runtime_seconds": rounded(safe_std(combo_runtimes), 4),
            "mean_dbi_before_ga": rounded(safe_mean([r.get("dbi_before_ga") for r in combo_results]), 4),
            "std_dbi_before_ga": rounded(safe_std([r.get("dbi_before_ga") for r in combo_results]), 4),
            "mean_dbi_after_ga": rounded(safe_mean([r.get("dbi_after_ga") for r in combo_results]), 4),
            "std_dbi_after_ga": rounded(safe_std([r.get("dbi_after_ga") for r in combo_results]), 4),
            "mean_dbi_improvement_pct": rounded(safe_mean([r.get("dbi_improvement_pct") for r in combo_results]), 4),
            "std_dbi_improvement_pct": rounded(safe_std([r.get("dbi_improvement_pct") for r in combo_results]), 4),
            "mean_silhouette": rounded(safe_mean([r.get("silhouette_score") for r in combo_results]), 4),
            "std_silhouette": rounded(safe_std([r.get("silhouette_score") for r in combo_results]), 4),
            "mean_ch_score": rounded(safe_mean([r.get("ch_score") for r in combo_results]), 4),
            "std_ch_score": rounded(safe_std([r.get("ch_score") for r in combo_results]), 4),
            "best_single_run_dbi": min(
                [r.get("dbi_after_ga") for r in combo_results if r.get("dbi_after_ga") is not None],
                default=None,
            ),
            "best_single_run_silhouette": max(
                [r.get("silhouette_score") for r in combo_results if r.get("silhouette_score") is not None],
                default=None,
            ),
        })

    return pd.DataFrame(per_run_rows), rank_summary(pd.DataFrame(summary_rows))


def build_manifest(args, data_path, output_dir, n_data_rows, per_run_df, summary_df):
    best_config = summary_df.iloc[0].to_dict() if not summary_df.empty else None
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(timespec="seconds"),
        "data_path": data_path,
        "n_data_rows": int(n_data_rows),
        "n_executions": int(len(per_run_df.index)) if not per_run_df.empty else 0,
        "n_clusters": args.n_clusters,
        "seeds": parse_int_list(args.seeds),
        "pop_sizes": parse_int_list(args.pop_sizes),
        "generations": parse_int_list(args.generations),
        "early_mutation_rates": parse_float_list(args.early_mutation_rates),
        "mid_mutation_rates": parse_float_list(args.mid_mutation_rates),
        "late_mutation_rates": parse_float_list(args.late_mutation_rates),
        "mutation_mode": args.mutation_mode,
        "max_stagnants": parse_int_list(args.max_stagnants),
        "output_dir": output_dir,
        "notes": [
            "Pipeline tuning memakai _run_kmeans_single dari clustering.kmeans_module.",
            "Adaptive mutation schedule dituning eksplisit per fase: early, mid, dan late mutation rate.",
            "Baseline KMeans di pipeline memakai random_state tetap 42, bukan seed tuning per run.",
        ],
        "best_config": best_config,
        "files": {
            "per_run_csv": os.path.join(output_dir, "ga_tuning_per_run.csv"),
            "summary_csv": os.path.join(output_dir, "ga_tuning_summary.csv"),
            "summary_xlsx": os.path.join(output_dir, "ga_tuning_results.xlsx"),
            "manifest_json": os.path.join(output_dir, "ga_tuning_manifest.json"),
        },
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Grid search hyperparameter Genetic Algorithm untuk pipeline KMeans + GA."
    )
    parser.add_argument("--data-pkl", help="Path ke data_gabungan_clean.pkl. Default: model aktif.")
    parser.add_argument("--active-model-file", default=DEFAULT_ACTIVE_MODEL_FILE)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--preset", choices=sorted(TUNING_PRESETS), default="balanced")
    parser.add_argument("--seeds")
    parser.add_argument("--pop-sizes")
    parser.add_argument("--generations")
    parser.add_argument("--early-mutation-rates")
    parser.add_argument("--mid-mutation-rates")
    parser.add_argument("--late-mutation-rates")
    parser.add_argument("--mutation-rates", help="Deprecated: alias untuk --late-mutation-rates.")
    parser.add_argument("--max-stagnants")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verbose", action="store_true", help="Tampilkan log detail setiap prodi selama tuning.")
    args = apply_preset(parser.parse_args())

    df, data_path = load_dataframe(args)
    output_dir = make_output_dir(args.output_root)

    per_run_df, summary_df = run_tuning(df, args, output_dir)

    per_run_csv = os.path.join(output_dir, "ga_tuning_per_run.csv")
    summary_csv = os.path.join(output_dir, "ga_tuning_summary.csv")
    summary_xlsx = os.path.join(output_dir, "ga_tuning_results.xlsx")
    manifest_json = os.path.join(output_dir, "ga_tuning_manifest.json")

    per_run_df.to_csv(per_run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(summary_xlsx) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_run_df.to_excel(writer, sheet_name="per_run", index=False)

    manifest = build_manifest(args, data_path, output_dir, len(df), per_run_df, summary_df)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nTuning selesai.")
    print(f"Per-run CSV : {per_run_csv}")
    print(f"Summary CSV : {summary_csv}")
    print(f"Excel       : {summary_xlsx}")
    print(f"Manifest    : {manifest_json}")

    if not summary_df.empty:
        best = summary_df.iloc[0]
        print("\nKonfigurasi terbaik berdasarkan ranking summary:")
        print(
            f"rank={int(best['rank'])}, pop={int(best['population_size'])}, "
            f"gen={int(best['generations'])}, "
            f"mut=({best['early_mutation_rate']}/{best['mid_mutation_rate']}/{best['late_mutation_rate']}), "
            f"stagnant={int(best['max_stagnant'])}, "
            f"mean_dbi={best['mean_dbi_after_ga']}, "
            f"mean_silhouette={best['mean_silhouette']}"
        )


if __name__ == "__main__":
    main()
