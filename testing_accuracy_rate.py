#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping test: EBT ↔ ABT accuracy from a CSV.
- Expects columns EBT_min and ABT_min (minutes).
- Outputs summary CSVs + prints metrics.
"""

import argparse
import os
import math
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sessions.csv")
    ap.add_argument("--outdir", default=".", help="Output folder for reports")
    ap.add_argument("--save-plots", action="store_true", help="Save PNG plots")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # --- Detect columns ---
    def pick(col_name):
        return col_name if col_name in df.columns else None

    ebt_col = pick("EBT_min")
    abt_col = pick("ABT_min")
    emp_col = pick("employee_id")
    site_col = pick("site")

    if ebt_col is None or abt_col is None:
        raise ValueError(
            f"Required columns not found. Got columns={list(df.columns)}. "
            f"Expected EBT_min and ABT_min (minutes)."
        )

    # --- Clean / numeric ---
    df[ebt_col] = pd.to_numeric(df[ebt_col], errors="coerce")
    df[abt_col] = pd.to_numeric(df[abt_col], errors="coerce")
    df = df.dropna(subset=[ebt_col, abt_col]).copy()

    # --- Per-session metrics (minutes) ---
    df["error_min"] = (df[abt_col] - df[ebt_col]).abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = 1 - (df["error_min"] / df[abt_col].replace(0, np.nan))
    df["accuracy"] = np.clip(acc, 0, 1)

    # --- Global metrics ---
    mae_min = float(df["error_min"].mean())
    rmse_min = float(np.sqrt(((df[abt_col]-df[ebt_col])**2).mean()))
    bias_min = float((df[ebt_col] - df[abt_col]).mean())
    overall_accuracy = float(df["accuracy"].mean()) * 100.0

    if df[ebt_col].std(ddof=0) > 0 and df[abt_col].std(ddof=0) > 0:
        r = float(np.corrcoef(df[ebt_col], df[abt_col])[0,1])
        r2 = float(r**2)
    else:
        r = float("nan")
        r2 = float("nan")

    summary = pd.DataFrame([
        {"metric":"overall_accuracy_pct", "value":round(overall_accuracy,2)},
        {"metric":"mae_minutes",          "value":round(mae_min,3)},
        {"metric":"rmse_minutes",         "value":round(rmse_min,3)},
        {"metric":"bias_minutes_mean(EBT-ABT)", "value":round(bias_min,3)},
        {"metric":"pearson_r",            "value":round(r,4) if not math.isnan(r) else None},
        {"metric":"r2",                   "value":round(r2,4) if not math.isnan(r2) else None},
        {"metric":"rows",                 "value":len(df)},
    ])

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "mapping_report.csv")
    summary.to_csv(summary_path, index=False)

    # --- Group summaries ---
    if emp_col:
        emp_stats = (df.groupby(emp_col)
            .agg(sessions=(abt_col,"count"),
                 accuracy_pct=("accuracy", lambda s: float(np.mean(s)*100.0)),
                 mae_min=("error_min","mean"),
                 bias_min=((ebt_col, lambda s: (s - df.loc[s.index, abt_col]).mean())))
            .reset_index())
        emp_stats["accuracy_pct"] = emp_stats["accuracy_pct"].round(2)
        emp_stats["mae_min"] = emp_stats["mae_min"].round(3)
        emp_stats["bias_min"] = emp_stats["bias_min"].round(3)
        emp_out = os.path.join(args.outdir, "mapping_by_employee.csv")
        emp_stats.to_csv(emp_out, index=False)

    if site_col:
        site_stats = (df.groupby(site_col)
            .agg(sessions=(abt_col,"count"),
                 accuracy_pct=("accuracy", lambda s: float(np.mean(s)*100.0)),
                 mae_min=("error_min","mean"),
                 bias_min=((ebt_col, lambda s: (s - df.loc[s.index, abt_col]).mean())))
            .reset_index())
        site_stats["accuracy_pct"] = site_stats["accuracy_pct"].round(2)
        site_stats["mae_min"] = site_stats["mae_min"].round(3)
        site_stats["bias_min"] = site_stats["bias_min"].round(3)
        site_out = os.path.join(args.outdir, "mapping_by_site.csv")
        site_stats.to_csv(site_out, index=False)

    # --- Optional plots ---
    if args.save_plots:
        import matplotlib.pyplot as plt
        # Error histogram
        plt.figure()
        plt.hist(df["error_min"].dropna().values, bins=30)
        plt.title("Mapping Error Distribution (|ABT - EBT|) — minutes")
        plt.xlabel("Minutes")
        plt.ylabel("Sessions")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "hist_error.png"), dpi=160)

        # Scatter EBT vs ABT with y=x
        plt.figure()
        plt.scatter(df[ebt_col].values, df[abt_col].values, alpha=0.6)
        mn = float(np.nanmin([df[ebt_col].min(), df[abt_col].min()]))
        mx = float(np.nanmax([df[ebt_col].max(), df[abt_col].max()]))
        xs = np.linspace(mn, mx, 200)
        plt.plot(xs, xs, linewidth=1.5)
        plt.title("EBT vs ABT (minutes)")
        plt.xlabel(ebt_col)
        plt.ylabel(abt_col)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "scatter_ebt_abt.png"), dpi=160)

    print("\n=== Mapping summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    if emp_col: print(f"Saved: {os.path.join(args.outdir, 'mapping_by_employee.csv')}")
    if site_col: print(f"Saved: {os.path.join(args.outdir, 'mapping_by_site.csv')}")
    if args.save_plots:
        print(f"Saved: hist_error.png, scatter_ebt_abt.png in {args.outdir}")

if __name__ == "__main__":
    main()
