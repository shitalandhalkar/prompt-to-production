"""
UC-0C — Budget Growth Analyser (ML-Enhanced)
Implements load_dataset and compute_growth skills per agents.md and skills.md,
augmented with:
  - Linear Regression trend fitting       (sklearn)
  - Isolation Forest anomaly detection    (sklearn)
  - Next-period spend forecasting         (sklearn)
"""
import argparse
import csv
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = {"period", "ward", "category", "budgeted_amount", "actual_spend", "notes"}


# ─────────────────────────────────────────────
# SKILL: load_dataset
# Reads CSV, validates columns, reports nulls before returning.
# ─────────────────────────────────────────────
def load_dataset(input_path: str):
    """
    Reads the budget CSV, validates required columns, and reports
    all null actual_spend rows (with their notes reason) upfront.
    Returns: (rows: list[dict], null_report: list[dict])
    """
    try:
        with open(input_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
                missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
                print(f"ERROR: Input CSV is missing required columns: {missing}")
                sys.exit(1)

            rows = []
            null_report = []

            for row in reader:
                raw_spend = row["actual_spend"].strip()
                row["actual_spend"] = float(raw_spend) if raw_spend else None
                row["budgeted_amount"] = float(row["budgeted_amount"].strip())
                rows.append(row)

                if row["actual_spend"] is None:
                    null_report.append({
                        "period":   row["period"],
                        "ward":     row["ward"],
                        "category": row["category"],
                        "reason":   row["notes"].strip() or "No reason provided",
                    })

        print(f"Dataset loaded: {len(rows)} rows, {len(null_report)} null actual_spend value(s).\n")

        if null_report:
            print("WARNING: NULL ROWS detected. Growth cannot be computed for these periods:")
            for n in null_report:
                print(f"   - {n['period']} | {n['ward']} | {n['category']} -> {n['reason']}")
            print()

        return rows, null_report

    except FileNotFoundError:
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)


# ─────────────────────────────────────────────
# ML LAYER 1: Linear Regression Trend Fitting
# ─────────────────────────────────────────────
def fit_trend(periods_idx: list, spends: list):
    """
    Fits a simple LinearRegression to the non-null time-indexed spend values.
    Returns: (model, r2_score, slope, intercept)
    """
    X = np.array(periods_idx).reshape(-1, 1)
    y = np.array(spends)
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    return model, r2, model.coef_[0], model.intercept_


# ─────────────────────────────────────────────
# ML LAYER 2: Isolation Forest Anomaly Detection
# ─────────────────────────────────────────────
def detect_anomalies(spends: list, contamination: float = 0.1):
    """
    Uses IsolationForest to detect anomalous spending months.
    Returns: list of bool — True if that index is anomalous.
    """
    if len(spends) < 4:
        return [False] * len(spends)

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(spends).reshape(-1, 1))
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(X)   # -1 = anomaly, 1 = normal
    return [p == -1 for p in preds]


# ─────────────────────────────────────────────
# ML LAYER 3: Next-Period Forecast
# ─────────────────────────────────────────────
def forecast_next(model: LinearRegression, last_idx: int):
    """
    Predicts the spend for the next period (last_idx + 1).
    Returns: float forecast value.
    """
    return float(model.predict([[last_idx + 1]])[0])


# ─────────────────────────────────────────────
# SKILL: compute_growth
# Strict per-ward + per-category. Never aggregates across wards.
# ─────────────────────────────────────────────
def compute_growth(rows: list, ward: str, category: str, growth_type: str) -> list:
    """
    Computes per-period growth for the requested ward+category+growth_type.
    Augments each row with ML trend, anomaly flag, and a next-period forecast.
    Strictly refuses cross-ward or cross-category aggregation.
    """
    growth_type = growth_type.upper()
    if growth_type not in ("MOM", "YOY"):
        print(f"ERROR: --growth-type must be 'MoM' or 'YoY'. Received: '{growth_type}'")
        print("Refusing to guess. Please re-run with an explicit --growth-type.")
        sys.exit(1)

    # ── Filter strictly to requested ward + category ───────────────────────
    subset = [
        r for r in rows
        if r["ward"] == ward and r["category"] == category
    ]

    if not subset:
        print(f"ERROR: No data found for ward='{ward}' / category='{category}'.")
        wards = sorted({r["ward"] for r in rows})
        cats  = sorted({r["category"] for r in rows})
        print(f"  Available wards:      {wards}")
        print(f"  Available categories: {cats}")
        sys.exit(1)

    subset.sort(key=lambda r: r["period"])

    # ── Build non-null arrays for ML ───────────────────────────────────────
    valid_idx    = []
    valid_spends = []
    for i, row in enumerate(subset):
        if row["actual_spend"] is not None:
            valid_idx.append(i)
            valid_spends.append(row["actual_spend"])

    # Fit ML models on available (non-null) data
    trend_model = None
    r2 = None
    slope = None
    intercept = None
    anomaly_map = {}   # period -> bool

    if len(valid_spends) >= 2:
        trend_model, r2, slope, intercept = fit_trend(valid_idx, valid_spends)
        anomalies = detect_anomalies(valid_spends)
        for i, idx in enumerate(valid_idx):
            anomaly_map[idx] = anomalies[i]

    # Next-period forecast
    forecast_val = None
    if trend_model is not None and valid_idx:
        forecast_val = forecast_next(trend_model, valid_idx[-1])

    # ── Compute growth per period ──────────────────────────────────────────
    results = []

    for i, row in enumerate(subset):
        period       = row["period"]
        actual_spend = row["actual_spend"]

        # Trend-predicted value for this period
        trend_pred = (
            round(float(trend_model.predict([[i]])[0]), 2)
            if trend_model is not None else None
        )
        is_anomaly = anomaly_map.get(i, False)

        # NULL guard — flag before computing (agents.md rule #2)
        if actual_spend is None:
            reason = row["notes"].strip() or "No reason provided"
            results.append({
                "period":        period,
                "ward":          ward,
                "category":      category,
                "actual_spend":  "NULL",
                "growth":        "N/A",
                "formula":       "N/A — null actual_spend",
                "trend_pred":    trend_pred if trend_pred is not None else "N/A",
                "ml_anomaly":    "N/A",
                "flag":          f"SKIPPED: {reason}",
            })
            continue

        # ── MoM growth ─────────────────────────────────────────────────────
        if growth_type == "MOM":
            if i == 0:
                results.append({
                    "period":       period,
                    "ward":         ward,
                    "category":     category,
                    "actual_spend": actual_spend,
                    "growth":       "N/A",
                    "formula":      "N/A — first period, no prior month",
                    "trend_pred":   trend_pred if trend_pred is not None else "N/A",
                    "ml_anomaly":   "YES" if is_anomaly else "no",
                    "flag":         "",
                })
                continue

            prev = subset[i - 1]
            prev_spend  = prev["actual_spend"]
            prev_period = prev["period"]

            if prev_spend is None:
                prev_reason = prev["notes"].strip() or "No reason provided"
                results.append({
                    "period":       period,
                    "ward":         ward,
                    "category":     category,
                    "actual_spend": actual_spend,
                    "growth":       "N/A",
                    "formula":      f"N/A — prior period ({prev_period}) has null actual_spend",
                    "trend_pred":   trend_pred if trend_pred is not None else "N/A",
                    "ml_anomaly":   "YES" if is_anomaly else "no",
                    "flag":         f"SKIPPED: prior month null — {prev_reason}",
                })
                continue

            growth_pct = ((actual_spend - prev_spend) / prev_spend) * 100
            formula    = f"(({actual_spend} - {prev_spend}) / {prev_spend}) x 100"
            results.append({
                "period":       period,
                "ward":         ward,
                "category":     category,
                "actual_spend": actual_spend,
                "growth":       f"{growth_pct:+.1f}%",
                "formula":      formula,
                "trend_pred":   trend_pred if trend_pred is not None else "N/A",
                "ml_anomaly":   "YES" if is_anomaly else "no",
                "flag":         "ANOMALY DETECTED" if is_anomaly else "",
            })

        # ── YoY growth — prior year data not in dataset, refuse gracefully ─
        elif growth_type == "YOY":
            results.append({
                "period":       period,
                "ward":         ward,
                "category":     category,
                "actual_spend": actual_spend,
                "growth":       "N/A",
                "formula":      "N/A — YoY requires prior-year data not present in dataset",
                "trend_pred":   trend_pred if trend_pred is not None else "N/A",
                "ml_anomaly":   "YES" if is_anomaly else "no",
                "flag":         "SKIPPED: prior year data unavailable",
            })

    # ── Append ML summary rows ─────────────────────────────────────────────
    if trend_model is not None:
        results.append({
            "period":       "ML_TREND",
            "ward":         ward,
            "category":     category,
            "actual_spend": "—",
            "growth":       "—",
            "formula":      f"LinearRegression: spend = {slope:.4f} x period_idx + {intercept:.4f}  (R2={r2:.3f})",
            "trend_pred":   "—",
            "ml_anomaly":   "—",
            "flag":         "",
        })

    if forecast_val is not None:
        results.append({
            "period":       "ML_FORECAST_NEXT",
            "ward":         ward,
            "category":     category,
            "actual_spend": "—",
            "growth":       "—",
            "formula":      f"LinearRegression prediction for period {valid_idx[-1] + 1}",
            "trend_pred":   round(forecast_val, 2),
            "ml_anomaly":   "—",
            "flag":         "",
        })

    return results


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
def write_output(results: list, output_path: str):
    if not results:
        print("No results to write.")
        return

    fieldnames = [
        "period", "ward", "category", "actual_spend",
        "growth", "formula", "trend_pred", "ml_anomaly", "flag"
    ]
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}  ({len(results)} rows)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="UC-0C Budget Growth Analyser — ML-Enhanced (LinearRegression + IsolationForest)"
    )
    parser.add_argument("--input",       required=True,  help="Path to ward_budget.csv")
    parser.add_argument("--ward",        required=True,  help="Exact ward name to filter")
    parser.add_argument("--category",    required=True,  help="Exact category name to filter")
    parser.add_argument("--growth-type", required=False, help="MoM or YoY — required, never guessed")
    parser.add_argument("--output",      required=True,  help="Path to write growth_output.csv")
    args = parser.parse_args()

    # Enforcement rule #4: refuse if --growth-type not given
    if not args.growth_type:
        print("ERROR: --growth-type was not specified.")
        print("Please re-run with:  --growth-type MoM  OR  --growth-type YoY")
        print("Refusing to guess the aggregation method.")
        sys.exit(1)

    # Skill 1: load_dataset
    rows, null_report = load_dataset(args.input)

    # Skill 2: compute_growth (ML-enhanced, strict per-ward per-category)
    results = compute_growth(rows, args.ward, args.category, args.growth_type)

    # Console summary
    print(f"\n{'Period':<20} {'Actual Spend':>14} {'Growth':>10} {'Trend Pred':>12} {'Anomaly':>8}")
    print("-" * 80)
    for r in results:
        flag_str = f"  <- {r['flag']}" if r["flag"] else ""
        print(
            f"{r['period']:<20} {str(r['actual_spend']):>14} "
            f"{str(r['growth']):>10} {str(r['trend_pred']):>12} "
            f"{str(r['ml_anomaly']):>8}{flag_str}"
        )

    # Write CSV
    write_output(results, args.output)


if __name__ == "__main__":
    main()
