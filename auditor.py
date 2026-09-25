# ==============================================================================
# AUTO-GENERATED FROM THE LIVE run_audit() FUNCTION IN THIS NOTEBOOK.
# Do not hand-edit this file. Re-run the "regenerate auditor.py" cell to
# refresh it — it always mirrors run_audit() exactly (via inspect.getsource),
# so the notebook function and this script can never silently diverge again.
# ==============================================================================
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import linregress

def run_audit(
    energy_csv="lecture_build_mains.csv",
    weather_csv="weather.csv",
    building_id="IBLEND_Lecture_Building",
    tariff_config=None,
    output_dir="artifacts/lecture_2016_09_16_2016_10_01"
):
    print("=" * 75)
    print(f"EXECUTING WATTSENSE UNIFIED AUDIT PIPELINE FOR {building_id}")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Ingestion & Explicit Timezone Alignment (Asia/Kolkata)
    # --------------------------------------------------------------------------
    print("Step 1: Ingesting telemetry and synchronizing timezone to Asia/Kolkata...")
    df_raw = pd.read_csv(energy_csv, usecols=['timestamp', 'power'])

    df_raw['timestamp_utc'] = pd.to_datetime(df_raw['timestamp'], unit='s', utc=True)
    df_raw['timestamp_ist'] = df_raw['timestamp_utc'].dt.tz_convert('Asia/Kolkata')
    df_raw['timestamp'] = df_raw['timestamp_ist'].dt.tz_localize(None)

    hourly_energy = (
        df_raw.set_index('timestamp')[['power']]
        .resample('1h')
        .mean()
        .reset_index()
    )
    hourly_energy['actual_kwh'] = (hourly_energy['power'] / 1000.0).interpolate(method='linear', limit=2).fillna(0.0)

    df_w = pd.read_csv(weather_csv)
    w_time = [c for c in df_w.columns if "time" in c.lower() or "date" in c.lower()][0]
    t_col = [c for c in df_w.columns if "temp" in c.lower()][0]
    df_w['timestamp'] = pd.to_datetime(df_w[w_time])

    hourly_weather = (
        df_w.set_index('timestamp')[[t_col]]
        .resample('1h')
        .mean()
        .reset_index()
    )
    hourly_weather.rename(columns={t_col: 'temperature'}, inplace=True)

    df_merged = pd.merge(hourly_energy[['timestamp', 'actual_kwh']], hourly_weather, on='timestamp', how='inner')
    df_merged['temperature'] = df_merged['temperature'].interpolate(method='linear', limit=2).bfill()

    df_merged['hour_of_day'] = df_merged['timestamp'].dt.hour
    df_merged['day_of_week'] = df_merged['timestamp'].dt.dayofweek
    df_merged['hour_of_week'] = (df_merged['day_of_week'] * 24) + df_merged['hour_of_day']

    # --------------------------------------------------------------------------
    # 2. Chronological Split (63 Days Calibration, 16 Days Audit)
    # --------------------------------------------------------------------------
    train_start = pd.to_datetime('2016-07-15 00:00:00')
    train_end   = pd.to_datetime('2016-09-15 23:00:00')
    audit_start = pd.to_datetime('2016-09-16 00:00:00')
    audit_end   = pd.to_datetime('2016-10-01 23:00:00')

    calib_df = df_merged[(df_merged['timestamp'] >= train_start) & (df_merged['timestamp'] <= train_end)].copy()
    audit_df = df_merged[(df_merged['timestamp'] >= audit_start) & (df_merged['timestamp'] <= audit_end)].copy()

    # --------------------------------------------------------------------------
    # 3. Fit Cyclical Hour-of-Week Median Baseline & MAD
    # --------------------------------------------------------------------------
    print("Step 2: Fitting Hour-of-Week median baseline and MAD noise floor...")
    profile = calib_df.groupby('hour_of_week')['actual_kwh'].agg(baseline_kwh='median').reset_index()

    def calc_mad(series):
        med = np.median(series)
        return np.median(np.abs(series - med))

    mad_df = calib_df.groupby('hour_of_week')['actual_kwh'].apply(calc_mad).reset_index(name='baseline_mad')
    profile = pd.merge(profile, mad_df, on='hour_of_week')
    profile['baseline_mad'] = profile['baseline_mad'].apply(lambda x: max(x, 0.20))

    # Standard error of median baseline per hour (used for the anomaly-excess uncertainty band)
    profile['baseline_se'] = (1.253 * (profile['baseline_mad'] * 1.4826)) / np.sqrt(9.0)

    # --------------------------------------------------------------------------
    # 4. Out-of-Sample Weather Model Selection & Error Diagnostics
    #    Weather is fit and validated on every run and reported as an optional
    #    contextual feature. It is only USED in final_expected_kwh when it
    #    demonstrably reduces held-out RMSE by >= 2%. For the current Lecture
    #    building, the held-out weather model has repeatedly underperformed the
    #    plain hour-of-week baseline, so it is expected to stay disabled here —
    #    but the decision is re-checked from data on every run, not hardcoded.
    # --------------------------------------------------------------------------
    print("Step 3: Conducting out-of-sample weather validation and CV(RMSE) analysis...")
    BASE_TEMP = 18.3
    calib_merged = pd.merge(calib_df, profile, on='hour_of_week')
    calib_merged['cdh'] = np.maximum(0.0, calib_merged['temperature'] - BASE_TEMP)
    residuals_train = calib_merged['actual_kwh'] - calib_merged['baseline_kwh']

    slope, intercept, r_val, p_val, std_err = linregress(calib_merged['cdh'], residuals_train)
    weather_beta_contextual = float(slope)  # always computed & reported, whether or not it's applied

    scored = pd.merge(audit_df, profile, on='hour_of_week', how='left')
    scored['cdh'] = np.maximum(0.0, scored['temperature'] - BASE_TEMP)
    scored['weather_expected_kwh'] = scored['baseline_kwh'] + (weather_beta_contextual * scored['cdh'])

    y_true = scored['actual_kwh'].values
    y_base = scored['baseline_kwh'].values
    y_weath = scored['weather_expected_kwh'].values
    mean_actual_all = np.mean(y_true)

    rmse_base = float(np.sqrt(np.mean((y_true - y_base) ** 2)))
    rmse_weath = float(np.sqrt(np.mean((y_true - y_weath) ** 2)))
    cv_rmse_all = float((rmse_base / mean_actual_all) * 100.0)

    # Occupied-only evaluation (Mon-Fri 08:00 to 18:00) — computed fresh from THIS run
    occ_mask = (scored['day_of_week'] < 5) & (scored['hour_of_day'] >= 8) & (scored['hour_of_day'] <= 18)
    y_true_occ = scored.loc[occ_mask, 'actual_kwh'].values
    y_base_occ = scored.loc[occ_mask, 'baseline_kwh'].values
    mean_actual_occ = np.mean(y_true_occ)
    rmse_base_occ = float(np.sqrt(np.mean((y_true_occ - y_base_occ) ** 2)))
    cv_rmse_occ = float((rmse_base_occ / mean_actual_occ) * 100.0)

    rmse_imp_pct = ((rmse_base - rmse_weath) / rmse_base) * 100.0
    ENABLE_WEATHER = (rmse_imp_pct >= 2.0) and (weather_beta_contextual > 0.0)

    weather_beta_applied = weather_beta_contextual if ENABLE_WEATHER else 0.0
    weather_flag = bool(ENABLE_WEATHER)
    decision = (
        f"Weather model accepted ({rmse_imp_pct:.2f}% RMSE reduction) and applied to final_expected_kwh."
        if ENABLE_WEATHER
        else f"Weather bypassed ({rmse_imp_pct:+.2f}% RMSE delta on held-out data; baseline-only model selected). "
             f"Weather beta retained below as an optional contextual feature, not used in scoring."
    )

    scored['final_expected_kwh'] = scored['weather_expected_kwh'] if ENABLE_WEATHER else scored['baseline_kwh']
    scored['residual_kwh'] = scored['actual_kwh'] - scored['final_expected_kwh']

    # --------------------------------------------------------------------------
    # 5. Transparent Anomaly Detection (unchanged model — no new ML algorithm)
    # --------------------------------------------------------------------------
    print("Step 4: Executing robust anomaly detection...")
    scored['anomaly_score'] = scored['residual_kwh'] / (1.4826 * scored['baseline_mad'])
    scored['anomaly'] = (scored['anomaly_score'] > 2.5) & (scored['residual_kwh'] > 0.50)

    def classify_meter_anomaly(row):
        if not row['anomaly']:
            return "Normal"
        if row['day_of_week'] in [5, 6]:
            return "Weekend excess load"
        if not (8 <= row['hour_of_day'] <= 19):
            return "After-hours excess load"
        return "Daytime operational excess"

    scored['anomaly_type'] = scored.apply(classify_meter_anomaly, axis=1)

    # --------------------------------------------------------------------------
    # 6. Anomaly-Excess Quantification (no longer called "CalTRACK savings")
    #
    #    NOTE (dev comment only — nothing below is surfaced in JSON/report):
    #    a true CalTRACK figure needs eemeter==4.1.1's pre/post-intervention
    #    comparison, which this single continuous monitoring period cannot
    #    supply. The outward-facing deliverables below never name
    #    "CalTRACK" or "eemeter".
    # --------------------------------------------------------------------------
    print("Step 5: Quantifying anomaly excess energy and uncertainty...")
    anomalies_df = scored[scored['anomaly']].copy()

    # Raw, audit-window (16-day) anomaly excess — a MEASURED quantity, not projected.
    anomaly_excess_kwh = float(anomalies_df['residual_kwh'].sum())

    # Uncertainty at 90% confidence around the anomaly-excess estimate itself
    se_anomaly_excess = float(np.sqrt(np.sum(anomalies_df['baseline_se'] ** 2))) if len(anomalies_df) else 0.0
    anomaly_excess_uncertainty_kwh = float(1.645 * se_anomaly_excess)
    anomaly_excess_uncertainty_pct = (
        float((anomaly_excess_uncertainty_kwh / anomaly_excess_kwh) * 100.0) if anomaly_excess_kwh > 0 else 0.0
    )

    # 365/16-style extrapolation — explicitly labeled PROJECTED/ANNUALIZED, never "verified savings"
    duration_days = max(1.0, len(audit_df) / 24.0)
    annual_mult = 365.0 / duration_days
    annualized_anomaly_excess_kwh = float(anomaly_excess_kwh * annual_mult)

    # Real, certified savings: not computed in this run (see comment above).
    estimated_savings_kwh = None
    savings_status = "not_computed"
    savings_reason = (
        "Verified savings require comparing a pre-intervention baseline period against a separate "
        "post-intervention reporting period, evaluated against an independent counterfactual model "
        "(standard M&V practice). This run covers a single continuous monitoring period with no "
        "intervention/retrofit boundary, so that comparison is not available yet. Use anomaly_excess_kwh "
        "(measured, audit-window) or annualized_anomaly_excess_kwh (projected) — neither is a savings "
        "certification."
    )

    # --------------------------------------------------------------------------
    # 7. Transparent Illustrative Tariff & Cost Impact
    # --------------------------------------------------------------------------
    tariff = tariff_config or {
        "tariff_source": "User-configured illustrative tariff",
        "currency": "INR",
        "base_rate_per_kwh": 9.50,
        "cost_metric_label": "Estimated energy-cost impact based on configured tariff"
    }

    base_rate = tariff["base_rate_per_kwh"]
    scored['estimated_cost_loss'] = np.where(
        scored['anomaly'],
        np.maximum(0.0, scored['residual_kwh']) * base_rate,
        0.0
    )

    total_actual = float(scored['actual_kwh'].sum())
    total_expected = float(scored['final_expected_kwh'].sum())
    total_cost_impact = float(scored['estimated_cost_loss'].sum())

    # Per-category anomaly excess is calculated from actual residual energy,
    # not from the number of anomaly hours. This prevents equal-weighting
    # anomalies that may have very different magnitudes.
    category_excess_kwh = (
        anomalies_df.groupby('anomaly_type')['residual_kwh'].sum().to_dict()
        if len(anomalies_df)
        else {}
    )
    counts = anomalies_df['anomaly_type'].value_counts().to_dict()

    def category_annual_impact(category):
        audit_excess = float(category_excess_kwh.get(category, 0.0))
        annual_kwh = audit_excess * annual_mult
        annual_cost = annual_kwh * base_rate
        return annual_kwh, annual_cost

    # --------------------------------------------------------------------------
    # 8. Recommendation Ranking (transparent, reproducible — no black box)
    #
    #    Ranking rule (documented in output as `ranking_method` per item and
    #    `recommendation_ranking_methodology` at the top level):
    #      - No-capex actions (implementation_cost == 0): ranked by
    #        estimated recurring annual cost-savings impact, weighted by an
    #        evidence/confidence multiplier (High=1.00, Medium=0.75, Low=0.50).
    #      - Capex actions (implementation_cost > 0): ranked by ROI
    #        (annual cost savings / implementation cost), equivalent to
    #        inverse payback — shorter payback ranks higher.
    # --------------------------------------------------------------------------
    print("Step 6: Ranking recommendations transparently...")
    CONFIDENCE_WEIGHT = {"High": 1.00, "Medium": 0.75, "Low": 0.50}

    candidate_recs = [
        {
            "action": "Investigate recurring weekend excess load",
            "scope": "Building-level operating schedule verification",
            "pattern_observed": f"Observed {counts.get('Weekend excess load', 0)} hours of excess load during weekend periods.",
            "estimated_annual_impact_kwh": round(category_annual_impact('Weekend excess load')[0], 2),
            "estimated_annual_cost_savings": round(category_annual_impact('Weekend excess load')[1], 2),
            "implementation_cost": 0.0,
            "payback_months": None,
            "payback_display": "Immediate (No-capex operational adjustment)",
            "confidence": "High"
        },
        {
            "action": "Review after-hours baseline load and shutdown schedule",
            "scope": "Building-level evening shutdown procedures",
            "pattern_observed": f"Observed {counts.get('After-hours excess load', 0)} hours of excess load outside the defined 08:00-19:00 operating window.",
            "estimated_annual_impact_kwh": round(category_annual_impact('After-hours excess load')[0], 2),
            "estimated_annual_cost_savings": round(category_annual_impact('After-hours excess load')[1], 2),
            "implementation_cost": 0.0,
            "payback_months": None,
            "payback_display": "Immediate (No-capex operational adjustment)",
            "confidence": "Medium"
        }
    ]

    def priority_score(rec):
        if rec["implementation_cost"] and rec["implementation_cost"] > 0:
            return rec["estimated_annual_cost_savings"] / rec["implementation_cost"]  # ROI, higher = better
        return rec["estimated_annual_cost_savings"] * CONFIDENCE_WEIGHT.get(rec["confidence"], 0.5)

    candidate_recs.sort(key=priority_score, reverse=True)
    for idx, rec in enumerate(candidate_recs, start=1):
        rec["rank"] = idx
        rec["ranking_score"] = round(priority_score(rec), 3)
        rec["ranking_method"] = (
            "ROI = annual cost savings / implementation cost (shorter payback ranks higher)"
            if rec["implementation_cost"] > 0
            else "Estimated recurring annual cost-savings impact x evidence/confidence weight"
        )
    recommendations = candidate_recs

    # --------------------------------------------------------------------------
    # 9. Single Source of Truth Output Construction
    #    Everything below is derived from THIS run only — the JSON, CSV and
    #    report are all written from the same `scored` / `canonical_json`
    #    objects, with nothing recomputed downstream.
    # --------------------------------------------------------------------------
    canonical_json = {
        "building_id": building_id,
        "timezone": "Asia/Kolkata (IST)",
        "analysis_period": {
            "start": str(audit_df['timestamp'].min()),
            "end": str(audit_df['timestamp'].max()),
            "duration_hours": len(audit_df),
            "duration_days": round(duration_days, 2)
        },
        "model_calibration": {
            "start": str(train_start),
            "end": str(train_end),
            "duration_hours": len(calib_df),
            "n_weekly_cycles": 9
        },
        "model_accuracy": {
            "cv_rmse_all_hours_pct": round(cv_rmse_all, 2),
            "cv_rmse_occupied_hours_pct": round(cv_rmse_occ, 2),
            "rmse_kwh": round(rmse_base, 4),
            "evaluation_note": (
                f"All-hours CV(RMSE) is elevated due to near-zero nighttime/weekend loads deflating the mean "
                f"denominator. Occupied hours (Mon-Fri 08:00-18:00) show {cv_rmse_occ:.2f}% CV(RMSE), computed "
                f"fresh from this run's held-out audit window."
            )
        },
        "weather_context": {
            "weather_used_in_final_model": weather_flag,
            "weather_beta_contextual": round(weather_beta_contextual, 5),
            "weather_beta_applied": round(weather_beta_applied, 5),
            "base_temp_c": BASE_TEMP,
            "note": "weather_beta_contextual is always computed and reported as an optional feature for future "
                    "buildings/periods where it may help; it is only folded into final_expected_kwh when "
                    "weather_used_in_final_model is true.",
            "validation_diagnostics": {
                "in_sample_r2": round(float(r_val ** 2), 4),
                "in_sample_p_value": round(float(p_val), 5),
                "rmse_improvement_pct": round(rmse_imp_pct, 2),
                "decision": decision
            }
        },
        "tariff_configuration": tariff,
        "energy_metrics": {
            "actual_kwh": round(total_actual, 2),
            "expected_baseline_kwh": round(total_expected, 2),
            "estimated_cost_impact": round(total_cost_impact, 2)
        },
        "anomaly_impact": {
            "anomaly_excess_kwh": round(anomaly_excess_kwh, 2),
            "anomaly_excess_uncertainty_kwh_90ci": round(anomaly_excess_uncertainty_kwh, 2),
            "anomaly_excess_fractional_uncertainty_pct": round(anomaly_excess_uncertainty_pct, 2),
            "measurement_note": "anomaly_excess_kwh is a MEASURED figure for the actual audit window only (not projected).",
            "annualized_anomaly_excess_kwh": round(annualized_anomaly_excess_kwh, 2),
            "annualization_note": (
                f"PROJECTED/ANNUALIZED figure only: anomaly_excess_kwh x (365 / {duration_days:.1f} audit days). "
                f"This is an extrapolation, not a measured annual total and not a verified savings figure."
            )
        },
        "verified_savings": {
            "estimated_savings_kwh": estimated_savings_kwh,
            "status": savings_status,
            "reason": savings_reason
        },
        "anomalies_detected": [
            {
                "timestamp": str(r['timestamp']),
                "actual_kwh": round(float(r['actual_kwh']), 2),
                "expected_kwh": round(float(r['final_expected_kwh']), 2),
                "residual_kwh": round(float(r['residual_kwh']), 2),
                "anomaly_score": round(float(r['anomaly_score']), 2),
                "category": r['anomaly_type']
            }
            for _, r in anomalies_df.iterrows()
        ],
        "recommendation_ranking_methodology": (
            "No-capex actions ranked by (estimated recurring annual cost-savings impact x evidence/confidence "
            "weight [High=1.00, Medium=0.75, Low=0.50]). Capex actions, when present, ranked by ROI "
            "(annual cost savings / implementation cost), which is equivalent to shortest payback first."
        ),
        "recommendations": recommendations
    }

    # --------------------------------------------------------------------------
    # 10. Export All Canonical Deliverables — all three artifacts come from
    #     this same canonical_json / scored run, nothing else.
    # --------------------------------------------------------------------------
    print("Step 7: Writing canonical artifacts...")
    with open(f"{output_dir}/analysis.json", "w") as f:
        json.dump(canonical_json, f, indent=2)
    with open("structured_audit_result.json", "w") as f:
        json.dump(canonical_json, f, indent=2)

    timeseries_contract = scored[[
        'timestamp', 'actual_kwh', 'baseline_kwh', 'weather_expected_kwh',
        'residual_kwh', 'anomaly', 'anomaly_type'
    ]].rename(columns={'baseline_kwh': 'expected_kwh'})

    timeseries_contract.to_csv(f"{output_dir}/anomalies.csv", index=False)
    timeseries_contract.to_csv(f"{output_dir}/timeseries.csv", index=False)
    timeseries_contract.to_csv("final_anomaly_table.csv", index=False)

    canonical_model_dict = {
        "building_id": building_id,
        "profile": profile[['hour_of_week', 'baseline_kwh', 'baseline_mad', 'baseline_se']].to_dict(orient="records"),
        "base_temp": BASE_TEMP,
        "weather_beta_contextual": weather_beta_contextual,
        "weather_beta_applied": weather_beta_applied,
        "weather_used_in_final_model": weather_flag,
        "model_accuracy": canonical_json["model_accuracy"],
        "calibration_start": str(train_start),
        "calibration_end": str(train_end)
    }
    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(canonical_model_dict, f)
    with open("wattsense_model.pkl", "wb") as f:
        pickle.dump(canonical_model_dict, f)

    report_md = f"""# Executive Energy Audit & M&V Investigation Report

**Target Facility:** `{canonical_json['building_id']}` ({canonical_json['timezone']})
**Evaluation Window:** {canonical_json['analysis_period']['start']} to {canonical_json['analysis_period']['end']} ({canonical_json['analysis_period']['duration_hours']} hours / {canonical_json['analysis_period']['duration_days']} days)
**Tariff Model:** {tariff['tariff_source']} ({tariff['currency']} {tariff['base_rate_per_kwh']:.2f}/kWh flat rate)
**Cost Metric Basis:** *{tariff['cost_metric_label']}*

---

## 1. Executive Consumption & Anomaly Summary

| Metric | Measured Value | Methodology / Analytical Note |
| :--- | :--- | :--- |
| **Total Recorded Consumption** | **{canonical_json['energy_metrics']['actual_kwh']:,.2f} kWh** | Metered whole-building facility load |
| **Expected Baseline** | **{canonical_json['energy_metrics']['expected_baseline_kwh']:,.2f} kWh** | Learned 63-day Hour-of-Week median demand |
| **Anomaly Excess (audit window, measured)** | **{canonical_json['anomaly_impact']['anomaly_excess_kwh']:,.2f} kWh** ± {canonical_json['anomaly_impact']['anomaly_excess_uncertainty_kwh_90ci']:.2f} kWh (90% CI) | Unmodeled positive deviation (z > 2.5) — *not projected, not a savings claim* |
| **Annualized Anomaly Excess (projected)** | **{canonical_json['anomaly_impact']['annualized_anomaly_excess_kwh']:,.2f} kWh/yr** | Linear extrapolation of the audit window — *projected estimate only* |
| **Verified Savings** | **Not computed** | {canonical_json['verified_savings']['reason']} |
| **Estimated Cost Impact (audit window)** | **₹{canonical_json['energy_metrics']['estimated_cost_impact']:,.2f}** | *{tariff['cost_metric_label']}* |

---

## 2. Baseline Model Performance & Out-of-Sample Accuracy

* **All-Hours Error:** RMSE = {canonical_json['model_accuracy']['rmse_kwh']:.4f} kWh, CV(RMSE) = {canonical_json['model_accuracy']['cv_rmse_all_hours_pct']:.2f}%.
* **Occupied Class Hours (Mon-Fri 08:00-18:00):** CV(RMSE) = {canonical_json['model_accuracy']['cv_rmse_occupied_hours_pct']:.2f}% (computed fresh from this run).
* **Analytical Assessment:** *{canonical_json['model_accuracy']['evaluation_note']}*
* **Weather Regression Diagnostic:** CDH vs. residuals yields R² = {canonical_json['weather_context']['validation_diagnostics']['in_sample_r2']}. *{canonical_json['weather_context']['validation_diagnostics']['decision']}*

---

## 3. Prioritized Building-Level Action Plan

*{canonical_json['recommendation_ranking_methodology']}*

"""
    for r in canonical_json['recommendations']:
        report_md += f"""### Priority {r['rank']}: {r['action']}
* **Scope:** {r['scope']}
* **Pattern Observed:** {r['pattern_observed']}
* **Projected Annual Impact:** {r['estimated_annual_impact_kwh']:,.1f} kWh/year (₹{r['estimated_annual_cost_savings']:,.2f}/year) — *projected, not verified*
* **Estimated Implementation Cost:** ₹{r['implementation_cost']:.2f}
* **Payback Period:** {r['payback_display']}
* **Confidence Level:** {r['confidence']}
* **Ranking Basis:** {r['ranking_method']} (score = {r['ranking_score']})

"""

    with open(f"{output_dir}/report.md", "w") as f:
        f.write(report_md)
    with open("executive_audit_report.md", "w") as f:
        f.write(report_md)

    print("Master execution completed successfully.")
    print(f"  - Run Directory:  '{output_dir}/'")
    print("  - Canonical JSON: 'structured_audit_result.json'")
    print("  - Time-series:    'final_anomaly_table.csv'")
    print("  - Report:         'executive_audit_report.md'")
    print("=" * 75)

    return canonical_json


if __name__ == "__main__":
    run_audit()
