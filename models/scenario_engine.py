"""
FloodFlow MVP - Scenario Testing Engine
Savage Ops | Adapt. Advance. Achieve.

Three scenario types:
1. Historical Replay - Run known past events through the model and compare predictions to documented outcomes
2. Synthetic Storm - Dial in hypothetical storm parameters and generate full prediction report
3. Threshold Testing - Test model outputs at defined flow percentages of Helene peak
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models.debris_accumulation import DebrisAccumulationPredictor, HELENE_DEPOSIT_LOCATIONS
from models.debris_flow_classifier import DebrisFlowClassifier
from models.lstm_predictor import LSTMPredictor


# ============================================================
# HISTORICAL EVENT LIBRARY
# Documented events for replay validation
# ============================================================

HISTORICAL_EVENTS = {
    "helene_2024": {
        "name": "Hurricane Helene",
        "date": "September 27, 2024",
        "description": "Category 4 landfall Florida, catastrophic inland flooding WNC. Geological-scale debris flows.",
        "rainfall_24hr_inches": {"swananoa": 12.4, "french_broad": 14.1, "broad_river": 13.3},
        "rainfall_total_inches": {"swananoa": 15.8, "french_broad": 17.2, "broad_river": 16.5},
        "observed_gauge_peaks": {
            "french_broad_asheville": {"peak_ft": 24.67, "hours_to_peak": 9.8},
            "swananoa_biltmore": {"peak_ft": 18.3, "hours_to_peak": 5.2},
            "broad_chimney_rock": {"peak_ft": 22.1, "hours_to_peak": 7.4}
        },
        "observed_flood_pcts": {"french_broad_asheville": 100, "swananoa_biltmore": 100, "broad_chimney_rock": 100},
        "debris_flows_confirmed": 10,
        "alert_tier_actual": "IMMINENT",
        "economic_damage_est": "$50B+",
        "fatalities_nc": 104
    },
    "fred_2021": {
        "name": "Tropical Storm Fred",
        "date": "August 17, 2021",
        "description": "Tropical Storm Fred caused significant flooding in Haywood and Transylvania Counties.",
        "rainfall_24hr_inches": {"swananoa": 4.2, "french_broad": 5.8, "broad_river": 4.9},
        "rainfall_total_inches": {"swananoa": 6.1, "french_broad": 8.3, "broad_river": 7.2},
        "observed_gauge_peaks": {
            "french_broad_asheville": {"peak_ft": 14.2, "hours_to_peak": 11.0},
            "swananoa_biltmore": {"peak_ft": 9.8, "hours_to_peak": 6.5},
            "broad_chimney_rock": {"peak_ft": 11.3, "hours_to_peak": 8.2}
        },
        "observed_flood_pcts": {"french_broad_asheville": 58, "swananoa_biltmore": 54, "broad_chimney_rock": 51},
        "debris_flows_confirmed": 3,
        "alert_tier_actual": "WARNING",
        "economic_damage_est": "$180M",
        "fatalities_nc": 4
    },
    "michael_2018": {
        "name": "Hurricane Michael Remnants",
        "date": "October 11, 2018",
        "description": "Remnants of Hurricane Michael produced heavy rainfall across WNC.",
        "rainfall_24hr_inches": {"swananoa": 3.1, "french_broad": 3.8, "broad_river": 3.4},
        "rainfall_total_inches": {"swananoa": 4.5, "french_broad": 5.2, "broad_river": 4.8},
        "observed_gauge_peaks": {
            "french_broad_asheville": {"peak_ft": 10.8, "hours_to_peak": 12.0},
            "swananoa_biltmore": {"peak_ft": 7.4, "hours_to_peak": 7.0},
            "broad_chimney_rock": {"peak_ft": 8.9, "hours_to_peak": 9.0}
        },
        "observed_flood_pcts": {"french_broad_asheville": 44, "swananoa_biltmore": 40, "broad_chimney_rock": 40},
        "debris_flows_confirmed": 1,
        "alert_tier_actual": "WATCH",
        "economic_damage_est": "$42M",
        "fatalities_nc": 0
    }
}

# ============================================================
# SYNTHETIC STORM TRACK TEMPLATES
# ============================================================

STORM_TEMPLATES = {
    "gulf_landfall_track": {
        "name": "Gulf Landfall - Appalachian Track",
        "description": "Hurricane makes landfall Gulf Coast, tracks northeast through Georgia into WNC. Helene-type scenario.",
        "track_hours_to_wnc": 36,
        "rainfall_multiplier": 1.0,
        "confidence": "HIGH"
    },
    "atlantic_landfall_track": {
        "name": "Atlantic Landfall - Inland Track",
        "description": "Hurricane makes landfall NC/SC coast, tracks inland toward WNC. Faster decay, lower totals.",
        "track_hours_to_wnc": 18,
        "rainfall_multiplier": 0.65,
        "confidence": "MODERATE"
    },
    "cutoff_low": {
        "name": "Cutoff Low / Stalled System",
        "description": "Non-tropical slow-moving low pressure system stalls over WNC. Extended multi-day rainfall.",
        "track_hours_to_wnc": 12,
        "rainfall_multiplier": 0.75,
        "confidence": "MODERATE"
    },
    "orographic_event": {
        "name": "Orographic Enhancement Event",
        "description": "Moderate synoptic rainfall enhanced by terrain lift over Blue Ridge. Localized intense rainfall.",
        "track_hours_to_wnc": 6,
        "rainfall_multiplier": 0.55,
        "confidence": "HIGH"
    }
}


class ScenarioEngine:
    """
    Runs the full FloodFlow prediction pipeline against
    user-defined or historical scenario inputs.
    """

    def __init__(self):
        self.lstm = LSTMPredictor()
        self.classifier = DebrisFlowClassifier()
        self.accumulation = DebrisAccumulationPredictor()

    # ============================================================
    # 1. HISTORICAL REPLAY
    # ============================================================

    def run_historical_replay(self, event_key: str) -> dict:
        """
        Run model against documented historical event.
        Compares model predictions to observed outcomes.
        Returns accuracy metrics and full prediction report.
        """
        event = HISTORICAL_EVENTS.get(event_key)
        if not event:
            return {"error": f"Event {event_key} not found"}

        # Build synthetic gauge data from historical observations
        gauge_data = self._build_gauge_data_from_event(event)
        predictions = self.lstm.predict(gauge_data, 24)

        # Run debris flow classification
        risk_df = self.classifier.classify(gauge_data, predictions)

        # Run debris accumulation
        acc_df = self.accumulation.predict_accumulation_zones(gauge_data, predictions)

        # Run alert evaluation
        from utils.alert_system import evaluate_alert_tier
        predicted_tier = evaluate_alert_tier(gauge_data, "Standard")

        # Compute accuracy vs observed
        accuracy = self._compute_accuracy(predictions, event["observed_gauge_peaks"])

        # Build validation comparison table
        validation_rows = []
        station_map = {
            "french_broad": "french_broad_asheville",
            "swananoa": "swananoa_biltmore",
            "broad": "broad_chimney_rock"
        }
        station_labels = {
            "french_broad_asheville": "French Broad @ Asheville",
            "swananoa_biltmore": "Swananoa @ Biltmore",
            "broad_chimney_rock": "Broad River @ Chimney Rock"
        }

        for pred_key, gauge_key in station_map.items():
            observed = event["observed_gauge_peaks"].get(gauge_key, {})
            predicted = predictions.get(pred_key, {})

            obs_peak = observed.get("peak_ft", 0)
            pred_peak = predicted.get("peak_ft", 0)
            obs_time = observed.get("hours_to_peak", 0)
            pred_time = predicted.get("hours_to_peak", 0)

            peak_error = abs(pred_peak - obs_peak)
            peak_error_pct = (peak_error / obs_peak * 100) if obs_peak > 0 else 0
            time_error = abs(pred_time - obs_time)

            validation_rows.append({
                "Station": station_labels.get(gauge_key, gauge_key),
                "Observed Peak (ft)": obs_peak,
                "Predicted Peak (ft)": round(pred_peak, 2),
                "Peak Error (ft)": round(peak_error, 2),
                "Peak Error %": f"{peak_error_pct:.1f}%",
                "Observed Time to Peak (hrs)": obs_time,
                "Predicted Time to Peak (hrs)": round(pred_time, 1),
                "Timing Error (hrs)": round(time_error, 1),
                "Within Target": "✅" if peak_error_pct <= 20 and time_error <= 3 else "❌"
            })

        return {
            "event": event,
            "gauge_data": gauge_data,
            "predictions": predictions,
            "risk_df": risk_df,
            "acc_df": acc_df,
            "predicted_alert_tier": predicted_tier,
            "actual_alert_tier": event["alert_tier_actual"],
            "tier_match": predicted_tier == event["alert_tier_actual"],
            "validation_df": pd.DataFrame(validation_rows),
            "accuracy": accuracy,
            "run_timestamp": datetime.utcnow().isoformat()
        }

    def _build_gauge_data_from_event(self, event: dict) -> dict:
        """Build synthetic gauge_data dict from historical event parameters."""
        gauge_data = {}
        station_map = {
            "french_broad_asheville": {"helene_peak": 24.67},
            "swananoa_biltmore": {"helene_peak": 18.3},
            "broad_chimney_rock": {"helene_peak": 22.1}
        }

        for gauge_key, params in station_map.items():
            obs = event["observed_gauge_peaks"].get(gauge_key, {})
            peak_ft = obs.get("peak_ft", params["helene_peak"] * 0.3)
            flood_pct = (peak_ft / params["helene_peak"]) * 100

            gauge_data[gauge_key] = {
                "current_ft": round(peak_ft * 0.85, 2),
                "delta_ft": round(peak_ft * 0.05, 3),
                "flood_pct": round(flood_pct, 1)
            }

        import pandas as pd
        hours = 72
        datetimes = [datetime.utcnow() - timedelta(hours=hours - i) for i in range(hours * 4)]
        gauge_data["historical_df"] = pd.DataFrame({
            "datetime": datetimes,
            "french_broad_ft": np.random.normal(
                event["observed_gauge_peaks"]["french_broad_asheville"]["peak_ft"] * 0.7,
                1.5, len(datetimes)
            ),
            "swananoa_ft": np.random.normal(
                event["observed_gauge_peaks"]["swananoa_biltmore"]["peak_ft"] * 0.7,
                1.0, len(datetimes)
            ),
            "broad_ft": np.random.normal(
                event["observed_gauge_peaks"]["broad_chimney_rock"]["peak_ft"] * 0.7,
                1.2, len(datetimes)
            )
        })

        return gauge_data

    def _compute_accuracy(self, predictions: dict, observed: dict) -> dict:
        """Compute accuracy metrics across all stations."""
        station_map = {
            "french_broad": "french_broad_asheville",
            "swananoa": "swananoa_biltmore",
            "broad": "broad_chimney_rock"
        }
        peak_errors = []
        time_errors = []

        for pred_key, gauge_key in station_map.items():
            obs = observed.get(gauge_key, {})
            pred = predictions.get(pred_key, {})
            if obs.get("peak_ft") and pred.get("peak_ft"):
                peak_errors.append(
                    abs(pred["peak_ft"] - obs["peak_ft"]) / obs["peak_ft"] * 100
                )
            if obs.get("hours_to_peak") and pred.get("hours_to_peak"):
                time_errors.append(
                    abs(pred["hours_to_peak"] - obs["hours_to_peak"])
                )

        return {
            "mean_peak_error_pct": round(np.mean(peak_errors), 1) if peak_errors else 0,
            "mean_timing_error_hrs": round(np.mean(time_errors), 1) if time_errors else 0,
            "peak_within_target": all(e <= 20 for e in peak_errors),
            "timing_within_target": all(e <= 3 for e in time_errors),
            "overall_pass": (
                all(e <= 20 for e in peak_errors) and
                all(e <= 3 for e in time_errors)
            )
        }

    # ============================================================
    # 2. SYNTHETIC STORM SCENARIO
    # ============================================================

    def run_synthetic_scenario(
        self,
        storm_name: str,
        rainfall_24hr_swananoa: float,
        rainfall_24hr_french_broad: float,
        rainfall_24hr_broad: float,
        hours_to_arrival: int,
        storm_template_key: str = "gulf_landfall_track",
        storm_intensity: str = "Major Hurricane"
    ) -> dict:
        """
        Run full prediction pipeline against user-defined storm scenario.
        Returns complete staging and deployment recommendations.
        """
        template = STORM_TEMPLATES.get(storm_template_key, STORM_TEMPLATES["gulf_landfall_track"])

        # Scale rainfall by template multiplier
        mult = template["rainfall_multiplier"]
        rainfall = {
            "swananoa": rainfall_24hr_swananoa * mult,
            "french_broad": rainfall_24hr_french_broad * mult,
            "broad_river": rainfall_24hr_broad * mult
        }

        # Build synthetic gauge data based on rainfall input
        gauge_data = self._rainfall_to_gauge_data(rainfall)
        predictions = self.lstm.predict(gauge_data, 72)
        risk_df = self.classifier.classify(gauge_data, predictions)
        acc_df = self.accumulation.predict_accumulation_zones(gauge_data, predictions)

        from utils.alert_system import evaluate_alert_tier
        alert_tier = evaluate_alert_tier(gauge_data, "Standard")

        # Build deployment timeline
        deploy_timeline = self._build_deployment_timeline(
            predictions, hours_to_arrival, alert_tier
        )

        # Flood potential categories
        def flood_potential(inches):
            if inches >= 12:
                return "CATASTROPHIC"
            elif inches >= 6:
                return "MAJOR"
            elif inches >= 3:
                return "MODERATE"
            elif inches >= 1.5:
                return "MINOR"
            return "MINIMAL"

        return {
            "storm_name": storm_name,
            "storm_intensity": storm_intensity,
            "template": template,
            "rainfall_inputs": rainfall,
            "flood_potential": {k: flood_potential(v) for k, v in rainfall.items()},
            "gauge_data": gauge_data,
            "predictions": predictions,
            "risk_df": risk_df,
            "acc_df": acc_df,
            "alert_tier": alert_tier,
            "deploy_timeline": deploy_timeline,
            "hours_to_arrival": hours_to_arrival,
            "run_timestamp": datetime.utcnow().isoformat()
        }

    def _rainfall_to_gauge_data(self, rainfall: dict) -> dict:
        """Convert rainfall forecast to estimated gauge conditions."""
        helene_rainfall = {"swananoa": 12.4, "french_broad": 14.1, "broad_river": 13.3}
        helene_peaks = {
            "french_broad_asheville": 24.67,
            "swananoa_biltmore": 18.3,
            "broad_chimney_rock": 22.1
        }

        river_to_gauge = {
            "swananoa": "swananoa_biltmore",
            "french_broad": "french_broad_asheville",
            "broad_river": "broad_chimney_rock"
        }

        gauge_data = {}
        for river_key, gauge_key in river_to_gauge.items():
            rain = rainfall.get(river_key, 0)
            helene_rain = helene_rainfall.get(river_key, 12)
            rain_ratio = min(rain / helene_rain, 1.0)

            helene_peak = helene_peaks[gauge_key]
            est_peak = helene_peak * rain_ratio * 0.85
            flood_pct = (est_peak / helene_peak) * 100

            gauge_data[gauge_key] = {
                "current_ft": round(est_peak * 0.6, 2),
                "delta_ft": round(est_peak * 0.04, 3),
                "flood_pct": round(flood_pct, 1)
            }

        import pandas as pd
        hours = 72
        datetimes = [datetime.utcnow() - timedelta(hours=hours - i) for i in range(hours * 4)]
        gauge_data["historical_df"] = pd.DataFrame({
            "datetime": datetimes,
            "french_broad_ft": np.random.normal(gauge_data["french_broad_asheville"]["current_ft"] * 0.7, 1.0, len(datetimes)),
            "swananoa_ft": np.random.normal(gauge_data["swananoa_biltmore"]["current_ft"] * 0.7, 0.8, len(datetimes)),
            "broad_ft": np.random.normal(gauge_data["broad_chimney_rock"]["current_ft"] * 0.7, 0.9, len(datetimes))
        })

        return gauge_data

    def _build_deployment_timeline(
        self, predictions: dict, hours_to_arrival: int, alert_tier: str
    ) -> list:
        """Build chronological deployment action timeline."""
        now = datetime.utcnow()
        timeline = []

        timeline.append({
            "time": now.strftime("%m/%d %H:%M UTC"),
            "hours_from_now": 0,
            "action": "SCENARIO INITIATED - Full model run complete",
            "priority": "INFO"
        })

        if hours_to_arrival > 36:
            timeline.append({
                "time": (now + timedelta(hours=2)).strftime("%m/%d %H:%M UTC"),
                "hours_from_now": 2,
                "action": "Notify county emergency management: Henderson, Buncombe, Rutherford",
                "priority": "HIGH"
            })

        timeline.append({
            "time": (now + timedelta(hours=max(hours_to_arrival - 24, 1))).strftime("%m/%d %H:%M UTC"),
            "hours_from_now": max(hours_to_arrival - 24, 1),
            "action": "Pre-position rescue assets at SZ-07 Henderson and SZ-03 Biltmore",
            "priority": "HIGH"
        })

        timeline.append({
            "time": (now + timedelta(hours=max(hours_to_arrival - 12, 2))).strftime("%m/%d %H:%M UTC"),
            "hours_from_now": max(hours_to_arrival - 12, 2),
            "action": "Deploy all staging zones. Confirm road access Chimney Rock corridor NC-9 / US-64A",
            "priority": "CRITICAL"
        })

        timeline.append({
            "time": (now + timedelta(hours=max(hours_to_arrival - 6, 3))).strftime("%m/%d %H:%M UTC"),
            "hours_from_now": max(hours_to_arrival - 6, 3),
            "action": "Evacuate civilians Swananoa Valley, Chimney Rock, Bat Cave. Block hazard road segments.",
            "priority": "CRITICAL"
        })

        timeline.append({
            "time": (now + timedelta(hours=hours_to_arrival)).strftime("%m/%d %H:%M UTC"),
            "hours_from_now": hours_to_arrival,
            "action": "Estimated storm arrival / rainfall onset in WNC watersheds",
            "priority": "INFO"
        })

        # Add gauge peak times
        for key, label in [("french_broad", "French Broad"), ("swananoa", "Swananoa"), ("broad", "Broad River")]:
            pred = predictions.get(key, {})
            peak_hrs = pred.get("hours_to_peak", 12)
            total_hrs = hours_to_arrival + peak_hrs
            timeline.append({
                "time": (now + timedelta(hours=total_hrs)).strftime("%m/%d %H:%M UTC"),
                "hours_from_now": total_hrs,
                "action": f"{label} gauge predicted peak: {pred.get('peak_ft', 0):.1f} ft",
                "priority": "WARNING"
            })

        timeline.append({
            "time": (now + timedelta(hours=hours_to_arrival + 24)).strftime("%m/%d %H:%M UTC"),
            "hours_from_now": hours_to_arrival + 24,
            "action": "Begin debris accumulation zone search operations. Priority: Chimney Rock gorge exit, Lake Lure inlet, Swananoa Valley.",
            "priority": "HIGH"
        })

        return sorted(timeline, key=lambda x: x["hours_from_now"])

    # ============================================================
    # 3. THRESHOLD TESTING
    # ============================================================

    def run_threshold_test(self, flow_pct: float) -> dict:
        """
        Run model at a specific percentage of Helene peak flow.
        Tests model outputs across the full prediction pipeline
        at 25, 50, 75, or 100 percent of Helene conditions.
        """
        helene_peaks = {
            "french_broad_asheville": {"peak_ft": 24.67, "helene_peak": 24.67},
            "swananoa_biltmore": {"peak_ft": 18.3, "helene_peak": 18.3},
            "broad_chimney_rock": {"peak_ft": 22.1, "helene_peak": 22.1}
        }

        gauge_data = {}
        for gauge_key, params in helene_peaks.items():
            current_ft = params["peak_ft"] * (flow_pct / 100) * 0.85
            gauge_data[gauge_key] = {
                "current_ft": round(current_ft, 2),
                "delta_ft": round(current_ft * 0.04, 3),
                "flood_pct": round(flow_pct, 1)
            }

        import pandas as pd
        hours = 72
        datetimes = [datetime.utcnow() - timedelta(hours=hours - i) for i in range(hours * 4)]
        gauge_data["historical_df"] = pd.DataFrame({
            "datetime": datetimes,
            "french_broad_ft": np.random.normal(gauge_data["french_broad_asheville"]["current_ft"] * 0.8, 0.8, len(datetimes)),
            "swananoa_ft": np.random.normal(gauge_data["swananoa_biltmore"]["current_ft"] * 0.8, 0.6, len(datetimes)),
            "broad_ft": np.random.normal(gauge_data["broad_chimney_rock"]["current_ft"] * 0.8, 0.7, len(datetimes))
        })

        predictions = self.lstm.predict(gauge_data, 24)
        risk_df = self.classifier.classify(gauge_data, predictions)
        acc_df = self.accumulation.predict_accumulation_zones(
            gauge_data, predictions, flood_pct_override=flow_pct
        )

        from utils.alert_system import evaluate_alert_tier
        alert_tier = evaluate_alert_tier(gauge_data, "Standard")

        primary_sites = len(acc_df[acc_df["Priority Class"] == "PRIMARY"]) if not acc_df.empty else 0
        high_risk_reaches = len(risk_df[risk_df["Risk Level"] == "HIGH"]) if not risk_df.empty else 0

        return {
            "flow_pct": flow_pct,
            "gauge_data": gauge_data,
            "predictions": predictions,
            "risk_df": risk_df,
            "acc_df": acc_df,
            "alert_tier": alert_tier,
            "summary": {
                "flow_scenario": f"{flow_pct:.0f}% of Helene Peak",
                "alert_tier": alert_tier,
                "high_risk_reaches": high_risk_reaches,
                "primary_accumulation_sites": primary_sites,
                "french_broad_predicted_ft": predictions.get("french_broad", {}).get("peak_ft", 0),
                "swananoa_predicted_ft": predictions.get("swananoa", {}).get("peak_ft", 0),
                "broad_predicted_ft": predictions.get("broad", {}).get("peak_ft", 0),
            },
            "run_timestamp": datetime.utcnow().isoformat()
        }

    def run_all_thresholds(self) -> pd.DataFrame:
        """Run threshold tests at 25, 50, 75, and 100 percent and return comparison table."""
        rows = []
        for pct in [10, 25, 50, 75, 100]:
            result = self.run_threshold_test(float(pct))
            s = result["summary"]
            rows.append({
                "Flow Scenario": s["flow_scenario"],
                "Alert Tier": s["alert_tier"],
                "French Broad Peak (ft)": s["french_broad_predicted_ft"],
                "Swananoa Peak (ft)": s["swananoa_predicted_ft"],
                "Broad River Peak (ft)": s["broad_predicted_ft"],
                "High Risk Reaches": s["high_risk_reaches"],
                "Primary Accumulation Sites": s["primary_accumulation_sites"],
            })
        return pd.DataFrame(rows)
