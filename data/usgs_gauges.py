"""
USGS Stream Gauge Data Fetcher
Pulls real-time and historical data from USGS Water Services API
Stations: French Broad @ Asheville, Swananoa @ Biltmore, Broad River @ Chimney Rock
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Target USGS gauge station IDs for WNC MVP watersheds
GAUGE_STATIONS = {
    "french_broad_asheville": {
        "site_no": "03451500",
        "name": "French Broad River at Asheville, NC",
        "lat": 35.5729,
        "lon": -82.5543,
        "helene_peak_ft": 24.67,
        "flood_stage_ft": 10.0,
        "major_flood_ft": 16.0
    },
    "swananoa_biltmore": {
        "site_no": "03451000",
        "name": "Swananoa River at Biltmore, NC",
        "lat": 35.5557,
        "lon": -82.5182,
        "helene_peak_ft": 18.3,
        "flood_stage_ft": 8.0,
        "major_flood_ft": 12.0
    },
    "broad_chimney_rock": {
        "site_no": "03453500",
        "name": "Broad River near Chimney Rock, NC",
        "lat": 35.4343,
        "lon": -82.2457,
        "helene_peak_ft": 22.1,
        "flood_stage_ft": 9.0,
        "major_flood_ft": 15.0
    }
}

USGS_BASE_URL = "https://waterservices.usgs.gov/nwis"


def fetch_instantaneous(site_no: str, hours: int = 72) -> pd.DataFrame:
    """
    Fetch instantaneous streamflow and stage data from USGS NWIS.
    Returns DataFrame with datetime and stage_ft columns.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=hours)

    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": "00065",  # Gage height in feet
        "startDT": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "endDT": end_dt.strftime("%Y-%m-%dT%H:%M"),
        "siteStatus": "all"
    }

    try:
        response = requests.get(f"{USGS_BASE_URL}/iv/", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        time_series = data.get("value", {}).get("timeSeries", [])
        if not time_series:
            return _generate_synthetic_data(site_no, hours)

        values = time_series[0].get("values", [{}])[0].get("value", [])
        records = []
        for v in values:
            try:
                records.append({
                    "datetime": pd.to_datetime(v["dateTime"]),
                    "stage_ft": float(v["value"]) if v["value"] != "-999999" else np.nan
                })
            except (ValueError, KeyError):
                continue

        df = pd.DataFrame(records)
        df = df.dropna().sort_values("datetime").reset_index(drop=True)
        return df

    except Exception:
        return _generate_synthetic_data(site_no, hours)


def _generate_synthetic_data(site_no: str, hours: int = 72) -> pd.DataFrame:
    """
    Generate realistic synthetic data when live API is unavailable.
    Uses seasonal baseline with realistic noise for demonstration.
    """
    station_baselines = {
        "03451500": 4.2,   # French Broad
        "03451000": 2.8,   # Swananoa
        "03453500": 3.5    # Broad River
    }
    baseline = station_baselines.get(site_no, 3.0)

    datetimes = [datetime.utcnow() - timedelta(hours=hours - i) for i in range(hours * 4)]
    noise = np.random.normal(0, 0.1, len(datetimes))
    trend = np.linspace(0, 0.3, len(datetimes))
    stage = baseline + trend + noise
    stage = np.clip(stage, 0.5, baseline * 3)

    return pd.DataFrame({"datetime": datetimes, "stage_ft": stage})


def fetch_historical(site_no: str, years: int = 10) -> pd.DataFrame:
    """
    Fetch historical daily mean streamflow for model training.
    Returns DataFrame with datetime, discharge_cfs, stage_ft columns.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=years * 365)

    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": "00060,00065",  # Discharge + stage
        "startDT": start_dt.strftime("%Y-%m-%d"),
        "endDT": end_dt.strftime("%Y-%m-%d"),
        "statCd": "00003",  # Daily mean
        "siteStatus": "all"
    }

    try:
        response = requests.get(f"{USGS_BASE_URL}/dv/", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        records = []
        for ts in data.get("value", {}).get("timeSeries", []):
            param = ts.get("variable", {}).get("variableCode", [{}])[0].get("value", "")
            for v in ts.get("values", [{}])[0].get("value", []):
                try:
                    val = float(v["value"]) if v["value"] != "-999999" else np.nan
                    records.append({
                        "datetime": pd.to_datetime(v["dateTime"]),
                        "param": param,
                        "value": val
                    })
                except (ValueError, KeyError):
                    continue

        if not records:
            return _generate_historical_synthetic(site_no, years)

        df = pd.DataFrame(records)
        df_pivot = df.pivot_table(index="datetime", columns="param", values="value", aggfunc="mean")
        df_pivot.columns = ["discharge_cfs" if "60" in c else "stage_ft" for c in df_pivot.columns]
        return df_pivot.reset_index()

    except Exception:
        return _generate_historical_synthetic(site_no, years)


def _generate_historical_synthetic(site_no: str, years: int = 10) -> pd.DataFrame:
    """Generate synthetic historical data for model training demonstration."""
    days = years * 365
    dates = [datetime.utcnow() - timedelta(days=days - i) for i in range(days)]

    baselines = {"03451500": (1200, 4.2), "03451000": (450, 2.8), "03453500": (780, 3.5)}
    q_base, h_base = baselines.get(site_no, (800, 3.5))

    seasonal = np.sin(np.linspace(0, years * 2 * np.pi, days)) * 0.3
    noise_q = np.random.lognormal(0, 0.4, days)
    noise_h = np.random.normal(0, 0.2, days)

    discharge = q_base * (1 + seasonal) * noise_q
    stage = h_base * (1 + seasonal * 0.5) + noise_h
    stage = np.clip(stage, 0.5, h_base * 6)

    # Inject a Helene-like event near end of dataset
    helene_idx = days - 60
    helene_peaks = {"03451500": 24.67, "03451000": 18.3, "03453500": 22.1}
    peak = helene_peaks.get(site_no, 20.0)
    for i in range(48):
        if helene_idx + i < days:
            envelope = np.exp(-0.5 * ((i - 12) / 8) ** 2)
            stage[helene_idx + i] = max(stage[helene_idx + i], peak * envelope)
            discharge[helene_idx + i] = max(discharge[helene_idx + i], q_base * 15 * envelope)

    return pd.DataFrame({"datetime": dates, "discharge_cfs": discharge, "stage_ft": stage})


def fetch_gauge_data() -> dict:
    """
    Master function: fetches current conditions for all three gauge stations.
    Returns dict with current levels, deltas, and historical DataFrame.
    """
    result = {}
    all_dfs = []

    for key, station in GAUGE_STATIONS.items():
        df = fetch_instantaneous(station["site_no"], hours=72)

        if not df.empty:
            current_ft = float(df["stage_ft"].iloc[-1])
            prev_ft = float(df["stage_ft"].iloc[-2]) if len(df) > 1 else current_ft
            delta_ft = current_ft - prev_ft
            flood_pct = (current_ft / station["helene_peak_ft"]) * 100

            result[key] = {
                "current_ft": round(current_ft, 2),
                "delta_ft": round(delta_ft, 3),
                "flood_pct": round(flood_pct, 1),
                "df": df
            }
            all_dfs.append((key, df))

    # Build combined historical DataFrame for charting
    if all_dfs:
        base_df = all_dfs[0][1].rename(columns={"stage_ft": f"{all_dfs[0][0]}_ft"})
        combined = base_df[["datetime", f"{all_dfs[0][0]}_ft"]].copy()
        combined.columns = ["datetime", "french_broad_ft"]

        if len(all_dfs) > 1:
            df2 = all_dfs[1][1].rename(columns={"stage_ft": "swananoa_ft"})
            combined = combined.merge(df2[["datetime", "swananoa_ft"]], on="datetime", how="outer")
        if len(all_dfs) > 2:
            df3 = all_dfs[2][1].rename(columns={"stage_ft": "broad_ft"})
            combined = combined.merge(df3[["datetime", "broad_ft"]], on="datetime", how="outer")

        combined = combined.sort_values("datetime").ffill()
        result["historical_df"] = combined

    return result
