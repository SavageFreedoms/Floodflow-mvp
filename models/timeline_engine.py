"""
FloodFlow MVP - Timeline Animation Engine
Savage Ops | Adapt. Advance. Achieve.

Generates hour-by-hour animated playback of historical flood events
using USGS gauge data and NOAA streamflow records.

Also contains enhanced debris accumulation logic separating
RESCUE phase from RECOVERY phase search priorities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests


# ============================================================
# HISTORICAL EVENT TIMELINE DATA
# Reconstructed from USGS NWIS records and USACE HWM dataset
# ============================================================

HELENE_TIMELINE_DATA = {
    "event": "Hurricane Helene",
    "start": datetime(2024, 9, 25, 0, 0),
    "peak": datetime(2024, 9, 27, 6, 0),
    "end": datetime(2024, 9, 29, 0, 0),
    "total_hours": 96,
    "description": "Category 4 landfall Florida Gulf Coast September 26. Catastrophic inland flooding WNC September 27.",
    "key_timestamps": [
        {"hour": 0, "label": "Storm forms Gulf of Mexico", "severity": "WATCH"},
        {"hour": 24, "label": "Landfall Florida Gulf Coast - Category 4", "severity": "WARNING"},
        {"hour": 36, "label": "Tropical storm conditions reach WNC. Rainfall onset.", "severity": "WARNING"},
        {"hour": 48, "label": "Extreme rainfall begins. French Broad rising rapidly.", "severity": "IMMINENT"},
        {"hour": 54, "label": "PEAK: French Broad 24.67 ft. Swananoa 18.3 ft. Debris flows initiating.", "severity": "IMMINENT"},
        {"hour": 58, "label": "Chimney Rock gorge debris flow. Rocky Broad destroys NC-9.", "severity": "IMMINENT"},
        {"hour": 60, "label": "Lake Lure inlet debris dam forming. Communications lost across WNC.", "severity": "IMMINENT"},
        {"hour": 72, "label": "Flows receding. Rescue operations begin. Access severely limited.", "severity": "WARNING"},
        {"hour": 84, "label": "Recovery phase begins. Body recovery operations initiated.", "severity": "WATCH"},
        {"hour": 96, "label": "Rivers returning toward baseline. Debris fields stable.", "severity": "NORMAL"},
    ]
}

FRED_TIMELINE_DATA = {
    "event": "Tropical Storm Fred",
    "start": datetime(2021, 8, 15, 0, 0),
    "peak": datetime(2021, 8, 17, 12, 0),
    "end": datetime(2021, 8, 19, 0, 0),
    "total_hours": 72,
    "description": "Tropical Storm Fred caused significant flooding in Haywood and Transylvania Counties.",
    "key_timestamps": [
        {"hour": 0, "label": "Fred enters Gulf of Mexico", "severity": "WATCH"},
        {"hour": 24, "label": "Landfall Florida Panhandle", "severity": "WARNING"},
        {"hour": 36, "label": "Rainfall onset WNC mountains", "severity": "WARNING"},
        {"hour": 48, "label": "PEAK: Significant flooding Haywood County. Pigeon River cresting.", "severity": "IMMINENT"},
        {"hour": 60, "label": "Flows receding. Rescue operations active.", "severity": "WARNING"},
        {"hour": 72, "label": "Rivers normalizing. Recovery phase.", "severity": "WATCH"},
    ]
}

EVENT_TIMELINES = {
    "helene_2024": HELENE_TIMELINE_DATA,
    "fred_2021": FRED_TIMELINE_DATA
}


def generate_event_timeline_df(event_key: str) -> pd.DataFrame:
    """
    Generate hour-by-hour gauge reading DataFrame for a historical event.
    Uses reconstructed flow curves calibrated to documented peak values.
    Returns DataFrame suitable for animated chart playback.
    """
    timeline = EVENT_TIMELINES.get(event_key, HELENE_TIMELINE_DATA)
    total_hours = timeline["total_hours"]

    # Station peak values for this event
    if event_key == "helene_2024":
        peaks = {"french_broad": 24.67, "swananoa": 18.3, "broad": 22.1}
        baselines = {"french_broad": 4.2, "swananoa": 2.8, "broad": 3.5}
        peak_hour = 54  # Hour when French Broad peaked
        rise_start = 36  # Hour when significant rise began
    else:
        peaks = {"french_broad": 14.2, "swananoa": 9.8, "broad": 11.3}
        baselines = {"french_broad": 4.2, "swananoa": 2.8, "broad": 3.5}
        peak_hour = 48
        rise_start = 36

    records = []
    for hour in range(total_hours + 1):
        dt = timeline["start"] + timedelta(hours=hour)

        # Generate realistic hydrograph shape for each station
        # Rising limb faster than falling limb (realistic)
        fb = _hydrograph(hour, baselines["french_broad"], peaks["french_broad"],
                        rise_start, peak_hour, lag_hrs=0)
        sw = _hydrograph(hour, baselines["swananoa"], peaks["swananoa"],
                        rise_start, peak_hour - 5, lag_hrs=-4)  # Swananoa peaks earlier
        br = _hydrograph(hour, baselines["broad"], peaks["broad"],
                        rise_start, peak_hour - 2, lag_hrs=-2)  # Broad intermediate

        # Add noise
        fb += np.random.normal(0, 0.15)
        sw += np.random.normal(0, 0.10)
        br += np.random.normal(0, 0.12)

        # Pct of Helene peak
        fb_pct = (fb / 24.67) * 100
        sw_pct = (sw / 18.3) * 100
        br_pct = (br / 22.1) * 100

        # Alert tier at this hour
        max_pct = max(fb_pct, sw_pct, br_pct)
        if max_pct >= 75:
            tier = "IMMINENT"
        elif max_pct >= 50:
            tier = "WARNING"
        elif max_pct >= 25:
            tier = "WATCH"
        else:
            tier = "NORMAL"

        # Phase
        if hour < rise_start:
            phase = "PRE-EVENT"
        elif hour <= peak_hour:
            phase = "RISING LIMB"
        elif hour <= peak_hour + 18:
            phase = "FALLING LIMB"
        elif hour <= peak_hour + 30:
            phase = "RESCUE PHASE"
        else:
            phase = "RECOVERY PHASE"

        records.append({
            "hour": hour,
            "datetime": dt,
            "datetime_label": dt.strftime("%m/%d %H:%M"),
            "french_broad_ft": round(max(fb, baselines["french_broad"]), 2),
            "swananoa_ft": round(max(sw, baselines["swananoa"]), 2),
            "broad_ft": round(max(br, baselines["broad"]), 2),
            "french_broad_pct_helene": round(fb_pct, 1),
            "alert_tier": tier,
            "phase": phase
        })

    return pd.DataFrame(records)


def _hydrograph(
    hour: int,
    baseline: float,
    peak: float,
    rise_start: int,
    peak_hour: int,
    lag_hrs: int = 0
) -> float:
    """
    Generate realistic hydrograph value at a given hour.
    Rising limb uses exponential rise, falling limb uses slower exponential decay.
    """
    effective_hour = hour + lag_hrs
    effective_peak = peak_hour + lag_hrs

    if effective_hour < rise_start:
        return baseline + np.random.uniform(0, 0.1)

    rise_duration = effective_peak - rise_start
    if rise_duration <= 0:
        rise_duration = 12

    if effective_hour <= effective_peak:
        # Rising limb - exponential
        progress = (effective_hour - rise_start) / rise_duration
        value = baseline + (peak - baseline) * (progress ** 1.8)
    else:
        # Falling limb - slower exponential decay
        fall_progress = (effective_hour - effective_peak) / max(rise_duration * 2.5, 20)
        value = baseline + (peak - baseline) * np.exp(-fall_progress * 2.2)

    return max(value, baseline)


def get_key_timestamps_for_event(event_key: str) -> list:
    """Return key event timestamps for timeline annotation."""
    timeline = EVENT_TIMELINES.get(event_key, HELENE_TIMELINE_DATA)
    return timeline.get("key_timestamps", [])


# ============================================================
# ENHANCED DEBRIS ACCUMULATION - RESCUE vs RECOVERY PHASE
# ============================================================

RESCUE_PHASE_HOURS = 72      # First 72 hours: active rescue
RECOVERY_PHASE_START = 72    # After 72 hours: body recovery


DEBRIS_RECOVERY_INTELLIGENCE = {
    "DEP-BR-002": {
        "name": "Lake Lure Delta / Inlet",
        "recovery_priority": 1,
        "recovery_notes": (
            "Lake Lure inlet is the terminal deposition zone for all Broad River debris. "
            "Complete velocity loss at lake entrance means ALL transported material - "
            "including victims - accumulates here. Primary body recovery search zone. "
            "Requires boat assets and underwater search capability."
        ),
        "search_method": "Boat-based sonar + dive team",
        "access": "Lake Lure Marina / Town Hall boat launch",
        "depth_concern": True,
        "underwater_search": True
    },
    "DEP-BR-001": {
        "name": "Chimney Rock Gorge Exit",
        "recovery_priority": 2,
        "recovery_notes": (
            "Gorge exit produces massive debris dam. Highest single-site debris volume "
            "in the Broad River corridor. Rocky Broad transported structural debris, "
            "vehicles, and trees from Chimney Rock community. Search entire debris field "
            "systematically from gorge exit downstream 0.9 miles."
        ),
        "search_method": "Ground search team + cadaver K9",
        "access": "US-64A from Lake Lure side - verify road status before entry",
        "depth_concern": True,
        "underwater_search": False
    },
    "DEP-SW-001": {
        "name": "Craigtown Complex - East Prong",
        "recovery_priority": 3,
        "recovery_notes": (
            "1.75 miles of continuous debris scour at 4-foot average depth. "
            "Largest single debris flow track in Swananoa watershed. "
            "Post-Helene LiDAR confirms massive sediment redistribution. "
            "Search entire scour track systematically - debris may cover victims."
        ),
        "search_method": "Ground search team + cadaver K9 + drone overwatch",
        "access": "Old US-70 to Swannanoa - verify bridge status",
        "depth_concern": True,
        "underwater_search": False
    },
    "DEP-BR-003": {
        "name": "Bat Cave Road / NC-9 Bridge",
        "recovery_priority": 4,
        "recovery_notes": (
            "Bridge debris dam creates upstream impoundment zone. "
            "Material trapped against bridge structure. Victims swept into "
            "bridge blockage may be submerged in debris field immediately upstream. "
            "Structural collapse risk - do not enter without engineer assessment."
        ),
        "search_method": "Structural engineer assessment first. Then dive team.",
        "access": "Approach from US-64 Henderson County side only",
        "depth_concern": True,
        "underwater_search": True
    },
    "DEP-FB-002": {
        "name": "French Broad / Swananoa Confluence",
        "recovery_priority": 5,
        "recovery_notes": (
            "Major confluence creates large low-velocity deposition zone. "
            "Material from both the French Broad and Swananoa systems converges here. "
            "Wide search area required. Biltmore Estate downstream floodplain "
            "should also be searched systematically."
        ),
        "search_method": "Ground search + boat-based search of confluence pool",
        "access": "US-25 Biltmore Village - coordinate with Biltmore Estate",
        "depth_concern": False,
        "underwater_search": False
    },
    "DEP-SW-003": {
        "name": "Swannanoa Valley Floodplain",
        "recovery_priority": 6,
        "recovery_notes": (
            "2.1-mile overbank deposit zone through Swannanoa community. "
            "Wide, shallow deposition. Systematic grid search required. "
            "Helene destroyed multiple residential structures in this reach - "
            "building debris mixed with natural debris throughout."
        ),
        "search_method": "Systematic grid search team + cadaver K9",
        "access": "Old US-70 / Business US-70 Swannanoa",
        "depth_concern": False,
        "underwater_search": False
    },
    "DEP-BR-004": {
        "name": "Gerton / US-74 Corridor",
        "recovery_priority": 7,
        "recovery_notes": (
            "Alluvial fan deposit below gorge. Material from upper Broad River "
            "corridor accumulates at valley widening. Secondary recovery zone "
            "after primary sites cleared."
        ),
        "search_method": "Ground search team",
        "access": "US-74A from Bat Cave east",
        "depth_concern": False,
        "underwater_search": False
    },
}


def get_recovery_intelligence_df() -> pd.DataFrame:
    """
    Build prioritized body recovery intelligence table.
    Separate from rescue staging - this is the post-rescue search framework.
    """
    rows = []
    for deposit_id, intel in DEBRIS_RECOVERY_INTELLIGENCE.items():
        rows.append({
            "Recovery Priority": f"#{intel['recovery_priority']}",
            "Location": intel["name"],
            "Search Method": intel["search_method"],
            "Underwater Search Required": "YES" if intel["underwater_search"] else "NO",
            "Access Route": intel["access"],
            "Notes": intel["recovery_notes"],
            "Deposit ID": deposit_id
        })

    return pd.DataFrame(rows).sort_values("Recovery Priority").reset_index(drop=True)


def get_phase_summary(current_hour: int, event_key: str = "helene_2024") -> dict:
    """
    Returns current operational phase and recommended actions
    based on hours elapsed since event onset.
    """
    if current_hour < 36:
        return {
            "phase": "PRE-EVENT",
            "color": "green",
            "headline": "PRE-EVENT STAGING WINDOW",
            "action": "Deploy rescue assets to staging zones. Evacuate high-risk communities.",
            "priority_sites": "Staging zones SZ-01 through SZ-07",
            "operation_type": "EVACUATION / PRE-POSITIONING"
        }
    elif current_hour < 72:
        return {
            "phase": "ACTIVE RESCUE",
            "color": "red",
            "headline": "ACTIVE RESCUE OPERATIONS",
            "action": "Execute rescue from pre-positioned staging zones. Focus on highest debris flow corridors.",
            "priority_sites": "Swananoa Valley, Chimney Rock Gorge, Lake Lure inlet",
            "operation_type": "RESCUE"
        }
    elif current_hour < 120:
        return {
            "phase": "TRANSITION",
            "color": "orange",
            "headline": "RESCUE TO RECOVERY TRANSITION",
            "action": "Continue active rescue where survivors possible. Begin systematic debris field search.",
            "priority_sites": "All primary accumulation zones - shift to recovery methodology",
            "operation_type": "RESCUE / RECOVERY"
        }
    else:
        return {
            "phase": "RECOVERY",
            "color": "purple",
            "headline": "BODY RECOVERY OPERATIONS",
            "action": "Systematic search of all ranked debris accumulation zones. Prioritize underwater sites.",
            "priority_sites": "Lake Lure inlet (#1), Chimney Rock gorge exit (#2), Craigtown complex (#3)",
            "operation_type": "RECOVERY"
        }
