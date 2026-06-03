"""
Alert Tier Evaluation System
Evaluates current gauge conditions against Helene-calibrated thresholds
and returns operational alert tier for rescue pre-deployment decisions.
"""


def evaluate_alert_tier(gauge_data: dict, sensitivity: str = "Standard") -> str:
    """
    Evaluate current conditions and return alert tier.

    Tiers:
    - NORMAL: All stations below watch thresholds
    - WATCH: Elevated conditions, prepare assets
    - WARNING: Approaching Helene-scale, deploy to staging zones
    - IMMINENT: Debris flow expected within 6 hours, execute deployment
    """

    # Sensitivity multipliers
    sensitivity_map = {
        "Standard": 1.0,
        "Elevated": 0.85,
        "Maximum": 0.70
    }
    mult = sensitivity_map.get(sensitivity, 1.0)

    # Helene peak percentages at each station
    french_broad_pct = gauge_data.get("french_broad_asheville", {}).get("flood_pct", 0)
    swananoa_pct = gauge_data.get("swananoa_biltmore", {}).get("flood_pct", 0)
    broad_pct = gauge_data.get("broad_chimney_rock", {}).get("flood_pct", 0)
    max_pct = max(french_broad_pct, swananoa_pct, broad_pct)

    # Rate of rise check
    french_delta = gauge_data.get("french_broad_asheville", {}).get("delta_ft", 0)
    swananoa_delta = gauge_data.get("swananoa_biltmore", {}).get("delta_ft", 0)
    broad_delta = gauge_data.get("broad_chimney_rock", {}).get("delta_ft", 0)
    max_delta = max(french_delta, swananoa_delta, broad_delta)

    # Tier evaluation
    if max_pct >= (75 * mult) or (max_pct >= (60 * mult) and max_delta > 0.5):
        return "IMMINENT"
    elif max_pct >= (50 * mult) or max_delta > 0.3:
        return "WARNING"
    elif max_pct >= (25 * mult) or max_delta > 0.1:
        return "WATCH"
    else:
        return "NORMAL"


def format_alert_message(tier: str, gauge_data: dict) -> dict:
    """Returns formatted alert message and recommended actions by tier."""

    messages = {
        "NORMAL": {
            "color": "green",
            "headline": "TIER 1 - NORMAL CONDITIONS",
            "body": "All monitored reaches within normal flow parameters.",
            "actions": [
                "Continue standard monitoring cycle",
                "Verify all staging zone access routes are clear",
                "Confirm personnel and equipment readiness"
            ]
        },
        "WATCH": {
            "color": "yellow",
            "headline": "TIER 2 - ELEVATED WATCH",
            "body": "Flow levels elevated above baseline. Debris flow risk increasing.",
            "actions": [
                "Notify rescue team leadership immediately",
                "Pre-position assets at SZ-07 Henderson and SZ-03 Biltmore",
                "Increase gauge monitoring to 15-minute intervals",
                "Confirm road access to all staging zones"
            ]
        },
        "WARNING": {
            "color": "orange",
            "headline": "TIER 3 - FLOOD WARNING",
            "body": "Flow approaching Helene-scale thresholds. Debris flow probable.",
            "actions": [
                "DEPLOY rescue assets to all designated staging zones NOW",
                "Activate incident command structure",
                "Notify Henderson, Buncombe, Rutherford County Emergency Management",
                "Position boat assets at Lake Lure (SZ-05)",
                "Establish communications relay at Black Mountain (SZ-01)"
            ]
        },
        "IMMINENT": {
            "color": "red",
            "headline": "TIER 4 - DEBRIS FLOW IMMINENT",
            "body": "Debris flow predicted within 6 hours at flagged reaches. Execute deployment.",
            "actions": [
                "EXECUTE all pre-deployment orders immediately",
                "Evacuate civilians from Swananoa Valley, Chimney Rock, Bat Cave",
                "Block access to NC-9, US-64A Chimney Rock corridor",
                "All rescue personnel to assigned staging zones",
                "Activate mutual aid agreements with adjacent counties",
                "Request NCDOT road status confirmation for all access routes"
            ]
        }
    }

    return messages.get(tier, messages["NORMAL"])
