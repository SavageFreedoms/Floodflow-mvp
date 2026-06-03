"""
GIS Output Generator
Produces rescue staging zone recommendations and hazard map layers
based on ML model outputs and Helene ground truth data.
"""

import numpy as np
from datetime import datetime, timedelta


def generate_staging_zones() -> list:
    """
    Rescue staging zones with full operational intelligence.
    Each zone evaluated on four criteria:
    1. Ground safety - elevation above projected flood levels
    2. Population served - communities within response radius
    3. Access reliability - routes that stay open during flood events
    4. Choke point proximity - debris dam risk locations nearby
    """
    return [
        {
            "id": "SZ-01",
            "name": "Black Mountain Fairgrounds",
            "river": "Swananoa",
            "county": "Buncombe",
            "lat": 35.6143,
            "lon": -82.3243,
            "elevation_ft": 2180,
            "flood_safe": True,
            "flood_safe_reason": "Located 120ft above Swananoa 100-year flood elevation. Helene did not reach this site.",
            "capacity": "50 personnel / 20 vehicles",
            "access": "I-40 Exit 64 / US-70 W",
            "access_reliability": "HIGH",
            "access_notes": "I-40 stayed passable during Helene. US-70 W from Exit 64 is elevated above flood zone.",
            "population_served": ["Black Mountain (8,100)", "Swannanoa community (4,200)", "Old Fort corridor"],
            "population_notes": "Primary access point for upper Swananoa valley communities. Covers highest debris flow density from Helene.",
            "choke_points_nearby": [
                {"name": "Craigtown debris flow complex", "distance_mi": 2.1, "risk": "EXTREME",
                 "threat": "1.75-mile debris track. Dam failure would send debris wave downstream toward Black Mountain."},
                {"name": "Bee Tree Creek confluence", "distance_mi": 3.4, "risk": "HIGH",
                 "threat": "Tributary confluence debris accumulation. Periodic release events possible."}
            ],
            "commander_note": "Best forward staging for upper Swananoa. Safe ground, reliable access, covers highest victim density. Deploy here first.",
            "window": "Deploy 12-18 hrs before predicted peak",
            "deployment_priority": 1,
            "mission_type": ["RESCUE", "EVACUATION", "COMMAND POST"]
        },
        {
            "id": "SZ-02",
            "name": "Swannanoa Community Center",
            "river": "Swananoa",
            "county": "Buncombe",
            "lat": 35.5980,
            "lon": -82.4080,
            "elevation_ft": 2120,
            "flood_safe": True,
            "flood_safe_reason": "Above documented Helene inundation line by approximately 35ft. Verify on deployment.",
            "capacity": "75 personnel / 30 vehicles",
            "access": "US-70 Business / Old US-70",
            "access_reliability": "MODERATE",
            "access_notes": "US-70 Business sustained damage during Helene. Confirm road status before committing personnel. Have alternate route via I-40 ready.",
            "population_served": ["Swannanoa (4,200)", "Ridgecrest", "Adjacent residential corridors"],
            "population_notes": "Swannanoa community suffered catastrophic damage during Helene. Dense residential population in direct debris flow path.",
            "choke_points_nearby": [
                {"name": "Swannanoa Valley floodplain constriction", "distance_mi": 0.8, "risk": "HIGH",
                 "threat": "Valley narrows at railroad bridge. Debris dam formation likely. Dam failure threatens staging zone - evacuate if dam detected."},
            ],
            "commander_note": "High population density served but access reliability is MODERATE. Confirm road status before deployment. Have SZ-01 as fallback.",
            "window": "Deploy 8-12 hrs before predicted peak",
            "deployment_priority": 2,
            "mission_type": ["RESCUE", "EVACUATION"]
        },
        {
            "id": "SZ-03",
            "name": "Biltmore Village / US-25 Corridor",
            "river": "French Broad",
            "county": "Buncombe",
            "lat": 35.5600,
            "lon": -82.5300,
            "elevation_ft": 2134,
            "flood_safe": False,
            "flood_safe_reason": "WARNING: Lower sections of Biltmore Village flooded during Helene. Use upper US-25 parking only above 2,140ft elevation.",
            "capacity": "100 personnel / 40 vehicles",
            "access": "US-25 S from I-26 / Hendersonville Rd",
            "access_reliability": "HIGH",
            "access_notes": "US-25 from I-26 remained accessible during Helene. I-26 interchange provides reliable alternate routing.",
            "population_served": ["South Asheville", "Biltmore Estate area", "French Broad corridor residential"],
            "population_notes": "Largest single population served. French Broad mainstem flooding affects dense urban and suburban population.",
            "choke_points_nearby": [
                {"name": "French Broad / Swananoa confluence", "distance_mi": 1.2, "risk": "HIGH",
                 "threat": "Major confluence where combined debris load from both rivers deposits. Debris dam formation at confluence bend threatens upstream backflooding."},
            ],
            "commander_note": "Highest capacity zone but partial flood risk. USE UPPER STAGING AREA ONLY above 2,140ft. Do not stage below the US-25 / Hendersonville Rd intersection. Monitor French Broad stage continuously.",
            "window": "Deploy 6-10 hrs before predicted peak",
            "deployment_priority": 3,
            "mission_type": ["RESCUE", "COMMAND POST", "MEDICAL STAGING"]
        },
        {
            "id": "SZ-04",
            "name": "Chimney Rock State Park Upper Lot",
            "river": "Broad River",
            "county": "Rutherford",
            "lat": 35.4380,
            "lon": -82.2480,
            "elevation_ft": 1280,
            "flood_safe": False,
            "flood_safe_reason": "EXTREME CAUTION: Rocky Broad gorge produces highest debris flow velocities of any reach in the system. NC-9 was destroyed during Helene. Access from Lake Lure side only.",
            "capacity": "40 personnel / 15 vehicles",
            "access": "US-64A from Lake Lure - CONFIRM ROAD STATUS BEFORE ENTRY",
            "access_reliability": "LOW",
            "access_notes": "NC-9 through Chimney Rock gorge is HIGH RISK and was destroyed during Helene. Approach only from Lake Lure side via US-64A. Have extraction plan before entering.",
            "population_served": ["Chimney Rock community", "Bat Cave", "Gorge corridor residents"],
            "population_notes": "Small but extremely high-risk population. Gorge geography means minimal self-evacuation options. Communities are essentially trapped if roads fail.",
            "choke_points_nearby": [
                {"name": "Chimney Rock gorge narrows", "distance_mi": 0.2, "risk": "EXTREME",
                 "threat": "IMMEDIATE THREAT. Gorge constriction is primary debris dam formation point for entire Broad River system. Dam failure sends debris wave downstream at high velocity. This staging zone is in the impact zone if dam fails. HAVE EXTRACTION ROUTE READY AT ALL TIMES."},
                {"name": "Lake Lure inlet delta", "distance_mi": 1.8, "risk": "HIGH",
                 "threat": "Secondary accumulation point. Complete velocity loss at lake creates massive debris field."}
            ],
            "commander_note": "FORWARD POSITION - HIGHEST RISK STAGING ZONE. Only deploy here if you have confirmed road access, a standing extraction plan, and personnel who understand the gorge dam failure threat. Consider using SZ-05 Lake Lure as safer alternative with boat access.",
            "window": "Deploy 18-24 hrs before predicted peak",
            "deployment_priority": 4,
            "mission_type": ["RESCUE", "EVACUATION"]
        },
        {
            "id": "SZ-05",
            "name": "Lake Lure Town Hall / Marina",
            "river": "Broad River",
            "county": "Rutherford",
            "lat": 35.4250,
            "lon": -82.2050,
            "elevation_ft": 990,
            "flood_safe": True,
            "flood_safe_reason": "Town Hall and marina area above documented Helene flood line. Lake level monitoring available on site.",
            "capacity": "60 personnel / 25 vehicles",
            "access": "US-64A E from Chimney Rock / NC-9 S from Bat Cave",
            "access_reliability": "MODERATE",
            "access_notes": "US-64A sustained damage but remained passable. Monitor lake level - if Rocky Broad debris dam fails, lake level will spike rapidly.",
            "population_served": ["Lake Lure (1,100)", "Chimney Rock downstream", "Rutherford County lakeside communities"],
            "population_notes": "Lake Lure community suffered catastrophic property damage. Boat assets critical here - many residents were water-isolated after Helene.",
            "choke_points_nearby": [
                {"name": "Lake Lure inlet / Rocky Broad delta", "distance_mi": 0.5, "risk": "HIGH",
                 "threat": "120,000 CY debris deposit at lake inlet during Helene. Ongoing debris release into lake possible. Monitor lake depth changes."},
            ],
            "commander_note": "Preferred Broad River staging over SZ-04 due to better safety profile. BOAT ASSETS ARE ESSENTIAL HERE. Position at least two rescue watercraft at this site. Primary body recovery zone post-event.",
            "window": "Deploy 16-20 hrs before predicted peak",
            "deployment_priority": 5,
            "mission_type": ["RESCUE", "WATER RESCUE", "RECOVERY STAGING"]
        },
        {
            "id": "SZ-06",
            "name": "Bat Cave / NC-9 Junction",
            "river": "Broad River",
            "county": "Henderson",
            "lat": 35.4250,
            "lon": -82.1800,
            "elevation_ft": 1050,
            "flood_safe": True,
            "flood_safe_reason": "Junction sits above documented Helene flood elevation by approximately 60ft.",
            "capacity": "30 personnel / 12 vehicles",
            "access": "US-64 from Hendersonville / NC-9 from Lake Lure",
            "access_reliability": "MODERATE",
            "access_notes": "Very limited access routes. Confirm with Henderson County EM before committing. NC-9 through Bat Cave sustained significant damage during Helene.",
            "population_served": ["Bat Cave community", "Upper Henderson County rural"],
            "population_notes": "Small isolated rural population with very limited self-evacuation capability. Significant elderly population.",
            "choke_points_nearby": [
                {"name": "Bat Cave Road bridge", "distance_mi": 0.3, "risk": "HIGH",
                 "threat": "Bridge constriction creates debris dam. 45,000 CY debris deposited at this location during Helene. Structural integrity compromised - engineer assessment required before crossing."},
            ],
            "commander_note": "SMALL FORWARD POSITION. Limited capacity and access. Use only for direct Bat Cave community evacuation. Do not cross the Bat Cave bridge without engineer clearance.",
            "window": "Deploy 14-18 hrs before predicted peak",
            "deployment_priority": 6,
            "mission_type": ["EVACUATION", "RESCUE"]
        },
        {
            "id": "SZ-07",
            "name": "Henderson County Fairgrounds",
            "river": "French Broad",
            "county": "Henderson",
            "lat": 35.3150,
            "lon": -82.4600,
            "elevation_ft": 2130,
            "flood_safe": True,
            "flood_safe_reason": "Significantly above all flood projections. Never flooded in recorded history. Primary logistics hub.",
            "capacity": "150 personnel / 60 vehicles",
            "access": "US-64 / Four Seasons Blvd Hendersonville - I-26 accessible",
            "access_reliability": "HIGH",
            "access_notes": "Best access reliability of all seven zones. Multiple approach routes. I-26 interchange nearby. Did not flood during Helene.",
            "population_served": ["Hendersonville (14,000)", "Henderson County (120,000)", "All southern watershed communities"],
            "population_notes": "Largest population base. Primary logistics and medical hub for entire Henderson County operation.",
            "choke_points_nearby": [],
            "commander_note": "PRIMARY LOGISTICS AND COMMAND HUB. Safest site in the system. Highest capacity. Use as base of operations, medical staging, supply distribution, and personnel rotation point. Deploy here first if establishing incident command.",
            "window": "Deploy 24-36 hrs before predicted peak",
            "deployment_priority": 7,
            "mission_type": ["COMMAND POST", "LOGISTICS", "MEDICAL STAGING", "PERSONNEL ROTATION"]
        },
    ]

def generate_hazard_map() -> dict:
    """
    Generate GIS hazard map data for debris flow risk zones.
    Returns polygon and point data for map rendering.
    """
    hazard_polygons = [
        {
            "zone_id": "HZ-01",
            "name": "Swananoa Corridor High Risk",
            "risk": "HIGH",
            "bounds": [[35.60, -82.42], [35.62, -82.32]],
            "area_sqmi": 12.4,
            "helene_inundated": True
        },
        {
            "zone_id": "HZ-02",
            "name": "Chimney Rock Gorge Critical",
            "risk": "HIGH",
            "bounds": [[35.42, -82.26], [35.45, -82.22]],
            "area_sqmi": 3.8,
            "helene_inundated": True
        },
        {
            "zone_id": "HZ-03",
            "name": "French Broad Biltmore Floodplain",
            "risk": "MODERATE",
            "bounds": [[35.54, -82.54], [35.58, -82.48]],
            "area_sqmi": 8.2,
            "helene_inundated": True
        },
    ]

    return {
        "hazard_polygons": hazard_polygons,
        "generated_at": datetime.utcnow().isoformat(),
        "event_basis": "Hurricane Helene 2024-09-27",
        "model_version": "FloodFlow MVP v1.0"
    }
