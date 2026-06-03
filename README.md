# FloodFlow MVP
## WNC Flood and Debris Flow Prediction System
### Savage Ops | Adapt. Advance. Achieve.

---

## Overview

FloodFlow MVP is an AI-driven predictive analytics system for pre-positioning rescue personnel ahead of major flood and debris flow events in Western North Carolina.

The system ingests publicly available federal datasets, synthesizes them through machine learning models, and outputs geospatial recommendations for rescue staging zones before a storm event occurs.

**Target Watersheds:**
- Swananoa River (HUC 060101050600)
- French Broad River (HUC 06010105)
- Broad River / Rocky Broad (HUC 06010106)

**Calibration Event:** Hurricane Helene (September 27, 2024)

---

## Data Sources (All Free / Federal)

| Source | Data | URL |
|--------|------|-----|
| NOAA NWM | Streamflow forecasts | api.water.noaa.gov |
| USGS 3DEP | Pre/Post Helene LiDAR | OpenTopography |
| USGS NWIS | Real-time gauge data | waterservices.usgs.gov |
| USACE | Helene HWM dataset (2,587 pts) | STN Flood Event Viewer |
| NOAA Weather | Precipitation forecasts | api.weather.gov |

---

## Deployment

### Streamlit Cloud (Recommended)

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository, branch `main`, file `app.py`
5. Click Deploy

### Local Development

```bash
git clone https://github.com/savagefreedoms/floodflow-mvp
cd floodflow-mvp
pip install -r requirements.txt
streamlit run app.py
```

---

## System Architecture

```
LAYER 1: DATA SOURCES          LAYER 2: ML ENGINE         LAYER 3: OUTPUT
----------------------         ------------------         ---------------
NOAA NWM Forecasts      -->    LSTM Peak Predictor   -->  Hazard Maps
USGS LiDAR DEMs         -->    Debris Flow           -->  Timing Windows
USGS Stream Gauges      -->    Classifier            -->  Staging Zones
USACE Helene HWMs       -->    Alert Evaluator       -->  GPS Waypoints
NOAA Weather Forecasts  -->                          -->  Alert Dashboard
```

---

## Alert Tiers

| Tier | Condition | Action |
|------|-----------|--------|
| 1 - NORMAL | < 25% of Helene peak | Monitor |
| 2 - WATCH | 25-50% of Helene peak | Pre-position assets |
| 3 - WARNING | 50-75% of Helene peak | Deploy to staging zones |
| 4 - IMMINENT | > 75% of Helene peak | Execute deployment NOW |

---

## Staging Zones

Seven pre-defined rescue staging zones calibrated from Helene operational lessons learned:

- **SZ-01:** Black Mountain Fairgrounds (Swananoa upper)
- **SZ-02:** Swannanoa Community Center (Swananoa mid)
- **SZ-03:** Biltmore Village (French Broad)
- **SZ-04:** Chimney Rock State Park (Broad River gorge)
- **SZ-05:** Lake Lure Town Hall (Broad River downstream)
- **SZ-06:** Bat Cave / NC-9 Junction (Broad River Henderson)
- **SZ-07:** Henderson County Fairgrounds (logistics hub)

---

## License

Proprietary. Developed by Savage Ops. All rights reserved.

*Adapt. Advance. Achieve.*
