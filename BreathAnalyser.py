
# ECONyx FULL-PPT PROTOTYPE (Colab) — "Breath-Analyzer (Nyx)"



import sys, os, json, math, time, hashlib, subprocess
from datetime import datetime, timedelta, timezone

# -------------------------
# 0) Install deps 
# -------------------------
def pipq(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install"] + pkgs)

try:
    import numpy as np
    import pandas as pd
    import requests
    import joblib
except Exception:
    pipq(["numpy", "pandas", "requests", "joblib"])
    import numpy as np
    import pandas as pd
    import requests
    import joblib

try:
    import pyarrow  # noqa
except Exception:
    pipq(["pyarrow"])

try:
    from tqdm import tqdm
except Exception:
    pipq(["tqdm"])
    from tqdm import tqdm

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    pipq(["scikit-learn"])
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:
    pipq(["xgboost"])
    from xgboost import XGBRegressor

try:
    import folium
    from folium.plugins import HeatMap
except Exception:
    pipq(["folium"])
    import folium
    from folium.plugins import HeatMap

# OSMnx (optional but used for traffic proxy + routing)
HAS_OSM = True
try:
    import osmnx as ox
    import networkx as nx
except Exception:
    HAS_OSM = False
    pipq(["osmnx", "networkx"])
    import osmnx as ox
    import networkx as nx
    HAS_OSM = True

from IPython.display import display

# -------------------------
# 1) CONFIG
# -------------------------
PROJECT_NAME = "Breath-Analyzer (Nyx) - Hyper-Local Predictive Air Quality Intelligence"

CITY_NAME   = "Varanasi"
CITY_CENTER = (25.3176, 82.9739)  # (lat, lon)

# Use "recent" for smooth demo runs. Switch to "historic" if you must.
RUN_MODE = "recent"  # "recent" or "historic"
RECENT_DAYS = 45

START_DATE_UTC = "2025-10-01"  # used only if RUN_MODE="historic"
END_DATE_UTC   = "2026-02-20"  # used only if RUN_MODE="historic"

HORIZON_H = 12  # 12 or 24

# Sensor search radius can expand if sparse
RADIUS_KM = 25
AUTO_EXPAND_RADIUS = True
RADIUS_CANDIDATES_KM = [25, 50, 75, 100, 150]

# Performance safety caps (prevents huge [6/8])
MAX_MAP_RADIUS_KM = 50
GRID_STEP_DEG = 0.04     # 0.02 prettier but slower
MAX_HEATMAP_POINTS = 7000  # controls HTML size (prevents "Failed to fetch")

# Station filtering
MIN_ROWS_PER_STATION = 120
MAX_FETCH_DAYS_PER_SENSOR = 365

# IDW
IDW_POWER = 2
IDW_K_NEAREST = 8

# Traffic proxy
TRAFFIC_ENABLED = True
TRAFFIC_GRID_ENABLED = True  # set False if you want even faster map
TRAFFIC_PEAK_AM = 9
TRAFFIC_PEAK_PM = 18
TRAFFIC_SIGMA_H = 3.0
TRAFFIC_BETA_RISK = 0.25

# Zone thresholds (on risk)
GREEN_MAX = 35.0
RED_MIN   = 55.0

# Routing demo
ORIGIN = CITY_CENTER
DEST   = (CITY_CENTER[0] + 0.05, CITY_CENTER[1] + 0.05)

# Alerts
ROUTE_SAMPLE_EVERY = 10

# Preview control: avoids heavy iframe rendering that triggers Colab "Failed to fetch"
PREVIEW_MAP_INLINE = True   # shows folium map object (lighter than iframe)
PRINT_FILE_ONLY = True      # also prints artifacts so you can download from Files panel

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CACHE_DIR = "cache_nyx"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# 2) OpenAQ key handling (NO hardcode)
# -------------------------
OPENAQ_API_KEY = (os.getenv("OPENAQ_API_KEY") or "").strip()
try:
    from google.colab import userdata  # type: ignore
    if not OPENAQ_API_KEY:
        OPENAQ_API_KEY = (userdata.get("OPENAQ_API_KEY") or "").strip()
except Exception:
    pass

if not OPENAQ_API_KEY:
    raise RuntimeError("Missing OPENAQ_API_KEY. Add it in Colab Secrets as OPENAQ_API_KEY and restart runtime.")

def openaq_headers():
    return {"accept": "application/json", "X-API-Key": OPENAQ_API_KEY, "User-Agent": "nyx-prototype/1.0"}

def safe_request(url, headers=None, params=None, retries=7, base_sleep=1.2, timeout=60):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(base_sleep * (i + 1))
                last = f"{r.status_code}: {r.text[:220]}"
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(base_sleep * (i + 1))
    raise RuntimeError(f"Request failed after retries. Last error: {last}")

# Preflight auth check
_test = requests.get("https://api.openaq.org/v3/locations/8118", headers=openaq_headers(), timeout=30)
if _test.status_code == 401:
    raise RuntimeError("OpenAQ 401 Unauthorized: key not accepted. Check Colab Secrets name OPENAQ_API_KEY and rotate key if needed.")

# -------------------------
# 3) Time helpers
# -------------------------
def parse_dt_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def resolve_dates():
    if RUN_MODE.lower().strip() == "recent":
        today = datetime.now(timezone.utc).date()
        end = today - timedelta(days=1)
        start = end - timedelta(days=int(RECENT_DAYS))
        return str(start), str(end)
    return START_DATE_UTC, END_DATE_UTC

START_DATE_UTC, END_DATE_UTC = resolve_dates()
start_dt = parse_dt_date(START_DATE_UTC)
end_dt = parse_dt_date(END_DATE_UTC) + timedelta(days=1) - timedelta(seconds=1)

print(f"Project: {PROJECT_NAME}")
print(f"City: {CITY_NAME} | Center: {CITY_CENTER}")
print(f"Mode: {RUN_MODE} | UTC range: {START_DATE_UTC} → {END_DATE_UTC} | Horizon: +{HORIZON_H}h")

# -------------------------
# 4) Geometry + cache
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*(2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

def bbox_from_center_radius_km(lat, lon, radius_km):
    lat_span = radius_km / 111.0
    lon_span = radius_km / (111.0 * math.cos(math.radians(lat)) + 1e-9)
    return (lon - lon_span, lat - lat_span, lon + lon_span, lat + lat_span)

def haversine_vec_km(lat, lon, lats, lons):
    R = 6371.0
    lat1 = np.radians(lat); lon1 = np.radians(lon)
    lat2 = np.radians(np.asarray(lats, dtype=float))
    lon2 = np.radians(np.asarray(lons, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*(2*np.arctan2(np.sqrt(a), np.sqrt(1-a)))

def cache_path(prefix: str, key: str, ext="parquet"):
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.{ext}")

def cache_load_parquet(path):
    if os.path.exists(path):
        try: return pd.read_parquet(path)
        except Exception: return None
    return None

def cache_save_parquet(df, path):
    try: df.to_parquet(path, index=False)
    except Exception: pass

# -------------------------
# 5) OpenAQ data
# -------------------------
def fetch_locations_pm25_sensors(center_lat, center_lon, radius_km):
    base = "https://api.openaq.org/v3/locations"
    page = 1
    sensors = []

    if radius_km <= 25:
        params_base = {
            "coordinates": f"{center_lat:.4f},{center_lon:.4f}",
            "radius": int(radius_km * 1000),
            "limit": 1000,
            "page": 1,
            "order_by": "id",
            "sort_order": "desc",
        }
    else:
        lon_min, lat_min, lon_max, lat_max = bbox_from_center_radius_km(center_lat, center_lon, radius_km)
        params_base = {
            "bbox": f"{lon_min:.4f},{lat_min:.4f},{lon_max:.4f},{lat_max:.4f}",
            "limit": 1000,
            "page": 1,
            "order_by": "id",
            "sort_order": "desc",
        }

    while True:
        params = dict(params_base); params["page"] = page
        r = safe_request(base, headers=openaq_headers(), params=params)
        results = r.json().get("results", [])
        if not results:
            break

        for loc in results:
            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude"); lon = coords.get("longitude")
            if lat is None or lon is None:
                continue

            if radius_km > 25:
                if haversine_km(center_lat, center_lon, float(lat), float(lon)) > radius_km + 2:
                    continue

            for s in (loc.get("sensors") or []):
                p = s.get("parameter") or {}
                if (p.get("name") or "").lower() != "pm25":
                    continue
                sid = s.get("id")
                if sid is None:
                    continue
                sensors.append({
                    "sensors_id": int(sid),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location_name": loc.get("name"),
                    "sensor_name": s.get("name"),
                    "openaq_location_id": loc.get("id"),
                })
        page += 1

    return pd.DataFrame(sensors).drop_duplicates(subset=["sensors_id"]).reset_index(drop=True)

def fetch_sensor_meta(sensors_id: int) -> dict:
    key = f"meta:{sensors_id}"
    p = cache_path("meta", key, ext="json")
    if os.path.exists(p):
        try: return json.load(open(p, "r"))
        except Exception: pass
    url = f"https://api.openaq.org/v3/sensors/{int(sensors_id)}"
    js = safe_request(url, headers=openaq_headers()).json()
    res = js.get("results", [])
    meta = res[0] if res else {}
    try: json.dump(meta, open(p, "w"))
    except Exception: pass
    return meta

def parse_sensor_dt(meta: dict, key: str):
    v = meta.get(key)
    if isinstance(v, dict):
        s = v.get("utc") or v.get("local")
        if s: return pd.to_datetime(s, utc=True)
    if isinstance(v, str):
        return pd.to_datetime(v, utc=True)
    return None

def fetch_hours_pm25(sensors_id: int, start_dt: datetime, end_dt: datetime, limit=1000):
    base = f"https://api.openaq.org/v3/sensors/{int(sensors_id)}/hours"
    rows, page = [], 1
    while True:
        params = {"datetime_from": to_utc_iso(start_dt), "datetime_to": to_utc_iso(end_dt), "limit": limit, "page": page}
        js = safe_request(base, headers=openaq_headers(), params=params).json()
        results = js.get("results", [])
        if not results:
            break
        for m in results:
            dt = None
            if isinstance(m.get("datetime"), dict):
                dt = m["datetime"].get("utc") or m["datetime"].get("local")
            val = m.get("value")
            if dt is None or val is None:
                continue
            rows.append({"sensors_id": int(sensors_id), "time": pd.to_datetime(dt, utc=True), "pm25": float(val)})
        page += 1
    return pd.DataFrame(rows)

def fetch_raw_pm25(sensors_id: int, start_dt: datetime, end_dt: datetime, limit=1000):
    base = f"https://api.openaq.org/v3/sensors/{int(sensors_id)}/measurements"
    rows, page = [], 1
    while True:
        params = {"datetime_from": to_utc_iso(start_dt), "datetime_to": to_utc_iso(end_dt), "limit": limit, "page": page}
        js = safe_request(base, headers=openaq_headers(), params=params).json()
        results = js.get("results", [])
        if not results:
            break
        for m in results:
            val = m.get("value")
            if val is None:
                continue
            dt = None
            period = m.get("period")
            if isinstance(period, dict):
                dto = period.get("datetimeTo"); dfr = period.get("datetimeFrom")
                if isinstance(dto, dict):
                    dt = dto.get("utc") or dto.get("local")
                if dt is None and isinstance(dfr, dict):
                    dt = dfr.get("utc") or dfr.get("local")
            if dt is None and isinstance(m.get("datetime"), dict):
                dt = m["datetime"].get("utc") or m["datetime"].get("local")
            if dt is None:
                continue
            rows.append({"sensors_id": int(sensors_id), "time": pd.to_datetime(dt, utc=True), "pm25": float(val)})
        page += 1
    return pd.DataFrame(rows)

def clean_and_hourly(df_pm: pd.DataFrame):
    if df_pm.empty:
        return df_pm
    df = df_pm.dropna(subset=["time", "pm25"]).copy()
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= 1500)]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["time_hour"] = df["time"].dt.floor("h")
    dfh = df.groupby(["sensors_id", "time_hour"], as_index=False)["pm25"].mean()
    return dfh.rename(columns={"time_hour": "time"})

def fetch_pm25_any_cached(sensors_id: int, start_dt: datetime, end_dt: datetime):
    key = f"pm25:{sensors_id}:{to_utc_iso(start_dt)}:{to_utc_iso(end_dt)}"
    p = cache_path("pm", key, ext="parquet")
    cached = cache_load_parquet(p)
    if cached is not None and len(cached) > 0:
        return cached, "cache"

    dfh = fetch_hours_pm25(sensors_id, start_dt, end_dt)
    if len(dfh) > 0:
        df, src = dfh, "hours"
    else:
        dfr = fetch_raw_pm25(sensors_id, start_dt, end_dt)
        df, src = dfr, ("measurements" if len(dfr) > 0 else "none")

    if len(df) > 0:
        cache_save_parquet(df, p)
    return df, src

# -------------------------
# 6) Weather
# -------------------------
def fetch_openmeteo_hourly(lat, lon, start_date_utc: str, end_date_utc: str):
    key = f"wx:{lat:.4f}:{lon:.4f}:{start_date_utc}:{end_date_utc}"
    p = cache_path("wx", key, ext="parquet")
    cached = cache_load_parquet(p)
    if cached is not None and len(cached) > 0:
        return cached

    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_utc,
        "end_date": end_date_utc,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "wind_speed_10m", "wind_direction_10m", "surface_pressure"
        ],
        "timezone": "UTC",
    }
    js = safe_request(base, params=params).json()
    h = js.get("hourly", {})
    if not h or "time" not in h:
        return pd.DataFrame()

    t = pd.to_datetime(h["time"], utc=True)
    df = pd.DataFrame({
        "time": t,
        "temp": h.get("temperature_2m", [np.nan]*len(t)),
        "rh": h.get("relative_humidity_2m", [np.nan]*len(t)),
        "precip": h.get("precipitation", [np.nan]*len(t)),
        "wind_speed": h.get("wind_speed_10m", [np.nan]*len(t)),
        "wind_dir": h.get("wind_direction_10m", [np.nan]*len(t)),
        "pressure": h.get("surface_pressure", [np.nan]*len(t)),
    })
    cache_save_parquet(df, p)
    return df

# -------------------------
# 7) Traffic proxy + routing (fast)
# -------------------------
HIGHWAY_FACTOR = {
    "motorway": 1.00, "trunk": 0.95, "primary": 0.85, "secondary": 0.70,
    "tertiary": 0.55, "residential": 0.35, "unclassified": 0.40, "service": 0.25,
    "living_street": 0.20
}
HIGHWAY_SPEED_KPH = {
    "motorway": 70, "trunk": 55, "primary": 40, "secondary": 32,
    "tertiary": 25, "residential": 18, "unclassified": 20, "service": 12,
    "living_street": 10
}

def traffic_hour_multiplier(hour):
    am = math.exp(-((hour - TRAFFIC_PEAK_AM)/TRAFFIC_SIGMA_H)**2)
    pm = math.exp(-((hour - TRAFFIC_PEAK_PM)/TRAFFIC_SIGMA_H)**2)
    return 0.6 + 0.6*am + 0.8*pm

def normalize01(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + eps)

def edge_highway(edge_attr):
    h = edge_attr.get("highway")
    if isinstance(h, list) and len(h) > 0:
        h = h[0]
    if not isinstance(h, str):
        h = "unclassified"
    return h

def build_road_graph(center_lat, center_lon, radius_km):
    dist_m = int(radius_km * 1000)
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="drive", simplify=True)

    # Ensure length exists across versions
    try:
        has_length = any(("length" in data) for _, _, _, data in G.edges(keys=True, data=True))
    except Exception:
        has_length = True

    if not has_length:
        if hasattr(ox, "add_edge_lengths"):
            G = ox.add_edge_lengths(G)
        elif hasattr(ox, "distance") and hasattr(ox.distance, "add_edge_lengths"):
            G = ox.distance.add_edge_lengths(G)

    return G

def attach_edge_speeds_and_base_traffic(G):
    for u, v, k, data in G.edges(keys=True, data=True):
        hwy = edge_highway(data)
        base = HIGHWAY_FACTOR.get(hwy, 0.40)
        spd = HIGHWAY_SPEED_KPH.get(hwy, 20)
        data["traffic_base"] = float(base)
        data["speed_kph"] = float(spd)
        length_m = float(data.get("length", 1.0))
        data["travel_time_s"] = length_m / (spd*1000/3600 + 1e-9)
    return G

def nearest_edges_any(G, lats, lons):
    try:
        res = ox.distance.nearest_edges(G, X=np.asarray(lons), Y=np.asarray(lats))
        if isinstance(res, tuple) and len(res) == 3:
            if not hasattr(res[0], "__len__"):  # scalars
                return [res[0]], [res[1]], [res[2]]
            return list(res[0]), list(res[1]), list(res[2])
    except Exception:
        pass

    us, vs, ks = [], [], []
    for la, lo in zip(lats, lons):
        r = ox.distance.nearest_edges(G, X=float(lo), Y=float(la))
        if isinstance(r[0], (list, tuple, np.ndarray)):
            u, v, k = r[0][0], r[1][0], r[2][0]
        else:
            u, v, k = r
        us.append(u); vs.append(v); ks.append(k)
    return us, vs, ks

TRAFFIC_BASE_CACHE = {}

def get_station_traffic_base(G, lat, lon):
    key = (round(lat, 5), round(lon, 5))
    if key in TRAFFIC_BASE_CACHE:
        return TRAFFIC_BASE_CACHE[key]
    us, vs, ks = nearest_edges_any(G, [lat], [lon])
    data = G.get_edge_data(us[0], vs[0], key=ks[0])
    base = float(data.get("traffic_base", 0.40))
    TRAFFIC_BASE_CACHE[key] = base
    return base

def traffic_series_from_base(times_utc, base):
    hrs = pd.to_datetime(times_utc, utc=True).dt.hour.values
    mult = np.array([traffic_hour_multiplier(int(h)) for h in hrs], dtype=float)
    return base * mult

def traffic_for_points_fast(G, lats, lons, hour):
    us, vs, ks = nearest_edges_any(G, lats, lons)
    bases = []
    for u, v, k in zip(us, vs, ks):
        data = G.get_edge_data(u, v, key=k)
        bases.append(float(data.get("traffic_base", 0.40)))
    bases = np.asarray(bases, dtype=float)
    return bases * traffic_hour_multiplier(int(hour))

def route_between(G, origin_latlon, dest_latlon, weight="travel_time_s"):
    o_lat, o_lon = origin_latlon
    d_lat, d_lon = dest_latlon
    orig = ox.distance.nearest_nodes(G, X=o_lon, Y=o_lat)
    dest = ox.distance.nearest_nodes(G, X=d_lon, Y=d_lat)
    return nx.shortest_path(G, orig, dest, weight=weight)

# -------------------------
# 8) Feature engineering
# -------------------------
def add_time_features(df):
    t = pd.to_datetime(df["time"], utc=True)
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["day"] = t.dt.day
    return df

def make_features(df_station, horizon_h):
    df = df_station.sort_values("time").copy()
    df["pm25_now"] = df["pm25"]

    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f"pm25_lag{lag}"] = df["pm25"].shift(lag)

    for w in [3, 6, 12, 24]:
        df[f"pm25_rollmean{w}"] = df["pm25"].rolling(w).mean()
        df[f"pm25_rollstd{w}"] = df["pm25"].rolling(w).std()

    df["y"] = df["pm25"].shift(-horizon_h)
    df = add_time_features(df)

    for c in ["temp","rh","precip","wind_speed","wind_dir","pressure"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill()

    if "traffic" in df.columns:
        df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce").fillna(0.0)

    df = df.dropna().reset_index(drop=True)
    return df

# -------------------------
# 9) Grid + fast IDW
# -------------------------
def make_grid(center_lat, center_lon, radius_km, step_deg=0.04):
    lat_span = radius_km / 111.0
    lon_span = radius_km / (111.0 * math.cos(math.radians(center_lat)) + 1e-9)
    lat_min, lat_max = center_lat - lat_span, center_lat + lat_span
    lon_min, lon_max = center_lon - lon_span, center_lon + lon_span
    lats = np.arange(lat_min, lat_max + step_deg, step_deg)
    lons = np.arange(lon_min, lon_max + step_deg, step_deg)
    grid = []
    for la in lats:
        for lo in lons:
            if haversine_km(center_lat, center_lon, float(la), float(lo)) <= radius_km:
                grid.append((float(la), float(lo)))
    return grid

def idw_two_surfaces_fast(grid, sensor_lats, sensor_lons, now_vals, fut_vals, power=2, k=8):
    sensor_lats = np.asarray(sensor_lats, dtype=float)
    sensor_lons = np.asarray(sensor_lons, dtype=float)
    now_vals = np.asarray(now_vals, dtype=float)
    fut_vals = np.asarray(fut_vals, dtype=float)

    k = min(int(k), len(sensor_lats))
    out_now = np.empty(len(grid), dtype=float)
    out_fut = np.empty(len(grid), dtype=float)

    for i, (la, lo) in enumerate(grid):
        d = haversine_vec_km(la, lo, sensor_lats, sensor_lons)
        j0 = int(np.argmin(d))
        if d[j0] < 1e-6:
            out_now[i] = now_vals[j0]
            out_fut[i] = fut_vals[j0]
            continue
        idx = np.argpartition(d, k-1)[:k]
        dk = d[idx]
        wk = 1.0 / (dk**power + 1e-12)
        wsum = wk.sum()
        out_now[i] = float((wk * now_vals[idx]).sum() / wsum)
        out_fut[i] = float((wk * fut_vals[idx]).sum() / wsum)

    return out_now, out_fut

def classify_zone(risk_value):
    if risk_value >= RED_MIN: return "RED"
    if risk_value <= GREEN_MAX: return "GREEN"
    return "YELLOW"

def pm25_at_point_from_grid(lat, lon, grid_df, col="risk"):
    dlat = (grid_df["lat"] - lat).abs()
    dlon = (grid_df["lon"] - lon).abs()
    idx = (dlat + dlon).idxmin()
    return float(grid_df.loc[idx, col])

def compute_route_exposure(G, route_nodes, grid_df, col="risk"):
    total = 0.0
    total_len = 0.0
    coords = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        ed = G.get_edge_data(u, v)
        if ed is None:
            continue
        if isinstance(ed, dict):
            data = min(ed.values(), key=lambda d: d.get("length", 1e9))
        else:
            data = ed
        length = float(data.get("length", 0.0))

        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        mid_lon, mid_lat = (x1+x2)/2, (y1+y2)/2

        risk = pm25_at_point_from_grid(mid_lat, mid_lon, grid_df, col=col)
        total += length * risk
        total_len += length
        coords.append((y1, x1))
    coords.append((G.nodes[route_nodes[-1]]["y"], G.nodes[route_nodes[-1]]["x"]))
    return total, total_len, coords

# -------------------------
# 10) RUN PIPELINE
# -------------------------
print("\n[1/8] Fetching sensors...")
radii_to_try = [RADIUS_KM] if not AUTO_EXPAND_RADIUS else [r for r in RADIUS_CANDIDATES_KM if r >= RADIUS_KM]

final_radius_used = None
final_loc_df = None
final_pm_all = None

for radius_try in radii_to_try:
    loc_df = fetch_locations_pm25_sensors(CITY_CENTER[0], CITY_CENTER[1], radius_try)
    print(f"Radius {radius_try} km → candidate sensors: {len(loc_df)}")
    if loc_df.empty:
        continue

    print("[2/8] Fetching PM2.5 per sensor (cached)...")
    expected_hours = int((end_dt - start_dt).total_seconds() // 3600) + 1
    keep_threshold = min(MIN_ROWS_PER_STATION, max(120, int(0.15 * expected_hours)))

    pm_dfs = []
    diag_rows = []

    for _, row in tqdm(loc_df.iterrows(), total=len(loc_df)):
        sid = int(row["sensors_id"])
        try:
            meta = fetch_sensor_meta(sid)
            dt_first = parse_sensor_dt(meta, "datetimeFirst")
            dt_last  = parse_sensor_dt(meta, "datetimeLast")

            eff_end = min(end_dt, dt_last) if dt_last is not None else end_dt
            recent_start = eff_end - timedelta(days=MAX_FETCH_DAYS_PER_SENSOR)
            eff_start = max(start_dt, recent_start, dt_first) if dt_first is not None else max(start_dt, recent_start)

            if eff_end <= eff_start + timedelta(days=2):
                diag_rows.append({"sensors_id": sid, "kept": False, "reason": "too_short"})
                continue

            df_raw, src = fetch_pm25_any_cached(sid, eff_start, eff_end)
            df_h = clean_and_hourly(df_raw)

            kept = len(df_h) >= keep_threshold
            diag_rows.append({"sensors_id": sid, "kept": kept, "source": src, "rows_hourly": int(len(df_h))})
            if kept:
                pm_dfs.append(df_h)

        except Exception as e:
            diag_rows.append({"sensors_id": sid, "kept": False, "reason": str(e)[:120]})

    diag = pd.DataFrame(diag_rows)
    print("Coverage (top 20):")
    display(diag.sort_values(["kept","rows_hourly"], ascending=[False,False]).head(20))

    if not pm_dfs:
        print(f"No usable stations at radius={radius_try} km. Trying next...")
        continue

    pm_all = pd.concat(pm_dfs, ignore_index=True)
    kept_ids = sorted(pm_all["sensors_id"].unique().tolist())
    loc_df = loc_df[loc_df["sensors_id"].isin(kept_ids)].reset_index(drop=True)

    final_radius_used = radius_try
    final_loc_df = loc_df
    final_pm_all = pm_all
    break

if final_pm_all is None:
    raise RuntimeError("No stations passed coverage. Try RUN_MODE='recent' or larger radius candidates.")

print(f"\n Using radius {final_radius_used} km with sensors: {len(final_loc_df)}")

print("\n[3/8] Fetching weather (Open-Meteo)...")
wx = fetch_openmeteo_hourly(CITY_CENTER[0], CITY_CENTER[1], START_DATE_UTC, END_DATE_UTC)
if wx.empty:
    raise RuntimeError("Weather fetch returned empty.")

G = None
if TRAFFIC_ENABLED and HAS_OSM:
    print("\n[4/8] Building road graph (traffic proxy + routing)...")
    try:
        osm_radius = min(final_radius_used, MAX_MAP_RADIUS_KM)
        G = build_road_graph(CITY_CENTER[0], CITY_CENTER[1], osm_radius)
        G = attach_edge_speeds_and_base_traffic(G)
        print("OSM graph OK. Nodes:", len(G.nodes), "Edges:", len(G.edges))
    except Exception as e:
        print("OSM graph failed -> continuing without traffic/routing. Error:", str(e)[:200])
        G = None
else:
    print("\n[4/8] Traffic disabled or OSM unavailable — continuing without traffic/routing.")

print("\n[5/8] Feature engineering (AQI + weather + traffic) and training model...")
datasets = []
sensor_ids = sorted(final_pm_all["sensors_id"].unique().tolist())

for sid in tqdm(sensor_ids):
    df_s = final_pm_all[final_pm_all["sensors_id"] == sid].sort_values("time")
    df_s = df_s.merge(wx, on="time", how="left")

    df_s["traffic"] = 0.0
    if G is not None:
        la = float(final_loc_df.loc[final_loc_df["sensors_id"]==sid, "latitude"].iloc[0])
        lo = float(final_loc_df.loc[final_loc_df["sensors_id"]==sid, "longitude"].iloc[0])
        base = get_station_traffic_base(G, la, lo)   # ONE OSM lookup per station
        df_s["traffic"] = traffic_series_from_base(df_s["time"], base)

    df_feat = make_features(df_s, HORIZON_H)
    if len(df_feat) >= 200:
        df_feat["sensors_id"] = sid
        datasets.append(df_feat)

if not datasets:
    raise RuntimeError("No usable feature datasets. Reduce RECENT_DAYS or expand radius, or set RUN_MODE='recent'.")

data = pd.concat(datasets, ignore_index=True)
data = data.merge(final_loc_df[["sensors_id","latitude","longitude"]], on="sensors_id", how="left")
data["time"] = pd.to_datetime(data["time"], utc=True)
data = data.sort_values("time").reset_index(drop=True)

feature_cols = (
    ["pm25_now"]
    + [c for c in data.columns if c.startswith("pm25_lag")]
    + [c for c in data.columns if c.startswith("pm25_roll")]
    + ["temp","rh","precip","wind_speed","wind_dir","pressure"]
    + ["hour","dow","month","day"]
    + ["latitude","longitude"]
    + ["traffic"]
)
feature_cols = [c for c in feature_cols if c in data.columns]

X = data[feature_cols].astype(float)
y = data["y"].astype(float)

baseline_pred = X["pm25_now"].values
def print_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"{label} MAE: {mae:.3f} | RMSE: {rmse:.3f} | R^2: {r2:.3f}")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

baseline_metrics = print_metrics(y, baseline_pred, label="Persistence baseline")

model = XGBRegressor(
    n_estimators=700,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = []
for fold, (tr, te) in enumerate(tscv.split(X), start=1):
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    fold_metrics.append(print_metrics(y.iloc[te], pred, label=f"Fold {fold}"))

model.fit(X, y)
cv_mean = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
print("\nCV mean:", cv_mean)

print("\n[6/8] Creating city surfaces (IDW + Forecast + Traffic + Zones)...")

map_time = data["time"].max()
last_rows = data[data["time"] == map_time].groupby("sensors_id").tail(1)
if last_rows.empty:
    last_rows = data.sort_values("time").groupby("sensors_id").tail(1)

now_points, fut_points = [], []
for _, r in last_rows.iterrows():
    la, lo = float(r["latitude"]), float(r["longitude"])
    now_val = float(r["pm25_now"])
    xrow = pd.DataFrame([{c: float(r[c]) for c in feature_cols}])
    fut_val = float(model.predict(xrow)[0])
    now_points.append((la, lo, now_val))
    fut_points.append((la, lo, fut_val))

map_radius = min(final_radius_used, MAX_MAP_RADIUS_KM)
grid = make_grid(CITY_CENTER[0], CITY_CENTER[1], map_radius, step_deg=GRID_STEP_DEG)
print("Radius sensors:", final_radius_used, "| Radius map:", map_radius, "| Grid points:", len(grid), "| Sensors:", len(now_points))

sensor_lats = [p[0] for p in now_points]
sensor_lons = [p[1] for p in now_points]
now_vals    = [p[2] for p in now_points]
fut_vals    = [p[2] for p in fut_points]

pm_now_surf, pm_fut_surf = idw_two_surfaces_fast(grid, sensor_lats, sensor_lons, now_vals, fut_vals, power=IDW_POWER, k=IDW_K_NEAREST)

grid_lats = [la for la, _ in grid]
grid_lons = [lo for _, lo in grid]

if G is not None and TRAFFIC_GRID_ENABLED:
    try:
        grid_traffic = traffic_for_points_fast(G, grid_lats, grid_lons, int(map_time.hour))
    except Exception:
        grid_traffic = np.zeros(len(grid), dtype=float)
else:
    grid_traffic = np.zeros(len(grid), dtype=float)

traffic_norm = normalize01(grid_traffic)
risk = pm_fut_surf * (1.0 + TRAFFIC_BETA_RISK * traffic_norm)
zones = np.array([("RED" if v >= RED_MIN else ("GREEN" if v <= GREEN_MAX else "YELLOW")) for v in risk], dtype=object)

grid_df = pd.DataFrame({
    "lat": [p[0] for p in grid],
    "lon": [p[1] for p in grid],
    "pm25_now": pm_now_surf,
    "pm25_fut": pm_fut_surf,
    "traffic": grid_traffic,
    "traffic_norm": traffic_norm,
    "risk": risk,
    "zone": zones
})

red_pts = grid_df[grid_df["zone"]=="RED"][["lat","lon","risk"]].copy()
green_pts = grid_df[grid_df["zone"]=="GREEN"][["lat","lon","risk"]].copy()

# Routing
route_fast_coords = None
route_clean_coords = None
route_metrics = {}

if G is not None:
    print("\nRouting: computing fastest route and green corridor route...")
    try:
        fast_nodes = route_between(G, ORIGIN, DEST, weight="travel_time_s")
        fast_exposure, fast_len_m, fast_coords = compute_route_exposure(G, fast_nodes, grid_df, col="risk")

        for u, v, k, data_e in G.edges(keys=True, data=True):
            length = float(data_e.get("length", 0.0))
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            mid_lon, mid_lat = (x1+x2)/2, (y1+y2)/2
            data_e["exposure_w"] = length * pm25_at_point_from_grid(mid_lat, mid_lon, grid_df, col="risk")

        clean_nodes = route_between(G, ORIGIN, DEST, weight="exposure_w")
        clean_exposure, clean_len_m, clean_coords = compute_route_exposure(G, clean_nodes, grid_df, col="risk")

        route_fast_coords = fast_coords
        route_clean_coords = clean_coords

        exposure_reduction_pct = 0.0
        if fast_exposure > 1e-9:
            exposure_reduction_pct = 100.0 * (fast_exposure - clean_exposure) / fast_exposure

        route_metrics = {
            "fastest": {"len_m": fast_len_m, "exposure_score": fast_exposure},
            "green_corridor": {"len_m": clean_len_m, "exposure_score": clean_exposure},
            "exposure_reduction_pct": exposure_reduction_pct
        }
        print("Route metrics:", route_metrics)
    except Exception as e:
        print("Routing failed; continuing without route layers. Error:", str(e)[:200])

# Alerts
print("\n[7/8] Building predictive alerts (entry + route sampling)...")

alerts = []

def point_in_red(lat, lon):
    return (("RED") == ("RED" if pm25_at_point_from_grid(lat, lon, grid_df, col="risk") >= RED_MIN else ("GREEN" if pm25_at_point_from_grid(lat, lon, grid_df, col="risk") <= GREEN_MAX else "YELLOW")))

if point_in_red(ORIGIN[0], ORIGIN[1]):
    alerts.append({"type":"ENTRY_ALERT", "where":"ORIGIN", "message":"Origin is in/near predicted RED zone. Consider rerouting or delaying."})
if point_in_red(DEST[0], DEST[1]):
    alerts.append({"type":"ENTRY_ALERT", "where":"DEST", "message":"Destination is in/near predicted RED zone. Consider mask/alternate path."})

def route_hits_red(route_coords, sample_every=10):
    if not route_coords:
        return False
    for i in range(0, len(route_coords), max(1, int(sample_every))):
        la, lo = route_coords[i]
        if point_in_red(la, lo):
            return True
    return False

if route_fast_coords and route_hits_red(route_fast_coords, ROUTE_SAMPLE_EVERY):
    alerts.append({"type":"ROUTE_ALERT", "where":"FASTEST_ROUTE", "message":"Fastest route intersects predicted RED zones. Green corridor recommended."})
if route_clean_coords and route_hits_red(route_clean_coords, ROUTE_SAMPLE_EVERY):
    alerts.append({"type":"ROUTE_ALERT", "where":"GREEN_CORRIDOR", "message":"Green corridor still intersects RED zones. Consider time shift."})

def simulate_compliance(exposure_fast, exposure_clean, p=0.7, n=2000):
    if not np.isfinite(exposure_fast) or not np.isfinite(exposure_clean):
        return None
    rng = np.random.RandomState(RANDOM_SEED)
    choices = rng.rand(n) < p
    exp = np.where(choices, exposure_clean, exposure_fast)
    return {
        "compliance_prob": float(p),
        "simulated_compliance_rate": float(choices.mean()),
        "avg_exposure_score": float(exp.mean()),
        "avg_exposure_reduction_pct": float(100.0*(exposure_fast - exp.mean())/(exposure_fast+1e-9))
    }

compliance_metrics = None
if route_metrics:
    compliance_metrics = simulate_compliance(
        route_metrics["fastest"]["exposure_score"],
        route_metrics["green_corridor"]["exposure_score"],
        p=0.7
    )

# Map helpers to prevent huge HTML (avoids "Failed to fetch")
def heat_points(df, value_col, max_points=7000, seed=42):
    if len(df) > max_points:
        df = df.sample(max_points, random_state=seed)
    return df[["lat","lon",value_col]].values.tolist()

print("\n[8/8] Generating interactive map (lightweight layers)...")

m = folium.Map(location=list(CITY_CENTER), zoom_start=11, tiles="cartodbpositron")

# Station markers
for la, lo, v in now_points:
    folium.CircleMarker(
        location=[la, lo], radius=6,
        popup=f"<b>PM2.5 (latest)</b>: {v:.1f} µg/m³<br><small>{map_time} UTC</small>",
        fill=True, fill_opacity=0.9, weight=1
    ).add_to(m)

layer_now = folium.FeatureGroup(name="PM2.5 NOW (IDW)")
HeatMap(heat_points(grid_df, "pm25_now", MAX_HEATMAP_POINTS), radius=18, blur=16).add_to(layer_now)
layer_now.add_to(m)

layer_fut = folium.FeatureGroup(name=f"PM2.5 +{HORIZON_H}h Forecast (Model+IDW)")
HeatMap(heat_points(grid_df, "pm25_fut", MAX_HEATMAP_POINTS), radius=18, blur=16).add_to(layer_fut)
layer_fut.add_to(m)

layer_risk = folium.FeatureGroup(name="Combined Risk (Forecast × Traffic)")
HeatMap(heat_points(grid_df, "risk", MAX_HEATMAP_POINTS), radius=18, blur=16).add_to(layer_risk)
layer_risk.add_to(m)

if TRAFFIC_GRID_ENABLED:
    layer_tr = folium.FeatureGroup(name="Traffic Density Proxy (normalized)")
    HeatMap(heat_points(grid_df, "traffic_norm", MAX_HEATMAP_POINTS), radius=18, blur=16).add_to(layer_tr)
    layer_tr.add_to(m)

# Zone sampled points
layer_red = folium.FeatureGroup(name="RED Zones (sampled)")
if len(red_pts) > 0:
    for _, r in red_pts.sample(min(400, len(red_pts)), random_state=RANDOM_SEED).iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=2, popup=f"Risk: {r['risk']:.1f}",
                            fill=True, fill_opacity=0.7).add_to(layer_red)
layer_red.add_to(m)

layer_green = folium.FeatureGroup(name="GREEN Zones (sampled)")
if len(green_pts) > 0:
    for _, r in green_pts.sample(min(400, len(green_pts)), random_state=RANDOM_SEED).iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=2, popup=f"Risk: {r['risk']:.1f}",
                            fill=True, fill_opacity=0.4).add_to(layer_green)
layer_green.add_to(m)

# Routes
if route_fast_coords:
    folium.PolyLine(route_fast_coords, weight=6, opacity=0.8, tooltip="Fastest route").add_to(m)
if route_clean_coords:
    folium.PolyLine(route_clean_coords, weight=6, opacity=0.8, tooltip="Green corridor route").add_to(m)

folium.Marker(list(ORIGIN), tooltip="Origin").add_to(m)
folium.Marker(list(DEST), tooltip="Destination").add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Save artifacts
safe_city = CITY_NAME.lower().replace(" ", "_")
out_map = f"{safe_city}_nyx_fullppt.html"
model_path = f"{safe_city}_nyx_model.joblib"
grid_path  = f"{safe_city}_nyx_grid.parquet"
meta_path  = f"{safe_city}_nyx_metadata.json"
alerts_path= f"{safe_city}_nyx_alerts.json"

m.save(out_map)
joblib.dump(model, model_path)
grid_df.to_parquet(grid_path, index=False)
json.dump(alerts, open(alerts_path, "w"), indent=2)

meta = {
    "project": PROJECT_NAME,
    "city": CITY_NAME,
    "center": CITY_CENTER,
    "mode": RUN_MODE,
    "start_date_utc": START_DATE_UTC,
    "end_date_utc": END_DATE_UTC,
    "map_time_utc": str(map_time),
    "horizon_h": HORIZON_H,
    "sensor_radius_used_km": final_radius_used,
    "map_radius_used_km": map_radius,
    "grid_step_deg": GRID_STEP_DEG,
    "sensors_used": int(len(now_points)),
    "features": feature_cols,
    "baseline_metrics": baseline_metrics,
    "cv_metrics_mean": cv_mean,
    "thresholds": {"green_max": GREEN_MAX, "red_min": RED_MIN},
    "route_metrics": route_metrics,
    "compliance_metrics": compliance_metrics,
    "alerts": alerts,
    "notes": [
        "Traffic density is a proxy from OSM road class + time-of-day (historical pattern surrogate).",
        "For production: replace proxy with real traffic API or city open traffic feeds."
    ]
}
json.dump(meta, open(meta_path, "w"), indent=2)

# FastAPI backend file
api_code = f'''\
# nyx_api.py — FastAPI backend for ECONyx prototype
# Run locally:
#   pip install fastapi uvicorn pandas numpy joblib pyarrow
#   uvicorn nyx_api:app --reload --port 8000
#
# Endpoints:
#   GET  /health
#   POST /predict_point  {{lat, lon}}
#   POST /classify_zone  {{lat, lon}}
#   GET  /metrics
#   GET  /alerts

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd, joblib, json

MODEL_PATH = "{model_path}"
GRID_PATH  = "{grid_path}"
META_PATH  = "{meta_path}"
ALERTS_PATH= "{alerts_path}"

app = FastAPI(title="Nyx ECONyx API")

model = joblib.load(MODEL_PATH)
grid = pd.read_parquet(GRID_PATH)
meta = json.load(open(META_PATH, "r"))
alerts = json.load(open(ALERTS_PATH, "r"))

GREEN_MAX = float(meta["thresholds"]["green_max"])
RED_MIN   = float(meta["thresholds"]["red_min"])

class PointReq(BaseModel):
    lat: float
    lon: float

def classify_zone(val: float) -> str:
    if val >= RED_MIN: return "RED"
    if val <= GREEN_MAX: return "GREEN"
    return "YELLOW"

def nearest_grid_value(lat, lon, col="risk"):
    dlat = (grid["lat"] - lat).abs()
    dlon = (grid["lon"] - lon).abs()
    idx = (dlat + dlon).idxmin()
    return float(grid.loc[idx, col])

@app.get("/health")
def health():
    return {{"status":"ok"}}

@app.post("/predict_point")
def predict_point(req: PointReq):
    return {{
        "lat": req.lat,
        "lon": req.lon,
        "pm25_now": nearest_grid_value(req.lat, req.lon, "pm25_now"),
        "pm25_fut": nearest_grid_value(req.lat, req.lon, "pm25_fut"),
        "traffic_norm": nearest_grid_value(req.lat, req.lon, "traffic_norm"),
        "risk": nearest_grid_value(req.lat, req.lon, "risk")
    }}

@app.post("/classify_zone")
def classify(req: PointReq):
    risk = nearest_grid_value(req.lat, req.lon, "risk")
    return {{"risk": risk, "zone": classify_zone(risk)}}

@app.get("/metrics")
def metrics():
    return meta

@app.get("/alerts")
def get_alerts():
    return alerts
'''
open("nyx_api.py", "w").write(api_code)

print("\n DONE (FULL PPT) — 8/8 completed")
print("Artifacts saved:")
print(" - Map HTML:", out_map)
print(" - Model:", model_path)
print(" - Grid:", grid_path)
print(" - Alerts:", alerts_path)
print(" - Metadata:", meta_path)
print(" - Backend file:", "nyx_api.py")


if PREVIEW_MAP_INLINE:
    display(m)