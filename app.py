import io
import os
import zipfile
import tempfile
from pathlib import Path

import requests
import numpy as np
import streamlit as st
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator


# ============================================================
# Streamlit page
# ============================================================

st.set_page_config(
    page_title="Study Area Map Generator",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Study Area Map Generator")
st.caption(
    "Three-panel study area map: India inset, state/district inset, and main study area/DEM map."
)


# ============================================================
# Boundary source
# ============================================================

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master"
INDIA_STATES_URL = f"{GITHUB_RAW_BASE}/INDIA/INDIA_STATES.geojson"
INDIA_DISTRICTS_URL = f"{GITHUB_RAW_BASE}/INDIA/INDIA_DISTRICTS.geojson"

STATE_COLUMN_CANDIDATES = [
    "ST_NM", "st_nm", "STNAME", "stname", "STATE", "State", "state",
    "STATE_NAME", "State_Name", "state_name", "statename", "StateName",
    "ST_NAME", "st_name", "NAME_1", "Name", "NAME", "name",
    "ADM1_NAME", "adm1_name"
]

DISTRICT_COLUMN_CANDIDATES = [
    "DISTRICT", "District", "district", "DIST_NAME", "Dist_Name", "dist_name",
    "DISTNAME", "distname", "dtname", "DTNAME", "DT_NAME", "dt_name",
    "NAME_2", "Name", "NAME", "name", "ADM2_NAME", "adm2_name"
]

KNOWN_STATES = [
    "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH",
    "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JHARKHAND",
    "KARNATAKA", "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR",
    "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "ORISSA", "PUNJAB",
    "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TELANGANA", "TRIPURA",
    "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "DELHI", "NCT OF DELHI",
    "JAMMU AND KASHMIR", "JAMMU & KASHMIR", "LADAKH", "PUDUCHERRY",
    "CHANDIGARH", "LAKSHADWEEP", "ANDAMAN AND NICOBAR", "ANDAMAN & NICOBAR",
    "DADRA AND NAGAR HAVELI", "DAMAN AND DIU"
]


# ============================================================
# Utility functions
# ============================================================

def normalize_text(value):
    text = str(value).strip().upper()
    text = text.replace("&", "AND")
    text = text.replace(".", "")
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text


def find_column(gdf, candidates):
    lower_lookup = {c.lower(): c for c in gdf.columns}

    for col in candidates:
        if col in gdf.columns:
            return col
        if col.lower() in lower_lookup:
            return lower_lookup[col.lower()]

    return None


def detect_state_column(gdf):
    col = find_column(gdf, STATE_COLUMN_CANDIDATES)
    if col:
        return col

    best_col = None
    best_score = 0
    known = set(normalize_text(x) for x in KNOWN_STATES)

    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        try:
            vals = gdf[col].dropna().astype(str).map(normalize_text).unique().tolist()
            score = sum(1 for v in vals if v in known)
            if score > best_score:
                best_score = score
                best_col = col
        except Exception:
            pass

    return best_col if best_score > 0 else None


def detect_district_column(gdf):
    col = find_column(gdf, DISTRICT_COLUMN_CANDIDATES)
    if col:
        return col

    for col in gdf.columns:
        if col != gdf.geometry.name:
            try:
                if gdf[col].dtype == "object":
                    return col
            except Exception:
                pass

    return None


def safe_union(gdf):
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def clean_geometries(gdf):
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass
    return gdf


@st.cache_data(show_spinner=True)
def load_geojson_from_url(url):
    response = requests.get(url, headers={"User-Agent": "study-area-map-generator"}, timeout=120)
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
    try:
        tmp.write(response.content)
        tmp.close()
        gdf = gpd.read_file(tmp.name)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
        return clean_geometries(gdf)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def save_uploaded_file(uploaded_file, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / uploaded_file.name.replace(" ", "_")
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def read_uploaded_vector(uploaded_file, out_dir, default_crs="EPSG:4326"):
    saved_path = save_uploaded_file(uploaded_file, out_dir)
    suffix = saved_path.suffix.lower()

    if suffix == ".zip":
        extract_dir = Path(out_dir) / saved_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(saved_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        shp_files = sorted(extract_dir.rglob("*.shp"))
        if not shp_files:
            raise ValueError("No .shp file found inside uploaded ZIP.")
        gdf = gpd.read_file(shp_files[0])
    elif suffix in [".geojson", ".json", ".gpkg"]:
        gdf = gpd.read_file(saved_path)
    else:
        raise ValueError("Upload ZIP shapefile, GeoJSON, JSON, or GPKG.")

    if gdf.empty:
        raise ValueError("Uploaded vector file is empty.")

    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs)

    gdf = gdf.to_crs("EPSG:4326")
    return clean_geometries(gdf)


def read_uploaded_raster(uploaded_file, out_dir):
    if uploaded_file is None:
        return None
    return save_uploaded_file(uploaded_file, out_dir)


def prepare_state_layer(states_gdf, districts_gdf):
    state_col = detect_state_column(states_gdf)
    district_state_col = detect_state_column(districts_gdf)

    if state_col is not None:
        return states_gdf, state_col, districts_gdf, district_state_col

    if district_state_col is None:
        raise ValueError("Could not detect a state-name column in the India states or districts layers.")

    dissolved = districts_gdf.dissolve(by=district_state_col, as_index=False)
    dissolved = dissolved.to_crs("EPSG:4326")
    dissolved = clean_geometries(dissolved)
    return dissolved, district_state_col, districts_gdf, district_state_col


def filter_state(states_gdf, state_col, selected_state):
    target = normalize_text(selected_state)
    series = states_gdf[state_col].astype(str).map(normalize_text)

    out = states_gdf[series == target].copy()
    if not out.empty:
        return out

    out = states_gdf[series.str.contains(target, regex=False, na=False)].copy()
    if not out.empty:
        return out

    return states_gdf[series.map(lambda x: target in x or x in target)].copy()


def filter_districts_for_state(districts_gdf, district_state_col, selected_state_gdf, selected_state_name):
    if district_state_col is not None:
        target = normalize_text(selected_state_name)
        series = districts_gdf[district_state_col].astype(str).map(normalize_text)
        out = districts_gdf[series == target].copy()
        if not out.empty:
            return out
        out = districts_gdf[series.str.contains(target, regex=False, na=False)].copy()
        if not out.empty:
            return out

    geom = safe_union(selected_state_gdf)
    out = districts_gdf[districts_gdf.geometry.intersects(geom)].copy()
    return out if not out.empty else districts_gdf.copy()


def filter_by_attribute(gdf, column, value):
    if column == "None" or value == "All":
        return gdf.copy()
    return gdf[gdf[column].astype(str) == str(value)].copy()


# ============================================================
# Cartographic functions
# ============================================================

def padded_bounds(bounds, pad=0.15):
    minx, miny, maxx, maxy = bounds
    dx = maxx - minx
    dy = maxy - miny
    if dx == 0:
        dx = 0.1
    if dy == 0:
        dy = 0.1
    return [minx - dx * pad, miny - dy * pad, maxx + dx * pad, maxy + dy * pad]


def set_extent(ax, gdf, pad=0.15):
    minx, miny, maxx, maxy = padded_bounds(gdf.total_bounds, pad)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)


def format_dms(value, is_lon=True):
    hemi = "E" if is_lon and value >= 0 else "W" if is_lon else "N" if value >= 0 else "S"
    value = abs(float(value))
    deg = int(np.floor(value))
    minute_float = (value - deg) * 60
    minute = int(np.floor(minute_float))
    second = int(round((minute_float - minute) * 60))
    if second == 60:
        second = 0
        minute += 1
    if minute == 60:
        minute = 0
        deg += 1
    return f"{deg}°{minute:02d}'{second:02d}\"{hemi}"


def apply_degree_grid(ax, fontsize=7, xbins=4, ybins=4):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_dms(x, True)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: format_dms(y, False)))
    ax.tick_params(
        axis="both",
        labelsize=fontsize,
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=True,
        labelbottom=True,
        labelleft=True,
        labelright=True
    )
    for label in ax.get_yticklabels():
        label.set_rotation(90)
    ax.grid(False)


def add_map_border(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")


def add_title(ax, title, loc="upper_left", fontsize=12):
    positions = {
        "upper_left": (0.04, 0.96, "left", "top"),
        "upper_right": (0.96, 0.96, "right", "top"),
        "lower_left": (0.04, 0.18, "left", "bottom"),
        "lower_right": (0.96, 0.18, "right", "bottom"),
        "top_center": (0.50, 0.96, "center", "top")
    }
    x, y, ha, va = positions.get(loc, positions["upper_left"])
    ax.text(
        x, y, title,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        zorder=100,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=2.0)
    )


def add_compass(ax, loc="upper_right", fontsize=10):
    """
    Add a simple north arrow.
    The function name is kept as add_compass so the rest of the app remains compatible.
    """
    positions = {
        "upper_right": (0.88, 0.78),
        "upper_left": (0.14, 0.78),
        "lower_right": (0.88, 0.25),
        "lower_left": (0.14, 0.25),
    }
    x, y = positions.get(loc, positions["upper_right"])

    # North label
    ax.text(
        x, y + 0.115, "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize + 1,
        fontweight="bold",
        zorder=120
    )

    # Arrow body
    ax.annotate(
        "",
        xy=(x, y + 0.085),
        xytext=(x, y - 0.075),
        xycoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.8,
            color="black",
            mutation_scale=18,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=119
    )


def choose_nice_scale(width_km):
    target = width_km / 5
    for v in [1, 2, 5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000]:
        if v >= target:
            return v
    return 5000


def add_scale_bar(ax, loc="lower_left", segments=4, fontsize=7):
    locs = {
        "lower_left": (0.08, 0.08),
        "lower_right": (0.56, 0.08),
        "upper_left": (0.08, 0.82),
        "upper_right": (0.56, 0.82),
    }
    x_frac, y_frac = locs.get(loc, locs["lower_left"])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width_deg = xlim[1] - xlim[0]
    height_deg = ylim[1] - ylim[0]
    center_lat = (ylim[0] + ylim[1]) / 2
    width_km = abs(width_deg * 111.32 * np.cos(np.deg2rad(center_lat)))
    length_km = choose_nice_scale(width_km)
    denom = 111.32 * np.cos(np.deg2rad(center_lat))
    if abs(denom) < 1e-6:
        denom = 111.32
    length_deg = length_km / denom
    x0 = xlim[0] + x_frac * width_deg
    y0 = ylim[0] + y_frac * height_deg
    segment_deg = length_deg / segments
    bar_height = height_deg * 0.014

    for i in range(segments):
        face = "black" if i % 2 == 0 else "white"
        rect = Rectangle(
            (x0 + i * segment_deg, y0), segment_deg, bar_height,
            facecolor=face, edgecolor="black", linewidth=0.5, zorder=100
        )
        ax.add_patch(rect)

    ax.text(x0, y0 + bar_height * 1.8, "0", ha="center", va="bottom", fontsize=fontsize, zorder=101)
    ax.text(x0 + length_deg, y0 + bar_height * 1.8, f"{int(length_km)}", ha="center", va="bottom", fontsize=fontsize, zorder=101)
    ax.text(x0 + length_deg / 2, y0 - bar_height * 1.9, "Kilometers", ha="center", va="top", fontsize=fontsize, zorder=101)


def add_dem_legend(fig, ax, im, vmin, vmax, loc="middle_right"):
    locs = {
        "middle_right": [0.76, 0.38, 0.055, 0.23],
        "upper_right": [0.76, 0.64, 0.055, 0.23],
        "lower_right": [0.76, 0.14, 0.055, 0.23],
        "middle_left": [0.12, 0.38, 0.055, 0.23],
    }
    cax = ax.inset_axes(locs.get(loc, locs["middle_right"]))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([vmax, vmin])
    cbar.set_ticklabels([f"High : {int(round(vmax))}", f"Low : {int(round(vmin))}"])
    cbar.ax.tick_params(labelsize=8)
    cax.set_title("DEM", fontsize=10, fontweight="bold", pad=5)


# ============================================================
# Raster functions
# ============================================================

def clip_dem_to_study_area_wgs84(raster_path, study_gdf_wgs84):
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Uploaded raster has no CRS. Please upload a georeferenced GeoTIFF.")

        with WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
            study_for_raster = study_gdf_wgs84.to_crs(vrt.crs)
            shapes = [geom for geom in study_for_raster.geometry if geom is not None and not geom.is_empty]
            clipped, transform = mask(vrt, shapes, crop=True, filled=True)
            arr = clipped[0].astype(float)
            if vrt.nodata is not None:
                arr[arr == vrt.nodata] = np.nan
            arr[arr <= -9999] = np.nan
            if np.all(np.isnan(arr)):
                raise ValueError("DEM clipping produced empty raster. Check DEM and study-area overlap.")
            extent = plotting_extent(arr, transform)
            return arr, extent


def plot_dem(ax, raster_path, study_gdf_wgs84, cmap="plasma"):
    arr, extent = clip_dem_to_study_area_wgs84(raster_path, study_gdf_wgs84)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax), zorder=1)
    return im, vmin, vmax


# ============================================================
# Load administrative layers
# ============================================================

try:
    raw_states_gdf = load_geojson_from_url(INDIA_STATES_URL)
    raw_districts_gdf = load_geojson_from_url(INDIA_DISTRICTS_URL)
    states_gdf, state_col, districts_gdf, district_state_col = prepare_state_layer(raw_states_gdf, raw_districts_gdf)
except Exception as e:
    st.error("Could not load India/state/district boundaries.")
    st.exception(e)
    st.stop()

state_names = states_gdf[state_col].dropna().astype(str).sort_values().unique().tolist()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("1. Administrative Area")

state_names_norm = [normalize_text(x) for x in state_names]
default_state = "MAHARASHTRA"
default_idx = state_names_norm.index(default_state) if default_state in state_names_norm else 0

selected_state_name = st.sidebar.selectbox("Select State / UT", state_names, index=default_idx)
selected_state_gdf = filter_state(states_gdf, state_col, selected_state_name)

india_panel_mode = st.sidebar.selectbox(
    "India inset",
    ["Complete India with selected state", "Selected state only"],
    index=0
)

state_panel_mode = st.sidebar.selectbox(
    "State inset",
    ["Complete selected state with districts", "Selected district only", "Selected state with study area"],
    index=0
)

state_districts_gdf = filter_districts_for_state(districts_gdf, district_state_col, selected_state_gdf, selected_state_name)
district_col = detect_district_column(state_districts_gdf)
selected_district_gdf = None
selected_district_name = "None"

if district_col:
    district_names = state_districts_gdf[district_col].dropna().astype(str).sort_values().unique().tolist()
    selected_district_name = st.sidebar.selectbox("Optional: district to highlight", ["None"] + district_names)
    if selected_district_name != "None":
        selected_district_gdf = state_districts_gdf[state_districts_gdf[district_col].astype(str) == selected_district_name].copy()

st.sidebar.header("2. Upload Study Area")

study_file = st.sidebar.file_uploader(
    "Upload study area / catchment file",
    type=["zip", "geojson", "json", "gpkg"]
)

default_crs = st.sidebar.text_input("Default CRS if missing", value="EPSG:4326")

dem_file = st.sidebar.file_uploader("Optional DEM GeoTIFF", type=["tif", "tiff"])

st.sidebar.header("3. Titles")
country_title = st.sidebar.text_input("India panel title", value="India")
state_title = st.sidebar.text_input("State panel title", value=str(selected_state_name).upper())
main_title = st.sidebar.text_input("Main panel title", value="Study Area")

st.sidebar.header("4. Furniture Positions")
india_title_loc = st.sidebar.selectbox("India title position", ["upper_left", "upper_right", "lower_left", "lower_right", "top_center"], index=0)
state_title_loc = st.sidebar.selectbox("State title position", ["upper_left", "upper_right", "lower_left", "lower_right", "top_center"], index=0)
main_title_loc = st.sidebar.selectbox("Main title position", ["upper_left", "upper_right", "lower_left", "lower_right", "top_center"], index=0)

india_compass_loc = st.sidebar.selectbox("India north arrow position", ["upper_right", "upper_left", "lower_right", "lower_left"], index=0)
state_compass_loc = st.sidebar.selectbox("State north arrow position", ["upper_right", "upper_left", "lower_right", "lower_left"], index=0)
main_compass_loc = st.sidebar.selectbox("Main north arrow position", ["upper_right", "upper_left", "lower_right", "lower_left"], index=0)

main_legend_loc = st.sidebar.selectbox("DEM legend position", ["middle_right", "upper_right", "lower_right", "middle_left"], index=0)

st.sidebar.header("5. Style")
country_fill = st.sidebar.color_picker("India map fill", "#dff2cf")
state_highlight_color = st.sidebar.color_picker("Highlighted state", "#ff5a1f")
state_fill = st.sidebar.color_picker("State/district fill", "#c8b2f0")
district_highlight_color = st.sidebar.color_picker("District/study highlight", "#ffd43b")
study_boundary_color = st.sidebar.color_picker("Study boundary", "#000000")
study_fill_color = st.sidebar.color_picker("Study fill without DEM", "#1f78ff")
dem_cmap = st.sidebar.selectbox("DEM color palette", ["plasma", "terrain", "viridis", "turbo", "rainbow", "Spectral_r"], index=0)
output_dpi = st.sidebar.selectbox("Output DPI", [150, 200, 300, 400, 600], index=2)

st.sidebar.header("6. Padding")
india_padding = st.sidebar.slider("India padding", 0.04, 0.30, 0.10, 0.01)
state_padding = st.sidebar.slider("State padding", 0.05, 0.35, 0.18, 0.01)
main_padding = st.sidebar.slider("Main map padding", 0.05, 0.35, 0.20, 0.01)

show_main_district_background = st.sidebar.checkbox("Show district boundaries behind main map", value=False)
show_main_legend_without_dem = st.sidebar.checkbox("Show legend without DEM", value=True)


# ============================================================
# Wait for upload
# ============================================================

if study_file is None:
    st.info("Upload your study area/catchment file. The India and state/district layers are loaded automatically. The extent indicator/connector lines have been removed.")
    with st.expander("Detected fields"):
        st.write("State column:", state_col)
        st.write("District-state column:", district_state_col)
        st.write("District column:", district_col)
    st.stop()


# ============================================================
# Read uploaded files
# ============================================================

if "study_map_workdir" not in st.session_state:
    st.session_state["study_map_workdir"] = tempfile.mkdtemp(prefix="study_area_map_")

workdir = Path(st.session_state["study_map_workdir"])

try:
    study_gdf = read_uploaded_vector(study_file, workdir / "study", default_crs)
    dem_path = read_uploaded_raster(dem_file, workdir / "dem")
except Exception as e:
    st.error("Could not read uploaded file.")
    st.exception(e)
    st.stop()


# ============================================================
# Optional feature filter
# ============================================================

st.subheader("Study Area Feature Selection")

study_columns = [c for c in study_gdf.columns if c != study_gdf.geometry.name]
c1, c2, c3 = st.columns(3)

with c1:
    study_filter_col = st.selectbox("Filter study area by attribute", ["None"] + study_columns)

study_selected_gdf = study_gdf.copy()

with c2:
    if study_filter_col != "None":
        values = study_gdf[study_filter_col].dropna().astype(str).sort_values().unique().tolist()
        selected_value = st.selectbox("Select feature", ["All"] + values)
        study_selected_gdf = filter_by_attribute(study_gdf, study_filter_col, selected_value)

with c3:
    st.metric("Selected features", len(study_selected_gdf))

if study_selected_gdf.empty:
    st.error("Selected study area is empty.")
    st.stop()


# ============================================================
# Create figure
# ============================================================

try:
    fig = plt.figure(figsize=(13, 8), dpi=output_dpi)

    # More compact layout. No extent indicator / connector lines.
    # Values are [left, bottom, width, height] in figure-fraction units.
    ax_india = fig.add_axes([0.035, 0.525, 0.365, 0.415])
    ax_state = fig.add_axes([0.035, 0.065, 0.365, 0.415])
    ax_main = fig.add_axes([0.405, 0.065, 0.565, 0.875])

    # India panel
    if india_panel_mode == "Complete India with selected state":
        states_gdf.plot(ax=ax_india, color=country_fill, edgecolor="black", linewidth=0.35, zorder=1)
        selected_state_gdf.plot(ax=ax_india, color=state_highlight_color, edgecolor="black", linewidth=0.65, zorder=3)
        set_extent(ax_india, states_gdf, india_padding)
    else:
        selected_state_gdf.plot(ax=ax_india, color=state_highlight_color, edgecolor="black", linewidth=0.65, zorder=3)
        set_extent(ax_india, selected_state_gdf, india_padding)

    add_title(ax_india, country_title, india_title_loc, fontsize=12)
    add_compass(ax_india, india_compass_loc, fontsize=10)
    add_scale_bar(ax_india, "lower_left", fontsize=7)
    apply_degree_grid(ax_india, fontsize=7, xbins=4, ybins=4)
    add_map_border(ax_india)

    # State panel
    if state_panel_mode == "Complete selected state with districts":
        state_districts_gdf.plot(ax=ax_state, color=state_fill, edgecolor="black", linewidth=0.35, zorder=1)
        selected_state_gdf.boundary.plot(ax=ax_state, color="black", linewidth=0.8, zorder=2)
        if selected_district_gdf is not None and not selected_district_gdf.empty:
            selected_district_gdf.plot(ax=ax_state, color=district_highlight_color, edgecolor="black", linewidth=0.6, zorder=3)
        study_selected_gdf.plot(ax=ax_state, color=district_highlight_color, edgecolor="black", linewidth=0.7, alpha=0.9, zorder=4)
        set_extent(ax_state, selected_state_gdf, state_padding)
    elif state_panel_mode == "Selected district only" and selected_district_gdf is not None and not selected_district_gdf.empty:
        selected_district_gdf.plot(ax=ax_state, color=state_fill, edgecolor="black", linewidth=0.7, zorder=1)
        study_selected_gdf.plot(ax=ax_state, color=district_highlight_color, edgecolor="black", linewidth=0.7, alpha=0.9, zorder=3)
        set_extent(ax_state, selected_district_gdf, state_padding)
    else:
        selected_state_gdf.plot(ax=ax_state, color=state_fill, edgecolor="black", linewidth=0.8, zorder=1)
        study_selected_gdf.plot(ax=ax_state, color=district_highlight_color, edgecolor="black", linewidth=0.7, alpha=0.9, zorder=3)
        set_extent(ax_state, selected_state_gdf, state_padding)

    add_title(ax_state, state_title, state_title_loc, fontsize=12)
    add_compass(ax_state, state_compass_loc, fontsize=10)
    add_scale_bar(ax_state, "lower_left", fontsize=7)
    apply_degree_grid(ax_state, fontsize=7, xbins=4, ybins=4)
    add_map_border(ax_state)

    # Main panel
    dem_plotted = False
    im = None
    dem_min = None
    dem_max = None

    if show_main_district_background:
        try:
            state_districts_gdf.boundary.plot(ax=ax_main, color="gray", linewidth=0.3, alpha=0.6, zorder=0)
        except Exception:
            pass

    if dem_path is not None:
        try:
            im, dem_min, dem_max = plot_dem(ax_main, dem_path, study_selected_gdf, cmap=dem_cmap)
            study_selected_gdf.boundary.plot(ax=ax_main, color=study_boundary_color, linewidth=0.8, zorder=5)
            dem_plotted = True
        except Exception as e:
            st.warning(f"DEM could not be plotted. Boundary map will be shown instead. Error: {e}")

    if not dem_plotted:
        study_selected_gdf.plot(ax=ax_main, color=study_fill_color, edgecolor=study_boundary_color, linewidth=0.8, alpha=0.9, zorder=3)
        if show_main_legend_without_dem:
            patch = Patch(facecolor=study_fill_color, edgecolor=study_boundary_color, label="Study Area")
            ax_main.legend(handles=[patch], loc="lower right", fontsize=9, frameon=True, framealpha=0.95)

    set_extent(ax_main, study_selected_gdf, main_padding)
    add_title(ax_main, main_title, main_title_loc, fontsize=12)
    add_compass(ax_main, main_compass_loc, fontsize=11)
    add_scale_bar(ax_main, "lower_left", fontsize=7)

    if dem_plotted and im is not None:
        add_dem_legend(fig, ax_main, im, dem_min, dem_max, loc=main_legend_loc)

    apply_degree_grid(ax_main, fontsize=8, xbins=4, ybins=7)
    add_map_border(ax_main)

    # Export
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=output_dpi, bbox_inches="tight", facecolor="white")
    png_buffer.seek(0)

    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight", facecolor="white")
    pdf_buffer.seek(0)

    st.subheader("Generated Study Area Map")
    st.image(png_buffer.getvalue(), caption="Generated study area map", use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download PNG", png_buffer.getvalue(), "study_area_map.png", "image/png")
    with d2:
        st.download_button("Download PDF", pdf_buffer.getvalue(), "study_area_map.pdf", "application/pdf")

    plt.close(fig)

except Exception as e:
    st.error("Map generation failed. The detailed error is shown below.")
    st.exception(e)


# ============================================================
# Info
# ============================================================

with st.expander("Layer information"):
    st.write("Detected state column:", state_col)
    st.write("Detected district-state column:", district_state_col)
    st.write("Detected district column:", district_col)
    st.write("Selected state:", selected_state_name)
    st.write("Study area CRS:", study_selected_gdf.crs)

st.markdown("---")
st.markdown(
    "**Note:** The extent indicator / connector lines have been removed. If furniture overlaps any boundary, adjust the position dropdowns and increase the padding sliders."
)
