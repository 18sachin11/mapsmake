import os
import io
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

from matplotlib.patches import Rectangle, Patch, ConnectionPatch
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="Study Area Map Generator",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Study Area Map Generator")
st.caption(
    "Generate publication-style study area maps with India inset, state/district inset, main study area map, DEM, north arrow, scale bar and legends."
)


# ============================================================
# GitHub boundary URLs
# ============================================================

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master"

INDIA_STATES_URL = f"{GITHUB_RAW_BASE}/INDIA/INDIA_STATES.geojson"
INDIA_DISTRICTS_URL = f"{GITHUB_RAW_BASE}/INDIA/INDIA_DISTRICTS.geojson"


# ============================================================
# Known Indian States / UTs
# ============================================================

KNOWN_STATES_UTS = [
    "Andaman & Nicobar",
    "Andaman and Nicobar",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu & Kashmir",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Ladakh",
    "Lakshadweep",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "NCT of Delhi",
    "Odisha",
    "Orissa",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal"
]


STATE_COLUMN_CANDIDATES = [
    "ST_NM", "st_nm",
    "STNAME", "stname",
    "STATE", "State", "state",
    "STATE_NAME", "State_Name", "state_name",
    "statename", "StateName",
    "ST_NAME", "st_name",
    "NAME_1", "Name", "NAME", "name",
    "ADM1_NAME", "adm1_name"
]

DISTRICT_COLUMN_CANDIDATES = [
    "DISTRICT", "District", "district",
    "DIST_NAME", "Dist_Name", "dist_name",
    "DISTNAME", "distname",
    "dtname", "DTNAME",
    "DT_NAME", "dt_name",
    "NAME_2", "Name", "NAME", "name",
    "ADM2_NAME", "adm2_name"
]


# ============================================================
# Text and column helpers
# ============================================================

def normalize_text(x):
    """
    Normalize names for flexible matching.
    """
    x = str(x).strip().upper()
    x = x.replace("&", "AND")
    x = x.replace(".", "")
    x = x.replace("-", " ")
    x = " ".join(x.split())
    return x


def find_first_existing_column(gdf, candidates):
    """
    Find direct or case-insensitive matching column.
    """
    if gdf is None or gdf.empty:
        return None

    lower_map = {c.lower(): c for c in gdf.columns}

    for col in candidates:
        if col in gdf.columns:
            return col

        if col.lower() in lower_map:
            return lower_map[col.lower()]

    return None


def detect_state_column(gdf):
    """
    Detect state-name column using candidates and known state names.
    """
    direct = find_first_existing_column(gdf, STATE_COLUMN_CANDIDATES)

    if direct is not None:
        return direct

    known_norm = set(normalize_text(x) for x in KNOWN_STATES_UTS)

    best_col = None
    best_score = 0

    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue

        try:
            values = (
                gdf[col]
                .dropna()
                .astype(str)
                .apply(normalize_text)
                .unique()
                .tolist()
            )

            score = sum(1 for v in values if v in known_norm)

            if score > best_score:
                best_score = score
                best_col = col

        except Exception:
            continue

    if best_score > 0:
        return best_col

    return None


def detect_district_column(gdf):
    """
    Detect district-name column.
    """
    direct = find_first_existing_column(gdf, DISTRICT_COLUMN_CANDIDATES)

    if direct is not None:
        return direct

    text_cols = []

    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue

        try:
            if gdf[col].dtype == "object":
                text_cols.append(col)
        except Exception:
            pass

    if text_cols:
        return text_cols[0]

    return None


# ============================================================
# Data loading functions
# ============================================================

@st.cache_data(show_spinner=True)
def load_geojson_from_url(url):
    """
    Download GeoJSON and read as GeoDataFrame.
    """
    headers = {
        "User-Agent": "study-area-map-generator"
    }

    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")

    try:
        tmp.write(response.content)
        tmp.close()

        gdf = gpd.read_file(tmp.name)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf[~gdf.geometry.isna()].copy()

        return gdf

    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def save_uploaded_file(uploaded_file, out_dir):
    """
    Save uploaded Streamlit file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = out_dir / safe_name

    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return out_path


def read_uploaded_vector(uploaded_file, out_dir, default_crs="EPSG:4326"):
    """
    Read uploaded vector file.
    Supports:
    - ZIP shapefile
    - GeoJSON
    - GPKG
    """
    saved_path = save_uploaded_file(uploaded_file, out_dir)
    suffix = saved_path.suffix.lower()

    if suffix == ".zip":
        extract_dir = Path(out_dir) / saved_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(saved_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        shp_files = sorted(list(extract_dir.rglob("*.shp")))

        if not shp_files:
            raise ValueError("No .shp file found inside uploaded ZIP.")

        gdf = gpd.read_file(shp_files[0])

    elif suffix in [".geojson", ".json", ".gpkg"]:
        gdf = gpd.read_file(saved_path)

    else:
        raise ValueError("Unsupported vector format. Upload ZIP shapefile, GeoJSON, JSON or GPKG.")

    if gdf.empty:
        raise ValueError("Uploaded vector file is empty.")

    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs)

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[~gdf.geometry.isna()].copy()

    return gdf


def read_uploaded_raster(uploaded_file, out_dir):
    """
    Save uploaded DEM GeoTIFF.
    """
    if uploaded_file is None:
        return None

    return save_uploaded_file(uploaded_file, out_dir)


# ============================================================
# Administrative boundary preparation
# ============================================================

def prepare_admin_layers(states_gdf, districts_gdf):
    """
    Prepare state and district layers robustly.

    If state names cannot be detected in the states layer, the code tries
    to dissolve the district layer by state column to create state polygons.
    """
    state_col = detect_state_column(states_gdf)

    if state_col is not None:
        prepared_states = states_gdf.copy()
        prepared_state_col = state_col
    else:
        district_state_col = detect_state_column(districts_gdf)

        if district_state_col is None:
            st.error("Could not detect state name column in both states and districts layers.")

            st.write("Columns in INDIA_STATES.geojson:")
            st.write(states_gdf.columns.tolist())

            st.write("Columns in INDIA_DISTRICTS.geojson:")
            st.write(districts_gdf.columns.tolist())

            st.stop()

        prepared_states = districts_gdf.dissolve(
            by=district_state_col,
            as_index=False
        )

        prepared_states = prepared_states.to_crs("EPSG:4326")
        prepared_state_col = district_state_col

    district_state_col = detect_state_column(districts_gdf)

    return prepared_states, prepared_state_col, districts_gdf, district_state_col


def get_state_names(states_gdf, state_col):
    """
    Extract state names for dropdown.
    """
    names = (
        states_gdf[state_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    return names


def filter_state(states_gdf, state_col, selected_state):
    """
    Select state polygon by name.
    """
    selected_norm = normalize_text(selected_state)
    state_series = states_gdf[state_col].astype(str).apply(normalize_text)

    exact = states_gdf[state_series == selected_norm].copy()

    if not exact.empty:
        return exact

    partial = states_gdf[
        state_series.str.contains(selected_norm, na=False)
    ].copy()

    if not partial.empty:
        return partial

    reverse = states_gdf[
        state_series.apply(lambda x: selected_norm in x or x in selected_norm)
    ].copy()

    return reverse


def filter_districts_for_state(districts_gdf, district_state_col, selected_state_gdf, selected_state_name):
    """
    Filter districts of selected state using attribute first, then spatial intersection.
    """
    if district_state_col is not None:
        selected_norm = normalize_text(selected_state_name)

        state_series = districts_gdf[district_state_col].astype(str).apply(normalize_text)

        exact = districts_gdf[state_series == selected_norm].copy()

        if not exact.empty:
            return exact

        partial = districts_gdf[
            state_series.str.contains(selected_norm, na=False)
        ].copy()

        if not partial.empty:
            return partial

    if selected_state_gdf is not None and not selected_state_gdf.empty:
        state_geom = selected_state_gdf.geometry.union_all()

        spatial = districts_gdf[
            districts_gdf.geometry.intersects(state_geom)
        ].copy()

        if not spatial.empty:
            return spatial

    return districts_gdf.copy()


def filter_by_attribute(gdf, column, value):
    """
    Filter GeoDataFrame by selected attribute.
    """
    if column == "None" or value == "All":
        return gdf.copy()

    return gdf[gdf[column].astype(str) == str(value)].copy()


# ============================================================
# Cartographic helper functions
# ============================================================

def padded_bounds(bounds, pad=0.08):
    """
    Add padding to bounds.
    """
    minx, miny, maxx, maxy = bounds

    dx = maxx - minx
    dy = maxy - miny

    if dx == 0:
        dx = 0.1

    if dy == 0:
        dy = 0.1

    return [
        minx - dx * pad,
        miny - dy * pad,
        maxx + dx * pad,
        maxy + dy * pad
    ]


def set_extent(ax, gdf, pad=0.08):
    """
    Set map extent using GeoDataFrame.
    """
    minx, miny, maxx, maxy = padded_bounds(gdf.total_bounds, pad=pad)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)


def format_dms(value, is_lon=True):
    """
    Decimal degree to DMS label.
    """
    if is_lon:
        hemi = "E" if value >= 0 else "W"
    else:
        hemi = "N" if value >= 0 else "S"

    value = abs(float(value))

    degree = int(np.floor(value))
    minute_float = (value - degree) * 60
    minute = int(np.floor(minute_float))
    second = int(round((minute_float - minute) * 60))

    if second == 60:
        second = 0
        minute += 1

    if minute == 60:
        minute = 0
        degree += 1

    return f"{degree}°{minute:02d}'{second:02d}\"{hemi}"


def apply_degree_grid(ax, fontsize=7, xbins=4, ybins=4):
    """
    Add degree grid labels on all sides.
    """
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: format_dms(x, is_lon=True))
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: format_dms(y, is_lon=False))
    )

    ax.tick_params(
        axis="both",
        which="major",
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


def add_north_arrow(ax, x=0.90, y=0.78, size=0.11):
    """
    Add north arrow with N, S, E and W.
    """
    ax.annotate(
        "N",
        xy=(x, y + size),
        xytext=(x, y),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            edgecolor="black",
            width=3,
            headwidth=10,
            headlength=12
        ),
        zorder=50
    )

    ax.text(
        x,
        y - 0.055,
        "S",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        zorder=50
    )

    ax.text(
        x - 0.065,
        y + 0.035,
        "W",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        zorder=50
    )

    ax.text(
        x + 0.065,
        y + 0.035,
        "E",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        zorder=50
    )


def km_to_degree_lon(km, latitude):
    """
    Approximate km to longitude degrees at given latitude.
    """
    denom = 111.32 * np.cos(np.deg2rad(latitude))

    if abs(denom) < 1e-6:
        denom = 111.32

    return km / denom


def choose_nice_scale_length(width_km):
    """
    Choose a cartographically clean scale length.
    """
    target = width_km / 5

    nice_values = [
        1, 2, 5,
        10, 20, 25, 50, 75,
        100, 150, 200, 250, 500, 750,
        1000, 1500, 2000, 2500, 3000, 3500
    ]

    for value in nice_values:
        if value >= target:
            return value

    return 5000


def add_scale_bar_degree(ax, location=(0.08, 0.065), segments=4, fontsize=7):
    """
    Add approximate scale bar for geographic coordinate maps.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width_deg = xlim[1] - xlim[0]
    height_deg = ylim[1] - ylim[0]

    center_lat = (ylim[0] + ylim[1]) / 2

    width_km = abs(width_deg * 111.32 * np.cos(np.deg2rad(center_lat)))

    length_km = choose_nice_scale_length(width_km)
    length_deg = km_to_degree_lon(length_km, center_lat)

    x0 = xlim[0] + location[0] * width_deg
    y0 = ylim[0] + location[1] * height_deg

    segment_deg = length_deg / segments
    bar_height = height_deg * 0.014

    for i in range(segments):
        face = "black" if i % 2 == 0 else "white"

        rect = Rectangle(
            (x0 + i * segment_deg, y0),
            segment_deg,
            bar_height,
            facecolor=face,
            edgecolor="black",
            linewidth=0.5,
            zorder=40
        )

        ax.add_patch(rect)

    ax.text(
        x0,
        y0 + bar_height * 1.8,
        "0",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=41
    )

    ax.text(
        x0 + length_deg,
        y0 + bar_height * 1.8,
        f"{int(length_km)}",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=41
    )

    ax.text(
        x0 + length_deg / 2,
        y0 - bar_height * 1.9,
        "Kilometers",
        ha="center",
        va="top",
        fontsize=fontsize,
        zorder=41
    )


def add_panel_title(ax, title, fontsize=14):
    """
    Add map title inside panel.
    """
    ax.text(
        0.05,
        0.94,
        title,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
        zorder=60
    )


def add_map_border(ax, linewidth=1.0):
    """
    Add black frame around map panel.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor("black")


# ============================================================
# Raster helper functions
# ============================================================

def clip_dem_to_study_area_wgs84(raster_path, study_gdf_wgs84):
    """
    Reproject DEM to EPSG:4326 and clip to study area.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Uploaded raster has no CRS. Please upload a georeferenced GeoTIFF.")

        with WarpedVRT(
            src,
            crs="EPSG:4326",
            resampling=Resampling.bilinear
        ) as vrt:

            study_for_raster = study_gdf_wgs84.to_crs(vrt.crs)

            shapes = [
                geom for geom in study_for_raster.geometry
                if geom is not None and not geom.is_empty
            ]

            clipped, transform = mask(
                vrt,
                shapes,
                crop=True,
                filled=True
            )

            arr = clipped[0].astype(float)

            if vrt.nodata is not None:
                arr[arr == vrt.nodata] = np.nan

            arr[arr <= -9999] = np.nan

            if np.all(np.isnan(arr)):
                raise ValueError("DEM clipping produced empty raster. Check overlap between DEM and study area.")

            extent = plotting_extent(arr, transform)

            return arr, extent


def plot_dem(ax, raster_path, study_gdf_wgs84, cmap="plasma"):
    """
    Plot clipped DEM.
    """
    arr, extent = clip_dem_to_study_area_wgs84(
        raster_path,
        study_gdf_wgs84
    )

    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    im = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        zorder=1
    )

    return im, vmin, vmax


def add_dem_legend(fig, ax, im, vmin, vmax):
    """
    Add DEM legend.
    """
    cax = ax.inset_axes([0.70, 0.31, 0.055, 0.20])

    cbar = fig.colorbar(im, cax=cax)

    cbar.set_ticks([vmax, vmin])
    cbar.set_ticklabels([
        f"High : {int(round(vmax))}",
        f"Low : {int(round(vmin))}"
    ])

    cbar.ax.tick_params(labelsize=8)

    cax.set_title(
        "DEM",
        fontsize=10,
        fontweight="bold",
        pad=5
    )


# ============================================================
# Load administrative boundary data
# ============================================================

try:
    raw_states_gdf = load_geojson_from_url(INDIA_STATES_URL)
    raw_districts_gdf = load_geojson_from_url(INDIA_DISTRICTS_URL)

except Exception as e:
    st.error(f"Could not download India boundary data from GitHub: {e}")
    st.stop()


states_gdf, state_col, districts_gdf, district_state_col = prepare_admin_layers(
    raw_states_gdf,
    raw_districts_gdf
)

state_names = get_state_names(states_gdf, state_col)

if not state_names:
    st.error("State names could not be detected.")
    st.write("Available columns in states layer:")
    st.write(states_gdf.columns.tolist())
    st.stop()


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.header("1. India and State Selection")

default_state = "Chhattisgarh"

state_names_normalized = [normalize_text(x) for x in state_names]

if normalize_text(default_state) in state_names_normalized:
    default_state_index = state_names_normalized.index(normalize_text(default_state))
else:
    default_state_index = 0

selected_state_name = st.sidebar.selectbox(
    "Select State / UT",
    state_names,
    index=default_state_index
)

selected_state_gdf = filter_state(
    states_gdf,
    state_col,
    selected_state_name
)

if selected_state_gdf.empty:
    st.error("Selected state could not be extracted.")
    st.stop()


india_panel_mode = st.sidebar.selectbox(
    "Top-left India panel",
    [
        "Complete India with selected state highlighted",
        "Selected state only"
    ],
    index=0
)


state_panel_mode = st.sidebar.selectbox(
    "Bottom-left state panel",
    [
        "Complete selected state with district boundaries",
        "Selected district only",
        "Selected state with study area"
    ],
    index=0
)


state_districts_gdf = filter_districts_for_state(
    districts_gdf,
    district_state_col,
    selected_state_gdf,
    selected_state_name
)

district_col = detect_district_column(state_districts_gdf)

selected_district_gdf = None
selected_district_name = "None"

if district_col is not None and not state_districts_gdf.empty:
    district_names = (
        state_districts_gdf[district_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_district_name = st.sidebar.selectbox(
        "Optional: Select district to highlight",
        ["None"] + district_names,
        index=0
    )

    if selected_district_name != "None":
        selected_district_gdf = state_districts_gdf[
            state_districts_gdf[district_col].astype(str) == selected_district_name
        ].copy()


st.sidebar.header("2. Upload Study Area")

study_file = st.sidebar.file_uploader(
    "Upload study area / catchment file",
    type=["zip", "geojson", "json", "gpkg"],
    help="For shapefile, upload a ZIP containing .shp, .shx, .dbf and preferably .prj."
)

default_crs = st.sidebar.text_input(
    "Default CRS if uploaded study file has no CRS",
    value="EPSG:4326"
)

dem_file = st.sidebar.file_uploader(
    "Optional: upload DEM GeoTIFF",
    type=["tif", "tiff"]
)


st.sidebar.header("3. Map Titles")

country_title = st.sidebar.text_input(
    "Top-left panel title",
    value="India"
)

state_title = st.sidebar.text_input(
    "Bottom-left panel title",
    value=str(selected_state_name)
)

main_title = st.sidebar.text_input(
    "Main panel title",
    value="Study Area"
)


st.sidebar.header("4. Map Style")

country_fill = st.sidebar.color_picker(
    "India map fill",
    value="#cbe7f2"
)

state_highlight_color = st.sidebar.color_picker(
    "Highlighted state color",
    value="#ff5a1f"
)

state_fill = st.sidebar.color_picker(
    "State/district panel fill",
    value="#f4b6b6"
)

district_highlight_color = st.sidebar.color_picker(
    "District/study area highlight color",
    value="#1f78ff"
)

study_boundary_color = st.sidebar.color_picker(
    "Study area boundary color",
    value="#000000"
)

study_fill_color = st.sidebar.color_picker(
    "Study area fill color without DEM",
    value="#1f78ff"
)

dem_cmap = st.sidebar.selectbox(
    "DEM color palette",
    ["plasma", "terrain", "viridis", "turbo", "rainbow", "Spectral_r"],
    index=0
)

output_dpi = st.sidebar.selectbox(
    "Output DPI",
    [150, 200, 300, 400, 600],
    index=2
)

show_main_district_background = st.sidebar.checkbox(
    "Show district boundaries in main panel background",
    value=False
)

show_main_legend_without_dem = st.sidebar.checkbox(
    "Show main legend when DEM is not used",
    value=True
)


# ============================================================
# Stop until study file is uploaded
# ============================================================

if study_file is None:
    st.info(
        """
        Upload your **study area / catchment boundary** to generate the final map.

        India and district boundaries are loaded automatically.  
        You can select the required **State / UT** from the sidebar dropdown.
        """
    )

    st.write("### Required")
    st.write("- Study area/catchment file: ZIP shapefile, GeoJSON, JSON or GPKG")

    st.write("### Optional")
    st.write("- DEM GeoTIFF for elevation map in the main panel")

    with st.expander("Detected administrative columns"):
        st.write("Detected state column:", state_col)
        st.write("Detected district-state column:", district_state_col)
        st.write("Detected district column:", district_col)
        st.write("Available state names:")
        st.write(state_names)

    st.stop()


# ============================================================
# Read uploaded study area and DEM
# ============================================================

if "study_map_workdir" not in st.session_state:
    st.session_state["study_map_workdir"] = tempfile.mkdtemp(prefix="study_area_map_")

workdir = Path(st.session_state["study_map_workdir"])

try:
    study_gdf = read_uploaded_vector(
        study_file,
        workdir / "study",
        default_crs=default_crs
    )

    dem_path = read_uploaded_raster(
        dem_file,
        workdir / "dem"
    )

except Exception as e:
    st.error(f"Error while reading uploaded data: {e}")
    st.stop()


# ============================================================
# Study area feature filter
# ============================================================

st.subheader("Study Area Feature Selection")

study_columns = [
    c for c in study_gdf.columns
    if c != study_gdf.geometry.name
]

col1, col2, col3 = st.columns(3)

with col1:
    study_filter_col = st.selectbox(
        "Optional: filter study area by attribute",
        ["None"] + study_columns
    )

study_selected_gdf = study_gdf.copy()

with col2:
    if study_filter_col != "None":
        study_values = (
            study_gdf[study_filter_col]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        selected_study_value = st.selectbox(
            "Select study area feature",
            ["All"] + study_values
        )

        study_selected_gdf = filter_by_attribute(
            study_gdf,
            study_filter_col,
            selected_study_value
        )

with col3:
    st.metric("Selected features", len(study_selected_gdf))

if study_selected_gdf.empty:
    st.error("Selected study area is empty.")
    st.stop()


# ============================================================
# Create figure layout
# ============================================================

fig = plt.figure(figsize=(13, 8), dpi=output_dpi)

ax_india = fig.add_axes([0.05, 0.55, 0.38, 0.37])
ax_state = fig.add_axes([0.05, 0.08, 0.38, 0.37])
ax_main = fig.add_axes([0.50, 0.08, 0.45, 0.84])


# ============================================================
# Panel 1: India panel
# ============================================================

if india_panel_mode == "Complete India with selected state highlighted":
    states_gdf.plot(
        ax=ax_india,
        color=country_fill,
        edgecolor="black",
        linewidth=0.35,
        zorder=1
    )

    selected_state_gdf.plot(
        ax=ax_india,
        color=state_highlight_color,
        edgecolor="black",
        linewidth=0.65,
        zorder=3
    )

    set_extent(ax_india, states_gdf, pad=0.04)

else:
    selected_state_gdf.plot(
        ax=ax_india,
        color=state_highlight_color,
        edgecolor="black",
        linewidth=0.65,
        zorder=3
    )

    set_extent(ax_india, selected_state_gdf, pad=0.08)

add_panel_title(ax_india, country_title, fontsize=14)
add_north_arrow(ax_india, x=0.88, y=0.77, size=0.10)
add_scale_bar_degree(ax_india, location=(0.08, 0.06), fontsize=7)
apply_degree_grid(ax_india, fontsize=7, xbins=4, ybins=4)
add_map_border(ax_india)


# ============================================================
# Panel 2: State / district panel
# ============================================================

if state_panel_mode == "Complete selected state with district boundaries":
    state_districts_gdf.plot(
        ax=ax_state,
        color=state_fill,
        edgecolor="black",
        linewidth=0.35,
        zorder=1
    )

    selected_state_gdf.boundary.plot(
        ax=ax_state,
        color="black",
        linewidth=0.9,
        zorder=2
    )

    if selected_district_gdf is not None and not selected_district_gdf.empty:
        selected_district_gdf.plot(
            ax=ax_state,
            color=district_highlight_color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3
        )

    study_selected_gdf.plot(
        ax=ax_state,
        color=district_highlight_color,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.90,
        zorder=4
    )

    set_extent(ax_state, selected_state_gdf, pad=0.08)


elif state_panel_mode == "Selected district only" and selected_district_gdf is not None and not selected_district_gdf.empty:
    selected_district_gdf.plot(
        ax=ax_state,
        color=state_fill,
        edgecolor="black",
        linewidth=0.7,
        zorder=1
    )

    study_selected_gdf.plot(
        ax=ax_state,
        color=district_highlight_color,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.90,
        zorder=3
    )

    set_extent(ax_state, selected_district_gdf, pad=0.12)


else:
    selected_state_gdf.plot(
        ax=ax_state,
        color=state_fill,
        edgecolor="black",
        linewidth=0.8,
        zorder=1
    )

    study_selected_gdf.plot(
        ax=ax_state,
        color=district_highlight_color,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.90,
        zorder=3
    )

    set_extent(ax_state, selected_state_gdf, pad=0.08)

add_panel_title(ax_state, state_title, fontsize=14)
add_north_arrow(ax_state, x=0.88, y=0.77, size=0.10)
add_scale_bar_degree(ax_state, location=(0.08, 0.06), fontsize=7)
apply_degree_grid(ax_state, fontsize=7, xbins=4, ybins=4)
add_map_border(ax_state)


# ============================================================
# Panel 3: Main study area / DEM panel
# ============================================================

dem_plotted = False

if show_main_district_background:
    try:
        state_districts_gdf.boundary.plot(
            ax=ax_main,
            color="gray",
            linewidth=0.3,
            alpha=0.6,
            zorder=0
        )
    except Exception:
        pass

if dem_path is not None:
    try:
        im, dem_min, dem_max = plot_dem(
            ax=ax_main,
            raster_path=dem_path,
            study_gdf_wgs84=study_selected_gdf,
            cmap=dem_cmap
        )

        study_selected_gdf.boundary.plot(
            ax=ax_main,
            color=study_boundary_color,
            linewidth=0.8,
            zorder=5
        )

        add_dem_legend(
            fig=fig,
            ax=ax_main,
            im=im,
            vmin=dem_min,
            vmax=dem_max
        )

        dem_plotted = True

    except Exception as e:
        st.warning(f"DEM could not be plotted. Boundary map will be shown instead. Error: {e}")

if not dem_plotted:
    study_selected_gdf.plot(
        ax=ax_main,
        color=study_fill_color,
        edgecolor=study_boundary_color,
        linewidth=0.8,
        alpha=0.90,
        zorder=3
    )

    if show_main_legend_without_dem:
        main_patch = Patch(
            facecolor=study_fill_color,
            edgecolor=study_boundary_color,
            label="Study Area"
        )

        ax_main.legend(
            handles=[main_patch],
            loc="lower right",
            fontsize=9,
            frameon=True,
            framealpha=0.95
        )

set_extent(ax_main, study_selected_gdf, pad=0.08)

add_panel_title(ax_main, main_title, fontsize=14)
add_north_arrow(ax_main, x=0.90, y=0.82, size=0.10)
add_scale_bar_degree(ax_main, location=(0.08, 0.055), fontsize=7)
apply_degree_grid(ax_main, fontsize=8, xbins=4, ybins=7)
add_map_border(ax_main)


# ============================================================
# Connecting lines
# ============================================================

try:
    con1 = ConnectionPatch(
        xyA=(0.98, 0.12),
        coordsA=ax_india.transAxes,
        xyB=(0.02, 0.92),
        coordsB=ax_main.transAxes,
        color="black",
        linewidth=1.2,
        zorder=80
    )

    con2 = ConnectionPatch(
        xyA=(0.98, 0.78),
        coordsA=ax_state.transAxes,
        xyB=(0.02, 0.18),
        coordsB=ax_main.transAxes,
        color="black",
        linewidth=1.2,
        zorder=80
    )

    fig.add_artist(con1)
    fig.add_artist(con2)

except Exception:
    pass


# ============================================================
# Display and download
# ============================================================

st.subheader("Generated Study Area Map")

png_buffer = io.BytesIO()
fig.savefig(
    png_buffer,
    format="png",
    dpi=output_dpi,
    bbox_inches="tight",
    facecolor="white"
)
png_buffer.seek(0)

pdf_buffer = io.BytesIO()
fig.savefig(
    pdf_buffer,
    format="pdf",
    bbox_inches="tight",
    facecolor="white"
)
pdf_buffer.seek(0)

st.image(
    png_buffer,
    caption="Generated study area map",
    use_container_width=True
)

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        label="Download PNG",
        data=png_buffer.getvalue(),
        file_name="study_area_map.png",
        mime="image/png"
    )

with download_col2:
    st.download_button(
        label="Download PDF",
        data=pdf_buffer.getvalue(),
        file_name="study_area_map.pdf",
        mime="application/pdf"
    )

plt.close(fig)


# ============================================================
# Layer information
# ============================================================

with st.expander("Layer information and detected fields"):
    st.write("### India states layer")
    st.write(f"Features: {len(states_gdf)}")
    st.write(f"CRS: {states_gdf.crs}")
    st.write(f"Detected state column: {state_col}")

    st.write("### India districts layer")
    st.write(f"Features: {len(districts_gdf)}")
    st.write(f"CRS: {districts_gdf.crs}")
    st.write(f"Detected district-state column: {district_state_col}")
    st.write(f"Detected district column: {district_col}")

    st.write("### Selected state")
    st.write(selected_state_name)

    st.write("### State districts used")
    st.write(f"Features: {len(state_districts_gdf)}")

    st.write("### Study area")
    st.write(f"Features: {len(study_selected_gdf)}")
    st.write(f"CRS: {study_selected_gdf.crs}")

    st.write("### Raw states columns")
    st.write(raw_states_gdf.columns.tolist())

    st.write("### Raw districts columns")
    st.write(raw_districts_gdf.columns.tolist())


st.markdown("---")
st.markdown(
    """
    **Note:** Administrative boundaries from public repositories should be verified before use in official/legal maps.  
    This app is intended for research, academic reports, theses and publication-style study area figures.
    """
)
