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
    "Create publication-style study area maps with India inset, state/district inset, main catchment map, DEM, north arrow, scale bar, grid and legend."
)


# ============================================================
# GitHub boundary source
# ============================================================

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master"

INDIA_STATES_URL = f"{GITHUB_RAW_BASE}/INDIA/INDIA_STATES.geojson"
INDIA_DISTRICTS_URL = f"{GITHUB_RAW_BASE}/INDIA/INDIA_DISTRICTS.geojson"


# ============================================================
# Column name helpers
# ============================================================

STATE_COLUMN_CANDIDATES = [
    "ST_NM", "st_nm", "STATE", "State", "state",
    "STATE_NAME", "State_Name", "state_name",
    "NAME_1", "Name", "NAME", "name"
]

DISTRICT_COLUMN_CANDIDATES = [
    "DISTRICT", "District", "district",
    "DIST_NAME", "Dist_Name", "dist_name",
    "dtname", "DTNAME", "DT_NAME",
    "NAME_2", "Name", "NAME", "name"
]


def normalize_text(x):
    """Normalize text for matching names."""
    return str(x).strip().upper().replace("&", "AND")


def find_first_existing_column(gdf, candidates):
    """Find the first matching column from a list of possible names."""
    for col in candidates:
        if col in gdf.columns:
            return col
    return None


def find_best_name_column(gdf):
    """Try to identify a suitable name column."""
    all_candidates = STATE_COLUMN_CANDIDATES + DISTRICT_COLUMN_CANDIDATES

    for col in all_candidates:
        if col in gdf.columns:
            return col

    object_cols = [
        c for c in gdf.columns
        if c != gdf.geometry.name and gdf[c].dtype == "object"
    ]

    if object_cols:
        return object_cols[0]

    return None


# ============================================================
# Data loading functions
# ============================================================

@st.cache_data(show_spinner=True)
def load_geojson_from_url(url):
    """
    Download GeoJSON from GitHub raw URL and read using GeoPandas.
    Cached to avoid repeated downloading.
    """
    response = requests.get(url, timeout=120)
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
    """Save uploaded file to temporary working directory."""
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
    - zipped shapefile
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
            raise ValueError("No .shp file found inside the uploaded ZIP.")

        gdf = gpd.read_file(shp_files[0])

    elif suffix in [".geojson", ".json", ".gpkg"]:
        gdf = gpd.read_file(saved_path)

    else:
        raise ValueError("Unsupported vector format. Upload ZIP shapefile, GeoJSON or GPKG.")

    if gdf.empty:
        raise ValueError("Uploaded vector file is empty.")

    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs)

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[~gdf.geometry.isna()].copy()

    return gdf


def read_uploaded_raster(uploaded_file, out_dir):
    """Save optional uploaded DEM/GeoTIFF."""
    if uploaded_file is None:
        return None

    return save_uploaded_file(uploaded_file, out_dir)


# ============================================================
# Spatial filtering
# ============================================================

def get_state_names(states_gdf):
    """Extract state names for dropdown."""
    state_col = find_first_existing_column(states_gdf, STATE_COLUMN_CANDIDATES)

    if state_col is None:
        state_col = find_best_name_column(states_gdf)

    if state_col is None:
        return None, []

    names = (
        states_gdf[state_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    return state_col, names


def filter_state(states_gdf, state_col, selected_state):
    """Filter India state boundary."""
    if state_col is None:
        return gpd.GeoDataFrame(columns=states_gdf.columns, crs=states_gdf.crs)

    selected_norm = normalize_text(selected_state)

    out = states_gdf[
        states_gdf[state_col].astype(str).apply(normalize_text) == selected_norm
    ].copy()

    if out.empty:
        out = states_gdf[
            states_gdf[state_col].astype(str).apply(normalize_text).str.contains(selected_norm, na=False)
        ].copy()

    return out


def filter_districts_for_state(districts_gdf, selected_state_gdf, selected_state_name):
    """
    Filter district layer for selected state.
    First tries attribute-based filtering.
    If that fails, falls back to spatial intersection.
    """
    state_col = find_first_existing_column(districts_gdf, STATE_COLUMN_CANDIDATES)

    if state_col is not None:
        selected_norm = normalize_text(selected_state_name)

        out = districts_gdf[
            districts_gdf[state_col].astype(str).apply(normalize_text) == selected_norm
        ].copy()

        if not out.empty:
            return out

        out = districts_gdf[
            districts_gdf[state_col].astype(str).apply(normalize_text).str.contains(selected_norm, na=False)
        ].copy()

        if not out.empty:
            return out

    if selected_state_gdf is not None and not selected_state_gdf.empty:
        state_geom = selected_state_gdf.geometry.unary_union
        out = districts_gdf[districts_gdf.geometry.intersects(state_geom)].copy()

        if not out.empty:
            return out

    return districts_gdf.copy()


def filter_by_attribute(gdf, column, value):
    """Filter any GeoDataFrame by selected attribute value."""
    if column == "None" or value == "All":
        return gdf.copy()

    return gdf[gdf[column].astype(str) == str(value)].copy()


# ============================================================
# Cartographic helpers
# ============================================================

def padded_bounds(bounds, pad=0.08):
    """Apply padding to bounds."""
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
    """Set map extent based on GeoDataFrame bounds."""
    minx, miny, maxx, maxy = padded_bounds(gdf.total_bounds, pad=pad)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)


def format_dms(value, is_lon=True):
    """Format decimal degree as degree-minute-second label."""
    hemi = ""

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
    """Apply longitude-latitude tick formatting."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_dms(x, is_lon=True)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: format_dms(y, is_lon=False)))

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
    """Add simple north arrow with N, S, E, W labels."""
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

    ax.text(x, y - 0.055, "S", transform=ax.transAxes,
            ha="center", va="center", fontsize=7, zorder=50)

    ax.text(x - 0.065, y + 0.035, "W", transform=ax.transAxes,
            ha="center", va="center", fontsize=7, zorder=50)

    ax.text(x + 0.065, y + 0.035, "E", transform=ax.transAxes,
            ha="center", va="center", fontsize=7, zorder=50)


def km_to_degree_lon(km, latitude):
    """Approximate km to longitude-degree conversion at given latitude."""
    denom = 111.32 * np.cos(np.deg2rad(latitude))

    if abs(denom) < 1e-6:
        denom = 111.32

    return km / denom


def choose_nice_scale_length(width_km):
    """Choose cartographically nice scale-bar length."""
    target = width_km / 5

    nice_values = [
        1, 2, 5, 10, 20, 25, 50, 75, 100,
        150, 200, 250, 500, 750, 1000, 1500,
        2000, 2500, 3000, 3500
    ]

    for v in nice_values:
        if v >= target:
            return v

    return 5000


def add_scale_bar_degree(ax, location=(0.08, 0.065), segments=4, fontsize=7):
    """
    Add approximate scale bar on maps plotted in geographic coordinates.
    Suitable for study-area and regional maps.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width_deg = xlim[1] - xlim[0]
    height_deg = ylim[1] - ylim[0]
    center_lat = (ylim[0] + ylim[1]) / 2

    width_km = width_deg * 111.32 * np.cos(np.deg2rad(center_lat))
    width_km = abs(width_km)

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
    """Add panel title inside the map frame."""
    ax.text(
        0.05,
        0.94,
        title,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="normal",
        ha="left",
        va="top",
        zorder=60
    )


def add_map_border(ax, linewidth=1.0):
    """Style map border."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor("black")


# ============================================================
# Raster helpers
# ============================================================

def clip_dem_to_study_area_wgs84(raster_path, study_gdf_wgs84):
    """
    Reproject DEM to EPSG:4326 using WarpedVRT, clip to study area,
    and return array and plotting extent in lon-lat coordinates.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Uploaded raster has no CRS. Please use a georeferenced GeoTIFF.")

        with WarpedVRT(
            src,
            crs="EPSG:4326",
            resampling=Resampling.bilinear
        ) as vrt:

            study_for_raster = study_gdf_wgs84.to_crs(vrt.crs)
            shapes = [geom for geom in study_for_raster.geometry if geom is not None]

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
                raise ValueError("DEM clipping resulted in empty raster. Check DEM and study area overlap.")

            extent = plotting_extent(arr, transform)

            return arr, extent


def plot_dem(ax, raster_path, study_gdf_wgs84, cmap="plasma"):
    """Plot clipped DEM on main map."""
    arr, extent = clip_dem_to_study_area_wgs84(raster_path, study_gdf_wgs84)

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
    """Add DEM colorbar similar to journal-style study area maps."""
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
# Sidebar controls
# ============================================================

st.sidebar.header("1. Boundary Data")

try:
    india_states_gdf = load_geojson_from_url(INDIA_STATES_URL)
    india_districts_gdf = load_geojson_from_url(INDIA_DISTRICTS_URL)
except Exception as e:
    st.error(f"Could not download India boundary data from GitHub: {e}")
    st.stop()

state_col, state_names = get_state_names(india_states_gdf)

if not state_names:
    st.error("Could not identify state names in India state boundary file.")
    st.stop()

default_state = "Chhattisgarh"

if default_state in state_names:
    default_state_index = state_names.index(default_state)
else:
    default_state_index = 0

selected_state_name = st.sidebar.selectbox(
    "Select state to highlight",
    state_names,
    index=default_state_index
)

selected_state_gdf = filter_state(
    india_states_gdf,
    state_col,
    selected_state_name
)

if selected_state_gdf.empty:
    st.error("Selected state could not be found in the state boundary file.")
    st.stop()

state_districts_gdf = filter_districts_for_state(
    india_districts_gdf,
    selected_state_gdf,
    selected_state_name
)

district_col = find_first_existing_column(
    state_districts_gdf,
    DISTRICT_COLUMN_CANDIDATES
)

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
        "Optional: district to highlight",
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
    value=selected_state_name
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
    "District/catchment highlight color",
    value="#1f78ff"
)

catchment_boundary_color = st.sidebar.color_picker(
    "Study area boundary color",
    value="#000000"
)

catchment_fill_color = st.sidebar.color_picker(
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

show_district_boundaries_main = st.sidebar.checkbox(
    "Show district boundaries in main map background",
    value=False
)

show_legend_main = st.sidebar.checkbox(
    "Show main map legend when no DEM is used",
    value=True
)


# ============================================================
# Main app
# ============================================================

if study_file is None:
    st.info(
        """
        Upload your **study area / catchment shapefile ZIP** to generate the map.

        The app will automatically use India state and district boundaries from GitHub.
        You may also upload a DEM GeoTIFF to create a colored elevation map in the main panel.
        """
    )

    st.write("### Required upload")
    st.write("- Study area/catchment boundary as ZIP shapefile, GeoJSON, or GPKG")

    st.write("### Optional upload")
    st.write("- DEM GeoTIFF for the main panel")

    st.stop()


# Persistent temporary working folder
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
# Study area feature filtering
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
    st.write("Selected feature count")
    st.metric("Features", len(study_selected_gdf))


if study_selected_gdf.empty:
    st.error("Selected study area is empty.")
    st.stop()


# ============================================================
# Figure creation
# ============================================================

fig = plt.figure(figsize=(13, 8), dpi=output_dpi)

# Layout similar to uploaded sample
ax_india = fig.add_axes([0.05, 0.55, 0.38, 0.37])
ax_state = fig.add_axes([0.05, 0.08, 0.38, 0.37])
ax_main = fig.add_axes([0.50, 0.08, 0.45, 0.84])


# ============================================================
# Panel 1: India location map
# ============================================================

india_states_gdf.plot(
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
    linewidth=0.55,
    zorder=3
)

set_extent(ax_india, india_states_gdf, pad=0.04)

add_panel_title(ax_india, country_title, fontsize=14)
add_north_arrow(ax_india, x=0.88, y=0.77, size=0.10)
add_scale_bar_degree(ax_india, location=(0.08, 0.06), fontsize=7)
apply_degree_grid(ax_india, fontsize=7, xbins=4, ybins=4)
add_map_border(ax_india)


# ============================================================
# Panel 2: State / district context map
# ============================================================

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
    linewidth=0.8,
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

# Plot actual study area in state panel
study_selected_gdf.plot(
    ax=ax_state,
    color=district_highlight_color,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.90,
    zorder=4
)

set_extent(ax_state, selected_state_gdf, pad=0.08)

add_panel_title(ax_state, state_title, fontsize=14)
add_north_arrow(ax_state, x=0.88, y=0.77, size=0.10)
add_scale_bar_degree(ax_state, location=(0.08, 0.06), fontsize=7)
apply_degree_grid(ax_state, fontsize=7, xbins=4, ybins=4)
add_map_border(ax_state)


# ============================================================
# Panel 3: Main study area / DEM map
# ============================================================

dem_plotted = False

if show_district_boundaries_main:
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
            color=catchment_boundary_color,
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
        st.warning(f"DEM could not be plotted. Showing study area boundary only. Error: {e}")

if not dem_plotted:
    study_selected_gdf.plot(
        ax=ax_main,
        color=catchment_fill_color,
        edgecolor=catchment_boundary_color,
        linewidth=0.8,
        alpha=0.90,
        zorder=3
    )

    if show_legend_main:
        main_patch = Patch(
            facecolor=catchment_fill_color,
            edgecolor=catchment_boundary_color,
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
# Connecting lines between inset panels and main map
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
    dpi=output_dpi,
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

with st.expander("Layer information"):
    st.write("### India states layer")
    st.write(f"Features: {len(india_states_gdf)}")
    st.write(f"CRS: {india_states_gdf.crs}")

    st.write("### India districts layer")
    st.write(f"Features: {len(india_districts_gdf)}")
    st.write(f"CRS: {india_districts_gdf.crs}")

    st.write("### Selected state")
    st.write(selected_state_name)

    st.write("### State districts used in bottom-left panel")
    st.write(f"Features: {len(state_districts_gdf)}")

    st.write("### Study area")
    st.write(f"Features: {len(study_selected_gdf)}")
    st.write(f"CRS: {study_selected_gdf.crs}")


st.markdown("---")

st.markdown(
    """
    **Note:** For official reports, verify the administrative boundary layer before submission.
    The app is intended for academic, research, and publication-style study area maps.
    """
)
