from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="EnergiSight Prototype", page_icon="ES", layout="wide")


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "deliverables" / "models"
DATA_PATH = ROOT_DIR / "2016_Building_Energy_Benchmarking.csv"

CO2_MODEL_PATH = MODEL_DIR / "co2_final_model.joblib"
ENERGY_MODEL_PATH = MODEL_DIR / "energy_final_model.joblib"

ALL_FEATURES = [
    "YearBuilt",
    "BuildingAge",
    "NumberofBuildings",
    "NumberofFloors",
    "PropertyGFATotal",
    "PropertyGFAParking",
    "PropertyGFABuilding(s)",
    "LargestPropertyUseTypeGFA",
    "SecondLargestPropertyUseTypeGFA",
    "ThirdLargestPropertyUseTypeGFA",
    "ParkingRatio",
    "MainUseRatio",
    "ENERGYSTARScore",
    "PrimaryPropertyType",
    "Neighborhood",
    "GeoCluster",
]

# Top 6 issues de la permutation importance (communes CO2 et Energie)
TOP6_FEATURES = [
    "PropertyGFATotal",
    "PropertyGFABuilding(s)",
    "LargestPropertyUseTypeGFA",
    "PrimaryPropertyType",
    "NumberofBuildings",
    "ENERGYSTARScore",
]

TOP6_NUMERIC = [
    "PropertyGFATotal",
    "PropertyGFABuilding(s)",
    "LargestPropertyUseTypeGFA",
    "NumberofBuildings",
    "ENERGYSTARScore",
]

NUMERIC_FEATURES = [c for c in ALL_FEATURES if c not in {"PrimaryPropertyType", "Neighborhood", "GeoCluster"}]
CATEGORICAL_FEATURES = ["PrimaryPropertyType", "Neighborhood", "GeoCluster"]

HIDDEN_ASSUMPTION_FIELDS = [
    "YearBuilt",
    "NumberofFloors",
    "PropertyGFAParking",
    "SecondLargestPropertyUseTypeGFA",
    "ThirdLargestPropertyUseTypeGFA",
    "Neighborhood",
    "GeoCluster",
]


def apply_custom_style():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --bg: #111820;
  --paper: #1a2530;
  --ink: #e9f0f3;
  --muted: #9eb0b8;
  --line: #31424c;
  --accent: #6ed2c8;
  --accent-2: #f0a56f;
  --chip: #21343c;
}

[data-testid="stAppViewContainer"] {
  background:
    linear-gradient(130deg, rgba(240,165,111,0.10) 0%, rgba(240,165,111,0.01) 30%, transparent 42%),
    radial-gradient(1200px 580px at 88% -12%, rgba(110,210,200,0.16) 0%, transparent 62%),
    var(--bg);
  color: var(--ink);
}

[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1620 0%, #171f2b 100%);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
}

section[data-testid="stSidebar"] * {
  color: #edf2f6 !important;
}

html, body, [class*="css"] {
  font-family: "Sora", "Segoe UI", sans-serif !important;
}

h1, h2, h3, h4 {
  font-family: "Fraunces", "Times New Roman", serif !important;
  color: var(--ink);
  letter-spacing: -0.01em;
}

.block-container {
  max-width: 1240px;
  padding-top: 1.5rem;
  padding-bottom: 2.2rem;
  animation: fadeInUp 420ms ease both;
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.hero {
  background:
    linear-gradient(160deg, #1d2b38 0%, #17232f 100%);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.1rem 1.4rem 1.25rem;
  box-shadow:
    0 16px 34px rgba(3, 8, 12, 0.38),
    inset 0 1px 0 rgba(255, 255, 255, 0.06);
  margin-bottom: 0.8rem;
}

.hero-kicker {
  margin: 0;
  font-size: 0.72rem;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--accent-2);
  font-weight: 800;
}

.hero-title {
  margin: 0.18rem 0 0.24rem;
  font-size: 2.2rem;
  line-height: 1.04;
}

.hero-sub {
  margin: 0;
  color: var(--muted);
  font-size: 0.95rem;
}

.kpi-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.6rem;
  margin: 0.5rem 0 1rem;
}

.kpi-tile {
  border: 1px solid var(--line);
  background: var(--paper);
  border-radius: 16px;
  padding: 0.62rem 0.78rem 0.7rem;
  box-shadow: 0 8px 22px rgba(20, 31, 44, 0.07);
}

.kpi-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  font-weight: 700;
}

.kpi-value {
  margin-top: 0.22rem;
  font-size: 1.35rem;
  font-weight: 800;
  color: var(--ink);
}

.section-chip {
  display: inline-block;
  margin: 0.2rem 0 0.65rem;
  padding: 0.26rem 0.72rem;
  border-radius: 999px;
  border: 1px solid #b8d1ce;
  background: var(--chip);
  color: var(--accent);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

div[data-baseweb="tab-list"] {
  gap: 0.52rem;
  margin-bottom: 0.18rem;
}

button[data-baseweb="tab"] {
  border-radius: 999px !important;
  border: 1px solid #3d4d55 !important;
  background: #1b2834 !important;
  padding: 0.34rem 0.95rem !important;
  font-weight: 600 !important;
  color: #d6e6ea !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  background: #20414a !important;
  border-color: #70bcb4 !important;
  color: #dbfffb !important;
}

[data-testid="stMetric"] {
  border: 1px solid var(--line);
  background: var(--paper);
  border-radius: 14px;
  padding: 0.72rem 0.88rem;
  box-shadow: 0 10px 24px rgba(18, 29, 40, 0.06);
}

[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--ink); }

div[data-testid="stCodeBlock"] {
  border-radius: 12px;
  border: 1px solid var(--line);
  background: #17212b;
}

div.stButton > button {
  border-radius: 999px;
  border: 1px solid #66c7be;
  background: linear-gradient(180deg, #2d7880 0%, #245f66 100%);
  color: #fff;
  font-weight: 700;
  box-shadow: 0 9px 20px rgba(8, 28, 33, 0.45);
}

div.stButton > button:hover {
  border-color: #78d8cf;
  background: linear-gradient(180deg, #356f78 0%, #285760 100%);
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}

div[data-baseweb="input"] {
  background: #151f29 !important;
  border-color: #3b4b53 !important;
}

div[data-baseweb="input"] input {
  color: #e7f0f3 !important;
}

div[data-baseweb="select"] > div {
  background: #151f29 !important;
  border-color: #3b4b53 !important;
  color: #e7f0f3 !important;
}

[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] button:hover {
  background: #2b4f5d !important;
  border: 1px solid #4d7380 !important;
  color: #ffffff !important;
}

[data-testid="stDownloadButton"] button {
  border-radius: 999px;
  border: 1px solid #d89e71 !important;
  background: linear-gradient(180deg, #d08a57 0%, #b7703e 100%) !important;
  color: #fff !important;
  font-weight: 700;
}

[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #16222d;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
label {
  color: var(--ink) !important;
}

@media (max-width: 980px) {
  .kpi-strip {
    grid-template-columns: 1fr;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_section_chip(text):
    st.markdown(f"<span class='section-chip'>{text}</span>", unsafe_allow_html=True)


def render_kpi_strip(intensity_q):
    html = f"""
<div class="kpi-strip">
  <div class="kpi-tile">
    <div class="kpi-label">Mediane intensite</div>
    <div class="kpi-value">{intensity_q['q50']:.2f} kBtu/sf</div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-label">P90 intensite</div>
    <div class="kpi-value">{intensity_q['q90']:.2f} kBtu/sf</div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-label">P95 intensite</div>
    <div class="kpi-value">{intensity_q['q95']:.2f} kBtu/sf</div>
  </div>
</div>
    """
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    if not CO2_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele introuvable: {CO2_MODEL_PATH}")
    if not ENERGY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele introuvable: {ENERGY_MODEL_PATH}")
    return joblib.load(CO2_MODEL_PATH), joblib.load(ENERGY_MODEL_PATH)


@st.cache_data
def load_reference_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df[df["BuildingType"] == "NonResidential"].copy()
    df = df[df["ComplianceStatus"] == "Compliant"].copy()
    if "Outlier" in df.columns:
        df = df[df["Outlier"].fillna("").str.lower() != "high outlier"]
    return df


def _safe_divide(num, den):
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    return np.where((den > 0) & den.notna() & num.notna(), num / den, 0.0)


def _mode_or_na(df, col):
    if col not in df.columns:
        return "NA"
    mode = df[col].dropna().astype(str).mode()
    return mode.iloc[0] if not mode.empty else "NA"


def build_default_row(ref_df):
    numeric_medians = ref_df.select_dtypes(include=["number"]).median(numeric_only=True)
    defaults = {}
    for col in NUMERIC_FEATURES:
        defaults[col] = float(numeric_medians.get(col, 0.0))

    defaults["YearBuilt"] = float(numeric_medians.get("YearBuilt", 1990.0))
    defaults["BuildingAge"] = max(0.0, 2016.0 - defaults["YearBuilt"])
    defaults["ParkingRatio"] = defaults["PropertyGFAParking"] / defaults["PropertyGFATotal"] if defaults["PropertyGFATotal"] > 0 else 0.0
    defaults["MainUseRatio"] = (
        defaults["LargestPropertyUseTypeGFA"] / defaults["PropertyGFATotal"] if defaults["PropertyGFATotal"] > 0 else 0.0
    )
    defaults["PrimaryPropertyType"] = _mode_or_na(ref_df, "PrimaryPropertyType")
    defaults["Neighborhood"] = _mode_or_na(ref_df, "Neighborhood")
    defaults["GeoCluster"] = _mode_or_na(ref_df, "GeoCluster")
    return defaults


def build_reference_ranges(ref_df):
    ranges = {}
    for col in TOP6_NUMERIC:
        s = pd.to_numeric(ref_df[col], errors="coerce").dropna() if col in ref_df.columns else pd.Series(dtype=float)
        if s.empty:
            ranges[col] = {"q01": None, "q99": None}
        else:
            ranges[col] = {"q01": float(s.quantile(0.01)), "q99": float(s.quantile(0.99))}
    return ranges


@st.cache_data
def build_intensity_reference(ref_df):
    energy = pd.to_numeric(ref_df["SiteEnergyUse(kBtu)"], errors="coerce")
    area = pd.to_numeric(ref_df["PropertyGFATotal"], errors="coerce")
    valid = energy.notna() & area.notna() & (area > 0)

    intensity = (energy[valid] / area[valid]).astype(float).to_numpy()
    intensity = intensity[np.isfinite(intensity)]
    intensity_sorted = np.sort(intensity)

    if intensity_sorted.size == 0:
        q = {"q25": np.nan, "q50": np.nan, "q75": np.nan, "q90": np.nan, "q95": np.nan}
    else:
        q = {
            "q25": float(np.quantile(intensity_sorted, 0.25)),
            "q50": float(np.quantile(intensity_sorted, 0.50)),
            "q75": float(np.quantile(intensity_sorted, 0.75)),
            "q90": float(np.quantile(intensity_sorted, 0.90)),
            "q95": float(np.quantile(intensity_sorted, 0.95)),
        }
    return intensity_sorted, q


def build_histogram_df(values, bins=40):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return pd.DataFrame(columns=["intensity", "count"])
    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"intensity": centers, "count": counts})


def build_cdf_df(values, max_points=700):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return pd.DataFrame(columns=["percentile", "intensity"])
    if arr.size > max_points:
        idx = np.linspace(0, arr.size - 1, max_points).astype(int)
        arr = arr[idx]
    pct = np.linspace(0, 100, arr.size)
    return pd.DataFrame({"percentile": pct, "intensity": arr})


def render_intensity_distribution(intensity_sorted, intensity_q, markers, title):
    hist_df = build_histogram_df(intensity_sorted)
    if hist_df.empty:
        st.info("Distribution de reference indisponible.")
        return

    marker_values = []
    for m in markers:
        x = float(m.get("x", np.nan))
        if np.isfinite(x):
            marker_values.append({"label": str(m.get("label", "Marker")), "x": x})

    for q_label, q_key in [("Q50", "q50"), ("Q90", "q90"), ("Q95", "q95")]:
        x = float(intensity_q.get(q_key, np.nan))
        if np.isfinite(x):
            marker_values.append({"label": q_label, "x": x})

    spec = {
        "width": "container",
        "height": 280,
        "layer": [
            {
                "data": {"values": hist_df.to_dict("records")},
                "mark": {"type": "bar", "opacity": 0.68, "color": "#73a9a2"},
                "encoding": {
                    "x": {"field": "intensity", "type": "quantitative", "title": "Intensite (kBtu/sf)"},
                    "y": {"field": "count", "type": "quantitative", "title": "Nombre de batiments"},
                    "tooltip": [
                        {"field": "intensity", "type": "quantitative", "format": ".2f"},
                        {"field": "count", "type": "quantitative"},
                    ],
                },
            },
            {
                "data": {"values": marker_values},
                "mark": {"type": "rule", "strokeWidth": 2.5},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative"},
                    "color": {"field": "label", "type": "nominal"},
                    "tooltip": [
                        {"field": "label", "type": "nominal"},
                        {"field": "x", "type": "quantitative", "format": ".2f"},
                    ],
                },
            },
        ],
    }
    st.markdown(f"**{title}**")
    st.vega_lite_chart(spec, use_container_width=True)


def render_reference_cdf(intensity_sorted):
    cdf_df = build_cdf_df(intensity_sorted)
    if cdf_df.empty:
        return

    spec = {
        "width": "container",
        "height": 260,
        "data": {"values": cdf_df.to_dict("records")},
        "mark": {"type": "line", "strokeWidth": 3, "color": "#1f7a7a"},
        "encoding": {
            "x": {"field": "percentile", "type": "quantitative", "title": "Percentile (%)"},
            "y": {"field": "intensity", "type": "quantitative", "title": "Intensite (kBtu/sf)"},
            "tooltip": [
                {"field": "percentile", "type": "quantitative", "format": ".1f"},
                {"field": "intensity", "type": "quantitative", "format": ".2f"},
            ],
        },
    }
    st.markdown("**Courbe cumulative de reference (intensite)**")
    st.vega_lite_chart(spec, use_container_width=True)


def energy_intensity_percentile(energy_pred, area, intensity_sorted):
    if area is None or not np.isfinite(area) or area <= 0:
        return np.nan, np.nan
    intensity = energy_pred / area
    if intensity_sorted.size == 0 or not np.isfinite(intensity):
        return intensity, np.nan
    rank = np.searchsorted(intensity_sorted, intensity, side="right")
    pct = 100.0 * rank / intensity_sorted.size
    return intensity, pct


def classify_intensity(intensity_value, q):
    if not np.isfinite(intensity_value):
        return "ND"
    if intensity_value <= q["q25"]:
        return "Bas"
    if intensity_value <= q["q75"]:
        return "Moyen"
    if intensity_value <= q["q90"]:
        return "Eleve"
    return "Tres eleve"


def intensity_vector(energy_pred, area_series, intensity_sorted):
    area = pd.to_numeric(area_series, errors="coerce").to_numpy(dtype=float)
    intensity = np.where(area > 0, np.asarray(energy_pred, dtype=float) / area, np.nan)
    percentile = np.full(intensity.shape, np.nan, dtype=float)

    if intensity_sorted.size > 0:
        valid = np.isfinite(intensity)
        ranks = np.searchsorted(intensity_sorted, intensity[valid], side="right")
        percentile[valid] = 100.0 * ranks / intensity_sorted.size
    return intensity, percentile


def intensity_band_vector(intensity_arr, q):
    arr = np.asarray(intensity_arr, dtype=float)
    out = np.full(arr.shape, "ND", dtype=object)
    valid = np.isfinite(arr)
    out[valid & (arr <= q["q25"])] = "Bas"
    out[valid & (arr > q["q25"]) & (arr <= q["q75"])] = "Moyen"
    out[valid & (arr > q["q75"]) & (arr <= q["q90"])] = "Eleve"
    out[valid & (arr > q["q90"])] = "Tres eleve"
    return out


def merge_assumptions(defaults, overrides):
    merged = defaults.copy()
    merged.update(overrides)
    return merged


def build_features_from_top6(raw_top6_df, assumptions):
    out = pd.DataFrame([assumptions] * len(raw_top6_df))
    for col in TOP6_FEATURES:
        if col in raw_top6_df.columns:
            out[col] = raw_top6_df[col]

    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ParkingRatio"] = _safe_divide(out["PropertyGFAParking"], out["PropertyGFATotal"])
    out["MainUseRatio"] = _safe_divide(out["LargestPropertyUseTypeGFA"], out["PropertyGFATotal"])
    out["BuildingAge"] = (2016 - out["YearBuilt"]).clip(lower=0)

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("NA").astype(str)
    return out[ALL_FEATURES]


def validate_top6_columns(df):
    return [c for c in TOP6_FEATURES if c not in df.columns]


def domain_warnings(raw_df):
    warnings = []
    numeric = raw_df.copy()
    for col in TOP6_NUMERIC:
        if col in numeric.columns:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    if (numeric["ENERGYSTARScore"] < 0).any() or (numeric["ENERGYSTARScore"] > 100).any():
        warnings.append("ENERGYSTARScore devrait etre entre 0 et 100.")
    if (numeric["PropertyGFABuilding(s)"] > numeric["PropertyGFATotal"]).any():
        warnings.append("PropertyGFABuilding(s) > PropertyGFATotal sur certaines lignes.")
    if (numeric["LargestPropertyUseTypeGFA"] > numeric["PropertyGFATotal"]).any():
        warnings.append("LargestPropertyUseTypeGFA > PropertyGFATotal sur certaines lignes.")
    if (numeric[TOP6_NUMERIC] < 0).any().any():
        warnings.append("Certaines variables numeriques sont negatives.")
    return warnings


def range_warnings(raw_df, ranges):
    warnings = []
    for col in TOP6_NUMERIC:
        if col not in raw_df.columns:
            continue
        lo = ranges[col]["q01"]
        hi = ranges[col]["q99"]
        if lo is None or hi is None:
            continue
        s = pd.to_numeric(raw_df[col], errors="coerce")
        if ((s < lo) | (s > hi)).any():
            warnings.append(f"{col}: valeurs hors plage de reference [{lo:,.2f}, {hi:,.2f}] (1%-99%).")
    return warnings


def range_flags_per_row(raw_df, ranges):
    flags = pd.DataFrame(index=raw_df.index)
    for col in TOP6_NUMERIC:
        lo = ranges[col]["q01"]
        hi = ranges[col]["q99"]
        s = pd.to_numeric(raw_df[col], errors="coerce")
        if lo is None or hi is None:
            flags[f"flag_{col}_out_of_range"] = False
        else:
            flags[f"flag_{col}_out_of_range"] = (s < lo) | (s > hi)
    flags["out_of_range_count"] = flags.sum(axis=1).astype(int)
    return flags


def predict_df(features_df, co2_model, energy_model):
    return co2_model.predict(features_df), energy_model.predict(features_df)


def render_scenario_summary_chart(summary_df):
    melt = summary_df.melt(
        id_vars=["Scenario"],
        value_vars=["CO2_pred", "Energy_pred_kBtu", "Intensity_kBtu_sf"],
        var_name="Metric",
        value_name="Value",
    )
    metric_labels = {
        "CO2_pred": "CO2",
        "Energy_pred_kBtu": "Energie (kBtu)",
        "Intensity_kBtu_sf": "Intensite (kBtu/sf)",
    }
    melt["Metric"] = melt["Metric"].map(metric_labels)

    spec = {
        "width": "container",
        "height": 250,
        "data": {"values": melt.to_dict("records")},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 5, "cornerRadiusTopRight": 5},
        "encoding": {
            "x": {"field": "Scenario", "type": "nominal", "title": ""},
            "y": {"field": "Value", "type": "quantitative", "title": "Valeur"},
            "color": {"field": "Scenario", "type": "nominal"},
            "column": {"field": "Metric", "type": "nominal", "title": ""},
            "tooltip": [
                {"field": "Scenario", "type": "nominal"},
                {"field": "Metric", "type": "nominal"},
                {"field": "Value", "type": "quantitative", "format": ",.2f"},
            ],
        },
    }
    st.markdown("**Comparaison visuelle des scenarios**")
    st.vega_lite_chart(spec, use_container_width=True)


def render_batch_dashboard(result_df):
    plot_df = result_df.copy()
    plot_df["PropertyGFATotal"] = pd.to_numeric(plot_df["PropertyGFATotal"], errors="coerce")
    plot_df["pred_SiteEnergyUse_kBtu"] = pd.to_numeric(plot_df["pred_SiteEnergyUse_kBtu"], errors="coerce")
    plot_df["pred_EnergyIntensity_kBtu_sf"] = pd.to_numeric(plot_df["pred_EnergyIntensity_kBtu_sf"], errors="coerce")
    plot_df = plot_df.dropna(subset=["PropertyGFATotal", "pred_SiteEnergyUse_kBtu", "pred_EnergyIntensity_kBtu_sf"]).copy()
    if plot_df.empty:
        return

    sample_df = plot_df.head(4000)
    c1, c2 = st.columns(2)

    scatter_spec = {
        "width": "container",
        "height": 300,
        "data": {"values": sample_df.to_dict("records")},
        "mark": {"type": "circle", "opacity": 0.75, "size": 70},
        "encoding": {
            "x": {"field": "PropertyGFATotal", "type": "quantitative", "title": "Surface totale (sf)"},
            "y": {"field": "pred_SiteEnergyUse_kBtu", "type": "quantitative", "title": "Energie predite (kBtu)"},
            "color": {"field": "pred_EnergyIntensity_band", "type": "nominal", "title": "Niveau intensite"},
            "tooltip": [
                {"field": "PropertyGFATotal", "type": "quantitative", "format": ",.2f"},
                {"field": "pred_SiteEnergyUse_kBtu", "type": "quantitative", "format": ",.2f"},
                {"field": "pred_EnergyIntensity_kBtu_sf", "type": "quantitative", "format": ".2f"},
                {"field": "pred_EnergyIntensity_percentile", "type": "quantitative", "format": ".1f"},
                {"field": "pred_EnergyIntensity_band", "type": "nominal"},
            ],
        },
    }

    hist_spec = {
        "width": "container",
        "height": 300,
        "data": {"values": sample_df.to_dict("records")},
        "mark": {"type": "bar", "opacity": 0.85},
        "encoding": {
            "x": {
                "bin": {"maxbins": 35},
                "field": "pred_EnergyIntensity_kBtu_sf",
                "type": "quantitative",
                "title": "Intensite predite (kBtu/sf)",
            },
            "y": {"aggregate": "count", "type": "quantitative", "title": "Nombre de batiments"},
            "color": {"field": "pred_EnergyIntensity_band", "type": "nominal", "title": "Niveau intensite"},
            "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Count"}],
        },
    }

    with c1:
        st.markdown("**Energie predite vs surface**")
        st.vega_lite_chart(scatter_spec, use_container_width=True)
    with c2:
        st.markdown("**Distribution des intensites predites**")
        st.vega_lite_chart(hist_spec, use_container_width=True)


def build_top6_input_form(prefix, property_types, defaults):
    c1, c2 = st.columns(2)
    property_type = c1.selectbox(
        f"PrimaryPropertyType {prefix}",
        options=property_types,
        index=0 if property_types else None,
        key=f"{prefix}_property_type",
    )
    energy_star = c1.number_input(
        f"ENERGYSTARScore {prefix}",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults["ENERGYSTARScore"]),
        key=f"{prefix}_energy_star",
    )
    number_of_buildings = c1.number_input(
        f"NumberofBuildings {prefix}",
        min_value=0.0,
        value=float(defaults["NumberofBuildings"]),
        key=f"{prefix}_nbuildings",
    )

    gfa_total = c2.number_input(
        f"PropertyGFATotal {prefix}",
        min_value=0.0,
        value=float(defaults["PropertyGFATotal"]),
        key=f"{prefix}_gfa_total",
    )
    gfa_building = c2.number_input(
        f"PropertyGFABuilding(s) {prefix}",
        min_value=0.0,
        value=float(defaults["PropertyGFABuilding(s)"]),
        key=f"{prefix}_gfa_building",
    )
    gfa_main = c2.number_input(
        f"LargestPropertyUseTypeGFA {prefix}",
        min_value=0.0,
        value=float(defaults["LargestPropertyUseTypeGFA"]),
        key=f"{prefix}_gfa_main",
    )

    return {
        "PropertyGFATotal": gfa_total,
        "PropertyGFABuilding(s)": gfa_building,
        "LargestPropertyUseTypeGFA": gfa_main,
        "PrimaryPropertyType": property_type,
        "NumberofBuildings": number_of_buildings,
        "ENERGYSTARScore": energy_star,
    }


def main():
    apply_custom_style()

    st.markdown(
        """
<div class="hero">
  <p class="hero-kicker">Dashboard Prediction Engine</p>
  <h1 class="hero-title">EnergiSight Prototype</h1>
  <p class="hero-sub">Interface de simulation pour emissions CO2 et energie, avec lecture d'intensite contextuelle et controles de qualite.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        co2_model, energy_model = load_models()
        ref_df = load_reference_data()
    except Exception as exc:  # pragma: no cover
        st.error(f"Erreur de chargement: {exc}")
        st.stop()

    defaults = build_default_row(ref_df)
    ranges = build_reference_ranges(ref_df)
    intensity_sorted, intensity_q = build_intensity_reference(ref_df)

    property_types = sorted(ref_df["PrimaryPropertyType"].dropna().astype(str).unique().tolist())
    neighborhoods = sorted(ref_df["Neighborhood"].dropna().astype(str).unique().tolist())
    geoclusters = sorted(ref_df["GeoCluster"].dropna().astype(str).unique().tolist()) if "GeoCluster" in ref_df.columns else ["NA"]

    with st.sidebar:
        st.subheader("Hypotheses avancees")
        st.caption("Variables cachees encore utilisees par le modele.")
        year_built = st.number_input("YearBuilt", min_value=1800.0, max_value=2026.0, value=float(defaults["YearBuilt"]))
        number_of_floors = st.number_input("NumberofFloors", min_value=0.0, value=float(defaults["NumberofFloors"]))
        gfa_parking = st.number_input("PropertyGFAParking", min_value=0.0, value=float(defaults["PropertyGFAParking"]))
        gfa_second = st.number_input(
            "SecondLargestPropertyUseTypeGFA", min_value=0.0, value=float(defaults["SecondLargestPropertyUseTypeGFA"])
        )
        gfa_third = st.number_input(
            "ThirdLargestPropertyUseTypeGFA", min_value=0.0, value=float(defaults["ThirdLargestPropertyUseTypeGFA"])
        )
        neighborhood = st.selectbox(
            "Neighborhood",
            options=neighborhoods if neighborhoods else [defaults["Neighborhood"]],
            index=0,
        )
        geocluster = st.selectbox(
            "GeoCluster",
            options=geoclusters if geoclusters else ["NA"],
            index=0,
        )

    assumptions = merge_assumptions(
        defaults,
        {
            "YearBuilt": year_built,
            "NumberofFloors": number_of_floors,
            "PropertyGFAParking": gfa_parking,
            "SecondLargestPropertyUseTypeGFA": gfa_second,
            "ThirdLargestPropertyUseTypeGFA": gfa_third,
            "Neighborhood": neighborhood,
            "GeoCluster": geocluster,
        },
    )

    render_section_chip("Top 6 Features")
    st.subheader("Variables d'entree prioritaires")
    st.code(", ".join(TOP6_FEATURES), language="text")
    render_kpi_strip(intensity_q)

    with st.expander("Visualiser la distribution de reference", expanded=False):
        cdf_col, hist_col = st.columns(2)
        with cdf_col:
            render_reference_cdf(intensity_sorted)
        with hist_col:
            render_intensity_distribution(
                intensity_sorted,
                intensity_q,
                markers=[],
                title="Histogramme de reference (intensite)",
            )

    tab1, tab2, tab3 = st.tabs(["Prediction unitaire", "Comparaison scenarios", "Prediction batch CSV"])

    with tab1:
        render_section_chip("Prediction Unitaire")
        with st.form("single_form"):
            top6_dict = build_top6_input_form("A", property_types, defaults)
            submitted = st.form_submit_button("Predire")

        if submitted:
            raw_input = pd.DataFrame([top6_dict])
            for msg in domain_warnings(raw_input) + range_warnings(raw_input, ranges):
                st.warning(msg)

            features_df = build_features_from_top6(raw_input, assumptions)
            co2_pred, energy_pred = predict_df(features_df, co2_model, energy_model)

            area = float(raw_input.loc[0, "PropertyGFATotal"])
            intensity_val, intensity_pct = energy_intensity_percentile(float(energy_pred[0]), area, intensity_sorted)
            intensity_band = classify_intensity(intensity_val, intensity_q)

            st.success("Prediction terminee")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CO2 predit", f"{co2_pred[0]:,.2f}")
            m2.metric("Energie predite (kBtu)", f"{energy_pred[0]:,.2f}")
            m3.metric("Intensite predite (kBtu/sf)", f"{intensity_val:,.2f}")
            m4.metric("Percentile intensite", f"{intensity_pct:,.1f}e")
            st.info(f"Niveau d'intensite estime: {intensity_band}")
            render_intensity_distribution(
                intensity_sorted,
                intensity_q,
                markers=[{"label": "Prediction", "x": float(intensity_val)}],
                title="Position de la prediction dans la distribution",
            )

    with tab2:
        render_section_chip("Comparaison Scenarios")
        with st.form("compare_form"):
            st.markdown("### Scenario A")
            a_dict = build_top6_input_form("Scenario_A", property_types, defaults)
            st.markdown("### Scenario B")
            b_dict = build_top6_input_form("Scenario_B", property_types, defaults)
            compare_submitted = st.form_submit_button("Comparer A vs B")

        if compare_submitted:
            raw_a = pd.DataFrame([a_dict])
            raw_b = pd.DataFrame([b_dict])
            feat_a = build_features_from_top6(raw_a, assumptions)
            feat_b = build_features_from_top6(raw_b, assumptions)
            co2_a, en_a = predict_df(feat_a, co2_model, energy_model)
            co2_b, en_b = predict_df(feat_b, co2_model, energy_model)

            ia, pca = energy_intensity_percentile(float(en_a[0]), float(raw_a.loc[0, "PropertyGFATotal"]), intensity_sorted)
            ib, pcb = energy_intensity_percentile(float(en_b[0]), float(raw_b.loc[0, "PropertyGFATotal"]), intensity_sorted)

            delta_co2 = float(co2_b[0] - co2_a[0])
            delta_en = float(en_b[0] - en_a[0])
            delta_i = float(ib - ia)

            c1, c2, c3 = st.columns(3)
            c1.metric("Delta CO2 (B - A)", f"{delta_co2:,.2f}")
            c2.metric("Delta energie (B - A)", f"{delta_en:,.2f}")
            c3.metric("Delta intensite (B - A)", f"{delta_i:,.2f}")

            summary = pd.DataFrame(
                [
                    {
                        "Scenario": "A",
                        "CO2_pred": float(co2_a[0]),
                        "Energy_pred_kBtu": float(en_a[0]),
                        "Intensity_kBtu_sf": ia,
                        "Intensity_percentile": pca,
                        "Intensity_band": classify_intensity(ia, intensity_q),
                    },
                    {
                        "Scenario": "B",
                        "CO2_pred": float(co2_b[0]),
                        "Energy_pred_kBtu": float(en_b[0]),
                        "Intensity_kBtu_sf": ib,
                        "Intensity_percentile": pcb,
                        "Intensity_band": classify_intensity(ib, intensity_q),
                    },
                ]
            )
            st.dataframe(summary, use_container_width=True)
            render_scenario_summary_chart(summary)
            render_intensity_distribution(
                intensity_sorted,
                intensity_q,
                markers=[
                    {"label": "Scenario A", "x": float(ia)},
                    {"label": "Scenario B", "x": float(ib)},
                ],
                title="Position A/B dans la distribution d'intensite",
            )

    with tab3:
        render_section_chip("Prediction Batch")
        st.write("Le CSV batch doit contenir exactement ces 6 colonnes:")
        st.code(", ".join(TOP6_FEATURES), language="text")

        template_row = {
            "PropertyGFATotal": 125000,
            "PropertyGFABuilding(s)": 110000,
            "LargestPropertyUseTypeGFA": 80000,
            "PrimaryPropertyType": defaults["PrimaryPropertyType"],
            "NumberofBuildings": 1,
            "ENERGYSTARScore": 72,
        }
        template_csv = pd.DataFrame([template_row]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Telecharger template CSV",
            data=template_csv,
            file_name="energisight_template_top6.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Importer un CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_batch = pd.read_csv(uploaded_file)
                missing = validate_top6_columns(raw_batch)
                if missing:
                    st.error(f"Colonnes manquantes: {missing}")
                else:
                    for msg in domain_warnings(raw_batch):
                        st.warning(msg)
                    for msg in range_warnings(raw_batch, ranges):
                        st.warning(msg)

                    batch_features = build_features_from_top6(raw_batch, assumptions)
                    co2_pred, energy_pred = predict_df(batch_features, co2_model, energy_model)
                    intensity, pct = intensity_vector(energy_pred, raw_batch["PropertyGFATotal"], intensity_sorted)

                    result = raw_batch.copy()
                    result["pred_TotalGHGEmissions"] = co2_pred
                    result["pred_SiteEnergyUse_kBtu"] = energy_pred
                    result["pred_EnergyIntensity_kBtu_sf"] = intensity
                    result["pred_EnergyIntensity_percentile"] = pct
                    result["pred_EnergyIntensity_band"] = intensity_band_vector(intensity, intensity_q)

                    flags = range_flags_per_row(raw_batch, ranges)
                    result = pd.concat([result, flags], axis=1)

                    st.success(f"{len(result)} lignes predites")
                    st.info(f"Lignes potentiellement hors distribution: {int((result['out_of_range_count'] > 0).sum())}")
                    st.dataframe(result.head(100), use_container_width=True)
                    render_batch_dashboard(result)

                    csv_bytes = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Telecharger les predictions",
                        data=csv_bytes,
                        file_name="energisight_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as exc:  # pragma: no cover
                st.error(f"Erreur pendant la prediction batch: {exc}")


if __name__ == "__main__":
    main()
