import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="SalaryIQ",
    page_icon="◈",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL STYLES — Bridge color palette + minimalist
# Bridge palette: deep steel #0D1117, span gray #1C2333,
# cable orange #E8692A, tension teal #2ABFBF,
# mist white #F0F4F8, iron #8B9BB4
# ──────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>

<style>
/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  background: #0D1117 !important;
  color: #C9D3E0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── App shell ── */
.stApp { background: #0D1117 !important; }
.main .block-container {
  padding: 2rem 2.5rem 3rem !important;
  max-width: 1360px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0A0E14 !important;
  border-right: 1px solid #1C2333 !important;
}
[data-testid="stSidebar"] * { color: #C9D3E0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { font-size: 0.78rem !important; color: #6B7B94 !important; }

/* ── Form controls ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: #1C2333 !important;
  border: 1px solid #252D3D !important;
  border-radius: 8px !important;
  color: #C9D3E0 !important;
  font-family: 'Outfit', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: #E8692A !important;
  box-shadow: 0 0 0 2px rgba(232, 105, 42, 0.12) !important;
}
.stSlider > div > div { color: #E8692A !important; }
.stSlider [data-testid="stThumbValue"] { color: #E8692A !important; }
.stRadio > div { gap: 0.5rem; }
.stRadio label span { color: #8B9BB4 !important; font-size: 0.85rem !important; }
.stCheckbox label span { color: #8B9BB4 !important; font-size: 0.82rem !important; }
.stFileUploader {
  background: #1C2333 !important;
  border: 1px dashed #252D3D !important;
  border-radius: 10px !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid #1C2333 !important;
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: #6B7B94 !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.02em !important;
  padding: 0.6rem 1.2rem !important;
  transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #C9D3E0 !important;
  background: rgba(232, 105, 42, 0.04) !important;
}
.stTabs [aria-selected="true"] {
  color: #E8692A !important;
  border-bottom-color: #E8692A !important;
  background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
  background: #E8692A !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.55rem 1.8rem !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.02em !important;
  font-family: 'Outfit', sans-serif !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 12px rgba(232, 105, 42, 0.25) !important;
}
.stButton > button:hover {
  background: #D45A1A !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(232, 105, 42, 0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary button variant */
.btn-ghost > button {
  background: transparent !important;
  color: #8B9BB4 !important;
  border: 1px solid #252D3D !important;
  box-shadow: none !important;
}
.btn-ghost > button:hover {
  border-color: #E8692A !important;
  color: #E8692A !important;
  background: rgba(232, 105, 42, 0.06) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: #1C2333;
  border: 1px solid #252D3D;
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: rgba(232, 105, 42, 0.4); }
[data-testid="stMetricValue"] {
  color: #F0F4F8 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
  color: #6B7B94 !important;
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}
[data-testid="stMetricDelta"] { font-size: 0.72rem !important; color: #2ABFBF !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: #1C2333 !important;
  border-radius: 8px !important;
  color: #C9D3E0 !important;
  border: 1px solid #252D3D !important;
}
.streamlit-expanderContent {
  background: #151B27 !important;
  border: 1px solid #252D3D !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; font-size: 0.85rem !important; }
div[data-testid="stNotificationContentInfo"] { background: rgba(42, 191, 191, 0.08) !important; border-left: 3px solid #2ABFBF !important; }
div[data-testid="stNotificationContentSuccess"] { background: rgba(42, 191, 191, 0.08) !important; border-left: 3px solid #2ABFBF !important; }
div[data-testid="stNotificationContentWarning"] { background: rgba(232, 105, 42, 0.08) !important; border-left: 3px solid #E8692A !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #252D3D; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #E8692A; }

/* ── Divider ── */
hr { border-color: #1C2333 !important; margin: 1.5rem 0 !important; }

/* ── COMPONENT CLASSES ── */

/* Page title wordmark */
.wordmark {
  font-family: 'Outfit', sans-serif;
  font-weight: 700;
  font-size: 1.4rem;
  color: #F0F4F8;
  letter-spacing: -0.02em;
}
.wordmark span { color: #E8692A; }

/* Section label */
.sec-label {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #6B7B94;
  margin-bottom: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.sec-label::before {
  content: '';
  display: inline-block;
  width: 16px;
  height: 2px;
  background: #E8692A;
  border-radius: 2px;
}

/* Section heading */
.sec-heading {
  font-family: 'Outfit', sans-serif;
  font-size: 1.25rem;
  font-weight: 600;
  color: #F0F4F8;
  letter-spacing: -0.01em;
  margin: 0 0 1.25rem 0;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #1C2333;
}

/* Stat card */
.stat-card {
  background: #1C2333;
  border: 1px solid #252D3D;
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover { border-color: rgba(232, 105, 42, 0.35); transform: translateY(-1px); }
.stat-card .label { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #6B7B94; margin-bottom: 0.4rem; }
.stat-card .value { font-size: 1.5rem; font-weight: 700; color: #F0F4F8; font-family: 'Outfit', sans-serif; line-height: 1.1; }
.stat-card .sub { font-size: 0.72rem; color: #2ABFBF; margin-top: 0.25rem; }

/* Info badge */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.22rem 0.75rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.03em;
}
.badge-orange { background: rgba(232, 105, 42, 0.12); color: #E8692A; border: 1px solid rgba(232, 105, 42, 0.25); }
.badge-teal   { background: rgba(42, 191, 191, 0.1);  color: #2ABFBF; border: 1px solid rgba(42, 191, 191, 0.2); }
.badge-mist   { background: rgba(240, 244, 248, 0.05); color: #8B9BB4; border: 1px solid #252D3D; }

/* Prediction result card */
.pred-card {
  background: linear-gradient(160deg, #1C2333 0%, #151B27 100%);
  border: 1px solid #252D3D;
  border-radius: 12px;
  padding: 2rem 2.4rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}
.pred-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, #E8692A, #2ABFBF);
}
.pred-card:hover { border-color: rgba(232, 105, 42, 0.4); }
.pred-amount {
  font-family: 'Outfit', sans-serif;
  font-size: 3.2rem;
  font-weight: 700;
  color: #F0F4F8;
  letter-spacing: -0.03em;
  line-height: 1;
}
.pred-range {
  font-size: 0.82rem;
  color: #6B7B94;
  margin-top: 0.4rem;
  font-family: 'JetBrains Mono', monospace;
}

/* Insight row */
.insight-row {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  background: #1C2333;
  border: 1px solid #252D3D;
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.6rem;
  transition: border-color 0.2s, transform 0.2s;
}
.insight-row:hover { border-color: rgba(232, 105, 42, 0.3); transform: translateX(2px); }
.insight-icon { font-size: 1.2rem; flex-shrink: 0; }
.insight-title { font-weight: 600; color: #F0F4F8; font-size: 0.85rem; margin-bottom: 0.2rem; }
.insight-body { font-size: 0.82rem; color: #8B9BB4; line-height: 1.55; }

/* Model navbar button */
.model-nav-btn {
  background: #1C2333;
  border: 1px solid #252D3D;
  border-radius: 8px;
  padding: 0.6rem 0.9rem;
  cursor: pointer;
  transition: all 0.18s ease;
  text-align: center;
  width: 100%;
}
.model-nav-btn:hover { border-color: #E8692A; background: rgba(232,105,42,0.06); }
.model-nav-btn.active {
  border-color: #E8692A;
  background: rgba(232,105,42,0.1);
}
.model-nav-btn .icon { font-size: 1.1rem; margin-bottom: 0.2rem; }
.model-nav-btn .name { font-size: 0.72rem; font-weight: 600; color: #C9D3E0; }
.model-nav-btn .desc { font-size: 0.65rem; color: #6B7B94; margin-top: 0.1rem; line-height: 1.3; }

/* Progress bar */
.prog-wrap { background: #0D1117; border-radius: 999px; height: 5px; overflow: hidden; }
.prog-bar { height: 5px; border-radius: 999px; background: linear-gradient(90deg, #E8692A, #2ABFBF); transition: width 1s cubic-bezier(0.22,1,0.36,1); }

/* Compare table */
.cmp-table { width: 100%; border-collapse: collapse; }
.cmp-table th {
  font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: #6B7B94; padding: 0.7rem 1rem; border-bottom: 1px solid #1C2333;
  text-align: left;
}
.cmp-table td {
  padding: 0.6rem 1rem; border-bottom: 1px solid #151B27;
  font-size: 0.84rem; color: #C9D3E0;
  font-family: 'JetBrains Mono', monospace;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tr:hover td { background: rgba(232,105,42,0.04); }
.cmp-table .winner { color: #2ABFBF; font-weight: 600; }

/* Tier pill */
.tier-pill {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.25rem 0.85rem; border-radius: 4px;
  font-size: 0.74rem; font-weight: 600;
}

/* Dot separator */
.dot-sep { color: #252D3D; margin: 0 0.4rem; }

/* Subtle rule */
.rule { height: 1px; background: #1C2333; margin: 1.5rem 0; }

/* Animated entry */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.45s cubic-bezier(0.22,1,0.36,1) both; }
.fade-up-1 { animation-delay: 0.05s; }
.fade-up-2 { animation-delay: 0.12s; }
.fade-up-3 { animation-delay: 0.19s; }
.fade-up-4 { animation-delay: 0.26s; }

/* Bridge cable line (hero accent) */
@keyframes cableDraw {
  from { stroke-dashoffset: 400; }
  to   { stroke-dashoffset: 0; }
}
.cable-line {
  stroke-dasharray: 400;
  animation: cableDraw 1.8s cubic-bezier(0.22,1,0.36,1) forwards;
}

/* Pulse dot */
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.4); }
}
.pulse-live { animation: pulse 2.2s ease-in-out infinite; }

/* Shimmer for loading */
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
.skeleton {
  background: linear-gradient(90deg, #1C2333 25%, #252D3D 50%, #1C2333 75%);
  background-size: 800px 100%;
  animation: shimmer 1.4s linear infinite;
  border-radius: 6px;
}

/* Counter number */
.counter-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: #2ABFBF;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MATPLOTLIB THEME — bridge palette
# ──────────────────────────────────────────────
BG      = "#0D1117"
SURFACE = "#1C2333"
CARD    = "#151B27"
ORANGE  = "#E8692A"
TEAL    = "#2ABFBF"
MIST    = "#F0F4F8"
IRON    = "#8B9BB4"
MUTED   = "#6B7B94"
GRID    = "#1C2333"

def dark_fig(w=10, h=4, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax_list = [axes] if nrows * ncols == 1 else axes.flatten()
    for ax in ax_list:
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
            spine.set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
        ax.xaxis.label.set_color(IRON)
        ax.yaxis.label.set_color(IRON)
        ax.title.set_color(MIST)
        ax.title.set_fontsize(10)
        ax.title.set_fontweight('500')
        ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    return fig, axes


# ──────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    data = data.copy()
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].str.strip()

    data.replace('?', 'Others', inplace=True)
    if 'native-country' in data.columns:
        data['native-country'].replace('?', 'United-States', inplace=True)

    for val in ['Without-pay', 'Never-worked']:
        if 'workclass' in data.columns:
            data = data[data['workclass'] != val]
    for val in ['1st-4th', '5th-6th', 'Preschool']:
        if 'education' in data.columns:
            data = data[data['education'] != val]

    def cap_outliers_iqr(df_col):
        Q1, Q3 = df_col.quantile(0.25), df_col.quantile(0.75)
        IQR = Q3 - Q1
        return pd.Series(np.clip(df_col, Q1 - 1.5*IQR, Q3 + 1.5*IQR), index=df_col.index)

    for col in ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']:
        if col in data.columns:
            data[col] = cap_outliers_iqr(data[col])

    if 'education' in data.columns:
        data.drop(columns=['education'], inplace=True)

    if 'income' in data.columns:
        data['income_clean'] = data['income'].str.strip().str.replace('.', '', regex=False)
        data['income_numeric'] = data['income_clean'].apply(lambda x: 25000 if x == '<=50K' else 75000)
        data.drop(columns=['income_clean'], inplace=True)

    return data


@st.cache_resource
def train_model(X_train_raw, y_train, _preprocessor_transformer, model_type="Random Forest"):
    model_map = {
        "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=1.0, max_iter=5000),
    }
    reg = model_map.get(model_type, RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
    model = Pipeline(steps=[
        ('preprocessor', _preprocessor_transformer),
        ('regressor', reg)
    ])
    model.fit(X_train_raw, y_train)
    return model


def get_feature_importance(model, X):
    try:
        reg = model.named_steps['regressor']
        if not hasattr(reg, 'feature_importances_'):
            return None
        preprocessor = model.named_steps['preprocessor']
        cat_features = preprocessor.transformers_[0][2]
        num_features = preprocessor.transformers_[1][2]
        ohe = preprocessor.transformers_[0][1]
        cat_names = ohe.get_feature_names_out(cat_features).tolist()
        all_names = cat_names + list(num_features)
        fi_df = pd.DataFrame({'Feature': all_names, 'Importance': reg.feature_importances_})
        return fi_df.sort_values('Importance', ascending=False).head(15)
    except Exception:
        return None


def salary_tier(val):
    if val < 30000:   return "Entry Level", ORANGE,   "rgba(232,105,42,0.12)",  "rgba(232,105,42,0.3)"
    elif val < 55000: return "Mid Level",   "#F5C842", "rgba(245,200,66,0.12)", "rgba(245,200,66,0.3)"
    elif val < 75000: return "Senior",      TEAL,     "rgba(42,191,191,0.12)",  "rgba(42,191,191,0.3)"
    else:             return "Executive",   MIST,     "rgba(240,244,248,0.1)",  "rgba(240,244,248,0.25)"


MODELS = ["Random Forest", "Gradient Boosting", "Extra Trees", "Ridge Regression", "Lasso Regression"]
MODEL_META = {
    "Random Forest":     ("🌲", "Ensemble · Importances"),
    "Gradient Boosting": ("⚡", "Boosted · Nonlinear"),
    "Extra Trees":       ("🌳", "Fast · Low variance"),
    "Ridge Regression":  ("◻", "Linear · L2 reg"),
    "Lasso Regression":  ("◈", "Linear · Sparse"),
}


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.2rem 0 0.8rem; display:flex; align-items:center; gap:0.55rem;">
      <span class="pulse-live" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#E8692A;flex-shrink:0;"></span>
      <span class="wordmark">Salary<span>IQ</span></span>
    </div>
    <div style="font-size:0.7rem;color:#6B7B94;margin-bottom:1rem;padding-left:1.3rem;letter-spacing:0.03em;">
      ML Salary Intelligence
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sec-label">Model Config</div>', unsafe_allow_html=True)

    model_choice_sidebar = st.selectbox(
        "Algorithm",
        MODELS,
        help="Tree-based models support feature importance charts.",
        label_visibility="collapsed"
    )

    test_size = st.slider("Test split", 10, 40, 20, step=5, format="%d%%") / 100

    st.divider()
    st.markdown('<div class="sec-label">Display</div>', unsafe_allow_html=True)
    show_confidence  = st.checkbox("Confidence intervals", value=True)
    enable_comparison = st.checkbox("Model comparison mode", value=False)
    show_feat_import = st.checkbox("Feature importance chart", value=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem; color:#6B7B94; line-height:1.8;">
      <span style="color:#C9D3E0;font-weight:500;">How to use</span><br>
      1 · Upload <code style="color:#E8692A;">adult.csv</code><br>
      2 · Explore data in EDA<br>
      3 · Train your model<br>
      4 · Predict single or batch<br>
      5 · Export results
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="badge badge-mist">UCI Census Income · adult.csv</span>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="fade-up" style="
  border-bottom: 1px solid #1C2333;
  padding-bottom: 2rem;
  margin-bottom: 2rem;
">
  <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
    <span class="badge badge-orange">◈ Live</span>
    <span class="badge badge-teal">v2.0</span>
  </div>

  <!-- Bridge cable SVG accent -->
  <svg width="340" height="32" viewBox="0 0 340 32" fill="none"
       xmlns="http://www.w3.org/2000/svg" style="display:block;margin-bottom:0.8rem;opacity:0.6;">
    <line x1="0" y1="28" x2="340" y2="28" stroke="#1C2333" stroke-width="1.5"/>
    <!-- towers -->
    <line x1="80"  y1="28" x2="80"  y2="4" stroke="#252D3D" stroke-width="1.5"/>
    <line x1="260" y1="28" x2="260" y2="4" stroke="#252D3D" stroke-width="1.5"/>
    <!-- main cable -->
    <path d="M0 20 Q80 4 170 14 Q260 4 340 20"
          stroke="#E8692A" stroke-width="1.5" fill="none"
          stroke-dasharray="480" stroke-dashoffset="480"
          class="cable-line"/>
    <!-- suspenders -->
    <line x1="110" y1="28" x2="113" y2="11" stroke="#2ABFBF" stroke-width="0.8" opacity="0.6"/>
    <line x1="140" y1="28" x2="144" y2="9"  stroke="#2ABFBF" stroke-width="0.8" opacity="0.6"/>
    <line x1="170" y1="28" x2="170" y2="14" stroke="#2ABFBF" stroke-width="0.8" opacity="0.6"/>
    <line x1="200" y1="28" x2="196" y2="9"  stroke="#2ABFBF" stroke-width="0.8" opacity="0.6"/>
    <line x1="230" y1="28" x2="227" y2="11" stroke="#2ABFBF" stroke-width="0.8" opacity="0.6"/>
  </svg>

  <h1 style="
    font-family:'Outfit',sans-serif;
    font-size:2.4rem;
    font-weight:700;
    color:#F0F4F8;
    letter-spacing:-0.03em;
    margin:0 0 0.5rem 0;
    line-height:1.1;
  ">
    Salary<span style="color:#E8692A;">IQ</span>
  </h1>

  <p style="
    font-size:0.95rem;
    color:#8B9BB4;
    max-width:520px;
    line-height:1.65;
    margin:0 0 1.2rem 0;
  ">
    Machine-learning salary intelligence built on census data.
    Upload, train in one click, and uncover compensation patterns instantly.
  </p>

  <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
    <span class="badge badge-orange">5 Algorithms</span>
    <span class="badge badge-teal">EDA + Visualizations</span>
    <span class="badge badge-mist">Batch Export</span>
    <span class="badge badge-mist">Salary Insights</span>
    <span class="badge badge-mist">Model Comparison</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODEL NAVBAR
# ──────────────────────────────────────────────
if 'active_model' not in st.session_state:
    st.session_state['active_model'] = model_choice_sidebar

st.markdown('<div class="sec-label fade-up fade-up-1">Select Algorithm</div>', unsafe_allow_html=True)

nav_cols = st.columns(len(MODELS))
for i, m in enumerate(MODELS):
    icon, desc = MODEL_META[m]
    is_active = st.session_state['active_model'] == m
    active_style = "border-color:#E8692A !important;background:rgba(232,105,42,0.08) !important;" if is_active else ""
    name_color   = "#E8692A" if is_active else "#C9D3E0"
    with nav_cols[i]:
        st.markdown(f"""
        <div class="model-nav-btn {'active' if is_active else ''}" style="{active_style}">
          <div class="icon">{icon}</div>
          <div class="name" style="color:{name_color};">{m}</div>
          <div class="desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select", key=f"navbtn_{i}", use_container_width=True,
                     help=f"Use {m}"):
            st.session_state['active_model'] = m
            st.rerun()

# Hide the Select button text, show only card
st.markdown("""
<style>
[key^="navbtn_"] button, div[data-testid="stButton"] button[kind="secondary"] {
  margin-top: -2.6rem !important;
  opacity: 0 !important;
  height: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  position: absolute !important;
}
</style>
""", unsafe_allow_html=True)

# Override: make the full card area the button — via zero-height overlay trick
# Actually just render small link-style buttons cleanly
# Active model info bar
active_model = st.session_state['active_model']
model_choice = active_model
icon_a, desc_a = MODEL_META[active_model]

st.markdown(f"""
<div class="fade-up fade-up-2" style="
  display:flex; align-items:center; gap:0.9rem;
  background:#1C2333; border:1px solid #252D3D;
  border-left: 3px solid #E8692A;
  border-radius:0 8px 8px 0;
  padding:0.75rem 1.2rem; margin:0.75rem 0 1.8rem 0;
">
  <span style="font-size:1.3rem;">{icon_a}</span>
  <div>
    <div style="font-weight:600;color:#F0F4F8;font-size:0.88rem;">{active_model}</div>
    <div style="font-size:0.74rem;color:#6B7B94;">{desc_a}</div>
  </div>
  <div style="margin-left:auto;">
    <span class="badge badge-orange">● Active</span>
  </div>
</div>
<div class="rule"></div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────
st.markdown('<div class="sec-heading fade-up fade-up-3">Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload adult.csv", type="csv", label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="fade-up fade-up-4" style="
      background:#1C2333; border:1px dashed #252D3D;
      border-radius:10px; padding:2.5rem; text-align:center; color:#6B7B94;
    ">
      <div style="font-size:2rem; margin-bottom:0.6rem; opacity:0.5;">⬆</div>
      <div style="font-size:0.9rem; color:#8B9BB4; margin-bottom:0.3rem;">
        Drop <code style="background:#151B27;padding:2px 7px;border-radius:4px;color:#E8692A;">adult.csv</code> above
      </div>
      <div style="font-size:0.75rem;">UCI Census Income · 14 features · ~48,000 records</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Preprocessing…"):
    df = load_and_preprocess_data(uploaded_file)

if df is None:
    st.error("Could not process the file — check its format.")
    st.stop()

n_rows, n_cols = df.shape
high_earners = (df['income_numeric'] == 75000).sum() if 'income_numeric' in df.columns else 0
pct_high     = high_earners / n_rows * 100
avg_salary   = df['income_numeric'].mean() if 'income_numeric' in df.columns else 0

# Dataset stat cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records",           f"{n_rows:,}",          "after cleaning")
c2.metric("Features",          f"{n_cols - 2}",         "input columns")
c3.metric("High Earners",      f"{pct_high:.1f}%",      f"{high_earners:,} records")
c4.metric("Active Algorithm",  model_choice,            "selected")


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_eda, tab_train, tab_predict, tab_batch, tab_insights, tab_compare = st.tabs([
    "  Data Explorer  ",
    "  Train Model  ",
    "  Single Prediction  ",
    "  Batch Prediction  ",
    "  Salary Insights  ",
    "  Compare Models  ",
])


# ═══════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="sec-heading">Explore Your Data</div>', unsafe_allow_html=True)

    eda_t1, eda_t2, eda_t3, eda_t4 = st.tabs(["Preview & Stats", "Distributions", "Correlation", "Income Breakdown"])

    with eda_t1:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown('<div class="sec-label">Dataset Preview — first 50 rows</div>', unsafe_allow_html=True)
            st.dataframe(df.drop(columns=['income_numeric'], errors='ignore').head(50), use_container_width=True)
        with col_b:
            st.markdown('<div class="sec-label">Column Stats</div>', unsafe_allow_html=True)
            num_df = df.select_dtypes(include=np.number)
            for col in num_df.columns[:6]:
                st.markdown(f"""
                <div class="stat-card" style="margin-bottom:0.45rem;padding:0.7rem 0.9rem;">
                  <div class="label">{col}</div>
                  <div style="font-size:0.8rem;color:#C9D3E0;font-family:'JetBrains Mono',monospace;">
                    μ {num_df[col].mean():,.1f}
                    <span class="dot-sep">·</span>
                    σ {num_df[col].std():,.1f}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:1.2rem;">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)

    with eda_t2:
        col_sel, col_plot = st.columns([1, 3])
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        with col_sel:
            chart_type = st.radio("Chart type", ["Histogram", "Categorical Bar", "Box Plot"])
            if chart_type in ["Histogram", "Box Plot"]:
                col_to_plot  = st.selectbox("Column", num_cols)
                split_income = st.checkbox("Split by income", value=True)
            else:
                col_to_plot  = st.selectbox("Column", cat_cols)
                split_income = False

        with col_plot:
            if chart_type == "Histogram":
                fig, ax = dark_fig(9, 4)
                if split_income and 'income' in df.columns:
                    for grp, clr in zip(df['income'].unique(), [ORANGE, TEAL]):
                        vals = df.loc[df['income'] == grp, col_to_plot].dropna()
                        ax.hist(vals, bins=40, alpha=0.55, color=clr, label=grp, edgecolor="none")
                    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=MIST, fontsize=9)
                else:
                    ax.hist(df[col_to_plot].dropna(), bins=40, color=ORANGE, edgecolor="none", alpha=0.8)
                ax.set_xlabel(col_to_plot); ax.set_ylabel("Count")
                ax.set_title(f"Distribution — {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_type == "Box Plot":
                fig, ax = dark_fig(9, 4)
                if split_income and 'income' in df.columns:
                    groups = [df.loc[df['income'] == g, col_to_plot].dropna() for g in df['income'].unique()]
                    labels = df['income'].unique().tolist()
                    bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=True,
                                    medianprops=dict(color=TEAL, linewidth=2))
                    for patch, clr in zip(bp['boxes'], [ORANGE, TEAL]):
                        patch.set_facecolor(clr); patch.set_alpha(0.35)
                    for el in ['whiskers', 'caps', 'fliers']:
                        for item in bp[el]: item.set_color(MUTED)
                else:
                    bp = ax.boxplot(df[col_to_plot].dropna(), patch_artist=True, notch=True,
                                    medianprops=dict(color=TEAL, linewidth=2))
                    bp['boxes'][0].set_facecolor(ORANGE); bp['boxes'][0].set_alpha(0.35)
                ax.set_ylabel(col_to_plot); ax.set_title(f"Box Plot — {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            else:
                counts = df[col_to_plot].value_counts()
                fig, ax = dark_fig(9, max(3, len(counts) * 0.42))
                grad = np.linspace(0.35, 0.85, len(counts))
                colors = [(ORANGE if v < 0.6 else TEAL) for v in grad]
                bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none", height=0.65)
                ax.set_xlabel("Count"); ax.set_title(f"Counts — {col_to_plot}"); ax.invert_yaxis()
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_width() + counts.values.max() * 0.01,
                            bar.get_y() + bar.get_height()/2,
                            f"{val:,}", va='center', color=MUTED, fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with eda_t3:
        num_df = df.select_dtypes(include=np.number)
        corr   = num_df.corr()
        n = len(corr)
        fig, ax = dark_fig(8, 6)
        # Custom diverging colormap using bridge palette
        from matplotlib.colors import LinearSegmentedColormap
        bridge_cmap = LinearSegmentedColormap.from_list("bridge", [TEAL, SURFACE, ORANGE])
        im = ax.imshow(corr.values, cmap=bridge_cmap, vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha='center', va='center',
                        color='white' if abs(corr.values[i,j]) > 0.5 else IRON, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03)
        cbar.ax.tick_params(colors=MUTED)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with eda_t4:
        if 'income' in df.columns and 'occupation' in df.columns:
            fig, axes = dark_fig(12, 9, nrows=2, ncols=2)
            ax1, ax2, ax3, ax4 = axes.flatten()

            occ_income = df.groupby('occupation')['income_numeric'].mean().sort_values()
            bar_clrs = [TEAL if v >= 50000 else ORANGE for v in occ_income.values]
            ax1.barh(occ_income.index, occ_income.values, color=bar_clrs, edgecolor="none", height=0.65)
            ax1.set_xlabel("Avg Income ($)"); ax1.set_title("Avg Income by Occupation")
            ax1.axvline(occ_income.mean(), color=MIST, linestyle='--', linewidth=0.8, alpha=0.5, label='Mean')
            ax1.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=IRON, fontsize=8)

            sample = df.sample(min(2000, len(df)), random_state=42)
            c_scatter = [ORANGE if v == 75000 else TEAL for v in sample['income_numeric']]
            ax2.scatter(sample['age'], sample['hours-per-week'], alpha=0.2, s=8, c=c_scatter, edgecolors='none')
            ax2.set_xlabel("Age"); ax2.set_ylabel("Hours / Week"); ax2.set_title("Age vs. Hours Worked")
            p1 = mpatches.Patch(color=ORANGE, label='>50K')
            p2 = mpatches.Patch(color=TEAL, label='≤50K')
            ax2.legend(handles=[p1, p2], facecolor=SURFACE, edgecolor=GRID, labelcolor=IRON, fontsize=8)

            if 'educational-num' in df.columns:
                edu_income = df.groupby('educational-num')['income_numeric'].mean()
                edu_colors = [ORANGE if v < edu_income.mean() else TEAL for v in edu_income.values]
                ax3.bar(edu_income.index, edu_income.values, color=edu_colors, edgecolor="none", width=0.75)
                ax3.set_xlabel("Education Level (1–16)"); ax3.set_ylabel("Avg Income ($)")
                ax3.set_title("Education Level vs. Avg Income")

            if 'gender' in df.columns:
                g_income = df.groupby('gender')['income_numeric'].mean()
                b = ax4.bar(g_income.index, g_income.values, color=[ORANGE, TEAL], edgecolor="none", width=0.4)
                ax4.set_ylabel("Avg Income ($)"); ax4.set_title("Income by Gender")
                for bar in b:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                             f"${bar.get_height():,.0f}", ha='center', va='bottom', color=MIST, fontsize=9)

            plt.tight_layout(pad=2.0); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Income breakdown requires `income` and `occupation` columns.")


# ═══════════════════════════════════════════════
# TAB 2 — TRAIN
# ═══════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="sec-heading">Train the Model</div>', unsafe_allow_html=True)

    X = df.drop(columns=[c for c in ['income', 'income_numeric'] if c in df.columns])
    y = df['income_numeric']

    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numerical_features   = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor_transformer = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Split info row
    col_tr, col_te, col_feat, col_alg = st.columns(4)
    col_tr.metric("Training Samples",  f"{len(X_train):,}")
    col_te.metric("Testing Samples",   f"{len(X_test):,}")
    col_feat.metric("Features",        f"{len(categorical_features) + len(numerical_features)}")
    col_alg.metric("Algorithm",        model_choice)

    st.markdown(f"""
    <div style="
      background:#1C2333; border:1px solid #252D3D; border-left:3px solid #E8692A;
      border-radius:0 8px 8px 0; padding:0.85rem 1.2rem; margin:1rem 0;
      display:flex; align-items:center; gap:1rem;
    ">
      <span style="font-size:1.1rem;">{MODEL_META[model_choice][0]}</span>
      <div>
        <div style="font-weight:600;color:#F0F4F8;font-size:0.88rem;">{model_choice}</div>
        <div style="font-size:0.74rem;color:#6B7B94;">
          {len(categorical_features)} categorical · {len(numerical_features)} numerical
          · {test_size*100:.0f}% test split
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  Train Model", use_container_width=True):
        with st.spinner(f"Training {model_choice}…"):
            model = train_model(X_train, y_train, preprocessor_transformer, model_type=model_choice)

        st.session_state['model']     = model
        st.session_state['X_columns'] = X.columns.tolist()
        st.session_state['df']        = df
        st.session_state['X_test']    = X_test
        st.session_state['y_test']    = y_test

        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        st.session_state['metrics'] = dict(mae=mae, rmse=rmse, r2=r2)
        st.success(f"✓ {model_choice} trained successfully.")

    if 'metrics' in st.session_state:
        m = st.session_state['metrics']

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE",      f"${m['mae']:,.0f}",  "mean absolute error")
        mc2.metric("RMSE",     f"${m['rmse']:,.0f}", "root mean squared error")
        mc3.metric("R² Score", f"{m['r2']:.3f}",     "1.0 = perfect fit")

        # Fit quality bar
        r2_pct = int(m['r2'] * 100)
        ring_color = TEAL if r2_pct >= 80 else (ORANGE if r2_pct >= 60 else "#F5C842")
        quality    = "Excellent" if r2_pct >= 80 else ("Good" if r2_pct >= 60 else "Moderate")
        qual_note  = ("Model explains most variance — ready for production." if r2_pct >= 80 else
                      "Reasonable predictive power." if r2_pct >= 60 else
                      "Consider a different algorithm or more data.")

        st.markdown(f"""
        <div style="
          background:#1C2333;border:1px solid #252D3D;border-radius:10px;
          padding:1.2rem 1.4rem;margin:1rem 0;
          display:flex;align-items:center;gap:1.4rem;
        ">
          <div style="
            width:72px;height:72px;border-radius:50%;
            border:2.5px solid {ring_color};flex-shrink:0;
            display:flex;align-items:center;justify-content:center;
            font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;
            color:{ring_color};box-shadow:0 0 16px {ring_color}44;
          ">{r2_pct}%</div>
          <div>
            <div style="font-weight:600;color:#F0F4F8;margin-bottom:0.2rem;">{quality} Fit</div>
            <div style="font-size:0.82rem;color:#8B9BB4;">{qual_note}</div>
            <div style="margin-top:0.7rem;">
              <div class="prog-wrap" style="width:200px;">
                <div class="prog-bar" style="width:{r2_pct}%;"></div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance
        if show_feat_import:
            fi_df = get_feature_importance(st.session_state['model'], X)
            if fi_df is not None:
                st.markdown('<div class="sec-label" style="margin-top:1.2rem;">Top 15 Feature Importances</div>', unsafe_allow_html=True)
                fig, ax = dark_fig(10, 5)
                fi_vals  = fi_df['Importance'].values
                fi_normed = fi_vals / fi_vals.max()
                clrs = [ORANGE if v > 0.5 else TEAL for v in fi_normed]
                bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=clrs, edgecolor="none", height=0.65)
                ax.invert_yaxis()
                ax.set_xlabel("Importance"); ax.set_title("Feature Importances (Top 15)")
                for bar, val in zip(bars, fi_df['Importance']):
                    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                            f"{val:.3f}", va='center', color=MUTED, fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Actual vs Predicted + Residuals
        model_ev = st.session_state['model']
        X_te_s   = st.session_state['X_test']
        y_te_s   = st.session_state['y_test']
        y_pr_s   = model_ev.predict(X_te_s)
        residuals = y_te_s - y_pr_s

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="sec-label">Actual vs. Predicted</div>', unsafe_allow_html=True)
            fig, ax = dark_fig(6, 4)
            jitter = np.random.RandomState(0).uniform(-1500, 1500, len(y_te_s))
            ax.scatter(y_te_s + jitter, y_pr_s, alpha=0.18, s=8, color=ORANGE, edgecolors="none")
            mn = min(y_te_s.min(), y_pr_s.min())
            mx = max(y_te_s.max(), y_pr_s.max())
            ax.plot([mn, mx], [mn, mx], color=TEAL, linewidth=1.2, linestyle="--", label="Perfect fit")
            ax.set_xlabel("Actual ($)"); ax.set_ylabel("Predicted ($)")
            ax.set_title("Actual vs. Predicted")
            ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=IRON, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with col_r:
            st.markdown('<div class="sec-label">Residual Distribution</div>', unsafe_allow_html=True)
            fig, ax = dark_fig(6, 4)
            ax.hist(residuals, bins=40, color=TEAL, edgecolor="none", alpha=0.75)
            ax.axvline(0, color=ORANGE, linestyle="--", linewidth=1.2, label="Zero error")
            ax.set_xlabel("Residual ($)"); ax.set_ylabel("Count")
            ax.set_title("Residual Distribution")
            ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=IRON, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Download summary
        summary_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²', 'Model', 'Test Size', 'Train Samples', 'Test Samples'],
            'Value':  [f"${m['mae']:,.0f}", f"${m['rmse']:,.0f}", f"{m['r2']:.4f}",
                       model_choice, f"{test_size*100:.0f}%",
                       f"{len(X_train):,}", f"{len(X_test):,}"]
        })
        st.download_button("⬇  Export Model Summary", data=summary_df.to_csv(index=False).encode(),
                           file_name="model_summary.csv", mime="text/csv")
    else:
        st.markdown("""
        <div style="
          background:#1C2333;border:1px dashed #252D3D;border-radius:10px;
          padding:2rem;text-align:center;
        ">
          <div style="font-size:1.5rem;margin-bottom:0.5rem;opacity:0.4;">◈</div>
          <div style="color:#6B7B94;font-size:0.88rem;">
            Click <b style="color:#E8692A;">Train Model</b> to begin.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 3 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="sec-heading">Single Employee Prediction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("Train the model first (Train Model tab) before making predictions.")
    else:
        model     = st.session_state['model']
        X_columns = st.session_state['X_columns']
        df_ref    = st.session_state.get('df', df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="sec-label">Personal Info</div>', unsafe_allow_html=True)
            age            = st.slider("Age", int(df_ref['age'].min()), int(df_ref['age'].max()), 30)
            gender         = st.radio("Gender", df_ref['gender'].unique().tolist())
            race           = st.selectbox("Race", df_ref['race'].unique().tolist())
            native_country = st.selectbox("Native Country", sorted(df_ref['native-country'].unique().tolist()))

        with col2:
            st.markdown('<div class="sec-label">Education & Work</div>', unsafe_allow_html=True)
            educational_num = st.slider("Education Level (1–16)",
                                        int(df_ref['educational-num'].min()),
                                        int(df_ref['educational-num'].max()), 10,
                                        help="1 = minimal, 16 = Doctorate")
            workclass       = st.selectbox("Work Class", df_ref['workclass'].unique().tolist())
            occupation      = st.selectbox("Occupation", df_ref['occupation'].unique().tolist())
            hours_per_week  = st.slider("Hours / Week",
                                        int(df_ref['hours-per-week'].min()),
                                        int(df_ref['hours-per-week'].max()), 40)

        with col3:
            st.markdown('<div class="sec-label">Household & Capital</div>', unsafe_allow_html=True)
            marital_status = st.selectbox("Marital Status", df_ref['marital-status'].unique().tolist())
            relationship   = st.selectbox("Relationship",   df_ref['relationship'].unique().tolist())
            capital_gain   = st.number_input("Capital Gain ($)", min_value=0,
                                              max_value=int(df_ref['capital-gain'].max()), value=0)
            capital_loss   = st.number_input("Capital Loss ($)", min_value=0,
                                              max_value=int(df_ref['capital-loss'].max()), value=0)
            fnlwgt         = st.number_input("Final Weight (fnlwgt)",
                                              min_value=int(df_ref['fnlwgt'].min()),
                                              max_value=int(df_ref['fnlwgt'].max()), value=200000)

        if st.button("◈  Predict Salary", use_container_width=True):
            new_data = pd.DataFrame([{
                'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt,
                'educational-num': educational_num, 'marital-status': marital_status,
                'occupation': occupation, 'relationship': relationship,
                'race': race, 'gender': gender,
                'capital-gain': capital_gain, 'capital-loss': capital_loss,
                'hours-per-week': hours_per_week, 'native-country': native_country
            }])[X_columns]

            try:
                predicted  = model.predict(new_data)[0]
                low, high  = predicted * 0.85, predicted * 1.15
                nat_avg    = df_ref['income_numeric'].mean()
                pct_above  = (df_ref['income_numeric'] < predicted).mean() * 100
                t_label, t_color, t_bg, t_border = salary_tier(predicted)

                st.markdown(f"""
                <div class="pred-card fade-up" style="margin-top:1.5rem;">
                  <div style="font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;
                               color:#6B7B94;margin-bottom:0.6rem;">Estimated Annual Salary</div>
                  <div class="pred-amount">${predicted:,.0f}</div>
                  {"<div class='pred-range'>Range: $" + f"{low:,.0f} – ${high:,.0f}</div>" if show_confidence else ""}
                  <div style="margin-top:0.9rem;">
                    <span class="tier-pill" style="background:{t_bg};color:{t_color};border:1px solid {t_border};">
                      {t_label}
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Stats row
                ca, cb, cc = st.columns(3)
                ca.metric("Dataset Average", f"${nat_avg:,.0f}")
                cb.metric("Percentile",       f"{pct_above:.0f}th")
                cc.metric("vs. Average",      f"${predicted - nat_avg:+,.0f}")

                # Percentile gauge
                fig, ax = dark_fig(7, 0.75)
                ax.barh([0], [100], color="#151B27", height=0.5, edgecolor="none")
                fill_c = TEAL if pct_above >= 70 else (ORANGE if pct_above >= 40 else "#F5C842")
                ax.barh([0], [pct_above], color=fill_c, height=0.5, edgecolor="none")
                ax.set_xlim(0, 100); ax.set_yticks([])
                ax.set_xlabel("Percentile in Dataset")
                ax.set_title(f"{pct_above:.0f}th percentile", fontsize=9)
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.grid(axis='y', visible=False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                # Save history
                if 'pred_history' not in st.session_state:
                    st.session_state['pred_history'] = []
                st.session_state['pred_history'].append({
                    'Age': age, 'Occupation': occupation, 'Hours/Wk': hours_per_week,
                    'Predicted ($)': int(predicted), 'Tier': t_label,
                    'Percentile': f"{pct_above:.0f}th"
                })

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Prediction history
        if 'pred_history' in st.session_state and len(st.session_state['pred_history']) > 0:
            with st.expander(f"Prediction History — {len(st.session_state['pred_history'])} runs"):
                hist_df = pd.DataFrame(st.session_state['pred_history'])
                st.dataframe(hist_df, use_container_width=True)

                dl_col, clr_col = st.columns(2)
                with dl_col:
                    st.download_button("⬇  Export History",
                                       data=hist_df.to_csv(index=False).encode(),
                                       file_name="prediction_history.csv", mime="text/csv")
                with clr_col:
                    if st.button("Clear History"):
                        st.session_state['pred_history'] = []
                        st.rerun()


# ═══════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ═══════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sec-heading">Batch Prediction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("Train the model first before running batch predictions.")
    else:
        model     = st.session_state['model']
        X_columns = st.session_state['X_columns']

        with st.expander("Expected CSV Format"):
            sample_csv = (
                "age,workclass,fnlwgt,educational-num,marital-status,occupation,"
                "relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country\n"
                "35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States\n"
                "28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States\n"
                "50,Self-emp-inc,180000,14,Married-civ-spouse,Prof-specialty,Wife,Asian-Pac-Islander,Female,10000,0,50,India"
            )
            st.code(sample_csv, language="csv")
            st.download_button("⬇  Download Sample CSV", data=sample_csv,
                               file_name="sample_batch.csv", mime="text/csv")

        batch_file = st.file_uploader("Upload batch CSV", type="csv", key="batch_uploader")

        if batch_file is not None:
            with st.spinner("Running batch predictions…"):
                try:
                    batch_data = pd.read_csv(batch_file)
                    for col in batch_data.select_dtypes(include='object').columns:
                        batch_data[col] = batch_data[col].str.strip()

                    preds = model.predict(batch_data[X_columns])
                    batch_data['Predicted_Salary'] = preds.round(0).astype(int)
                    batch_data['Salary_Tier']      = [salary_tier(p)[0] for p in preds]
                    batch_data['Percentile']       = [
                        f"{(df['income_numeric'] < p).mean()*100:.0f}th"
                        for p in preds
                    ]

                    st.success(f"✓ Predicted salaries for {len(batch_data):,} records.")

                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Average", f"${preds.mean():,.0f}")
                    bc2.metric("Median",  f"${np.median(preds):,.0f}")
                    bc3.metric("Min",     f"${preds.min():,.0f}")
                    bc4.metric("Max",     f"${preds.max():,.0f}")

                    col_left, col_right = st.columns([2, 1])
                    with col_left:
                        fig, ax = dark_fig(7, 3.5)
                        ax.hist(preds, bins=30, color=ORANGE, edgecolor="none", alpha=0.8)
                        ax.axvline(preds.mean(),      color=TEAL,  linewidth=1.2, linestyle="--",
                                   label=f"Mean ${preds.mean():,.0f}")
                        ax.axvline(np.median(preds),  color=MIST,  linewidth=1.2, linestyle=":",
                                   label=f"Median ${np.median(preds):,.0f}")
                        ax.set_xlabel("Predicted Salary ($)"); ax.set_ylabel("Count")
                        ax.set_title("Salary Distribution — Batch Results")
                        ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=IRON, fontsize=9)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                    with col_right:
                        st.markdown('<div class="sec-label">Tier Breakdown</div>', unsafe_allow_html=True)
                        tiers = batch_data['Salary_Tier'].value_counts()
                        for tier, count in tiers.items():
                            pct = count / len(batch_data) * 100
                            t_label, t_color, t_bg, t_border = salary_tier(
                                {'Entry Level': 20000, 'Mid Level': 42000,
                                 'Senior': 65000, 'Executive': 80000}.get(tier, 50000)
                            )
                            st.markdown(f"""
                            <div style="background:#1C2333;border:1px solid #252D3D;border-radius:8px;
                                        padding:0.65rem 0.9rem;margin-bottom:0.45rem;">
                              <div style="display:flex;justify-content:space-between;margin-bottom:0.35rem;">
                                <span style="font-size:0.8rem;color:#C9D3E0;font-weight:500;">{tier}</span>
                                <span style="font-size:0.75rem;color:{t_color};">{count} · {pct:.0f}%</span>
                              </div>
                              <div class="prog-wrap">
                                <div class="prog-bar" style="width:{pct}%;background:{t_color};"></div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.dataframe(batch_data, use_container_width=True)
                    st.download_button("⬇  Download Full Predictions",
                                       data=batch_data.to_csv(index=False).encode('utf-8'),
                                       file_name='predicted_salaries.csv', mime='text/csv')

                except KeyError as ke:
                    st.error(f"Missing column: {ke}. Ensure all required columns are present.")
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")


# ═══════════════════════════════════════════════
# TAB 5 — SALARY INSIGHTS
# ═══════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="sec-heading">Salary Insights</div>', unsafe_allow_html=True)

    if 'income_numeric' not in df.columns:
        st.info("Load a dataset with income data to see insights.")
    else:
        high_df = df[df['income_numeric'] == 75000]
        low_df  = df[df['income_numeric'] == 25000]

        top_occupation = high_df['occupation'].mode()[0]  if 'occupation' in df.columns else "N/A"
        top_workclass  = high_df['workclass'].mode()[0]   if 'workclass' in df.columns else "N/A"
        avg_edu_high   = high_df['educational-num'].mean() if 'educational-num' in df.columns else 0
        avg_edu_low    = low_df['educational-num'].mean()  if 'educational-num' in df.columns else 0
        avg_hrs_high   = high_df['hours-per-week'].mean()  if 'hours-per-week' in df.columns else 0
        avg_hrs_low    = low_df['hours-per-week'].mean()   if 'hours-per-week' in df.columns else 0

        insights = [
            ("💼", "Top High-Earning Occupation",
             f"<b style='color:#F0F4F8'>{top_occupation}</b> is the most common occupation among high earners."),
            ("🏢", "Dominant Work Class",
             f"High earners are most frequently in <b style='color:#F0F4F8'>{top_workclass}</b>."),
            ("🎓", "Education Premium",
             f"High earners average <b style='color:#E8692A'>{avg_edu_high:.1f}/16</b> education level vs "
             f"<b style='color:#6B7B94'>{avg_edu_low:.1f}/16</b> for low earners — "
             f"a <b style='color:#2ABFBF'>+{avg_edu_high - avg_edu_low:.1f} level gap</b>."),
            ("⏱", "Hours Worked",
             f"High earners work <b style='color:#E8692A'>{avg_hrs_high:.0f} hrs/week</b> on average vs "
             f"<b style='color:#6B7B94'>{avg_hrs_low:.0f} hrs/week</b> for low earners."),
            ("📊", "Dataset Composition",
             f"<b style='color:#2ABFBF'>{pct_high:.1f}%</b> high earners ({high_earners:,} records) vs "
             f"<b style='color:#6B7B94'>{100-pct_high:.1f}%</b> low earners."),
        ]

        for icon, title, body in insights:
            st.markdown(f"""
            <div class="insight-row">
              <div class="insight-icon">{icon}</div>
              <div>
                <div class="insight-title">{title}</div>
                <div class="insight-body">{body}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Salary Ranking by Category</div>', unsafe_allow_html=True)

        rank_col = st.selectbox("Rank by",
            [c for c in ['occupation', 'workclass', 'marital-status', 'race', 'gender'] if c in df.columns])

        fig, ax = dark_fig(10, max(4, df[rank_col].nunique() * 0.4))
        grp = df.groupby(rank_col)['income_numeric'].agg(['mean', 'std', 'count']).sort_values('mean')
        mean_val = df['income_numeric'].mean()
        bar_clrs = [TEAL if v >= mean_val else ORANGE for v in grp['mean'].values]
        bars = ax.barh(grp.index, grp['mean'], color=bar_clrs, edgecolor="none", height=0.65)
        ax.axvline(mean_val, color=MIST, linestyle='--', linewidth=0.8, alpha=0.4, label="Dataset mean")
        ax.set_xlabel("Average Salary ($)")
        ax.set_title(f"Average Salary by {rank_col.replace('-', ' ').title()}")
        ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=IRON, fontsize=9)
        for bar, (_, row) in zip(bars, grp.iterrows()):
            ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
                    f"${row['mean']:,.0f}  n={int(row['count'])}",
                    va='center', color=MUTED, fontsize=7.5)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════
# TAB 6 — MODEL COMPARISON
# ═══════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="sec-heading">Compare Models</div>', unsafe_allow_html=True)

    if not enable_comparison:
        st.markdown("""
        <div style="
          background:#1C2333;border:1px dashed #252D3D;border-radius:10px;
          padding:2rem;text-align:center;
        ">
          <div style="font-size:1.4rem;margin-bottom:0.5rem;opacity:0.4;">⚖</div>
          <div style="color:#6B7B94;font-size:0.88rem;">
            Enable <b style="color:#E8692A;">Model Comparison Mode</b> in the sidebar to compare all algorithms.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Trains all 5 models sequentially — takes ~1–2 minutes.")

        if st.button("▶  Run Full Comparison", use_container_width=True):
            X_cmp = df.drop(columns=[c for c in ['income', 'income_numeric'] if c in df.columns])
            y_cmp = df['income_numeric']

            cat_f = X_cmp.select_dtypes(include='object').columns.tolist()
            num_f = X_cmp.select_dtypes(include=np.number).columns.tolist()
            prep  = ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_f),
                ('num', 'passthrough', num_f)
            ])

            X_tr, X_te, y_tr, y_te = train_test_split(X_cmp, y_cmp, test_size=test_size, random_state=42)

            results  = []
            progress = st.progress(0)
            for i, mname in enumerate(MODELS):
                with st.spinner(f"Training {mname}…"):
                    m = train_model(X_tr, y_tr, prep, model_type=mname)
                    yp = m.predict(X_te)
                    results.append({
                        'Model':    mname,
                        'MAE ($)':  int(mean_absolute_error(y_te, yp)),
                        'RMSE ($)': int(np.sqrt(mean_squared_error(y_te, yp))),
                        'R²':       round(r2_score(y_te, yp), 4),
                    })
                progress.progress((i + 1) / len(MODELS))

            progress.empty()
            results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
            st.session_state['comparison_results'] = results_df
            st.success("✓ Comparison complete.")

        if 'comparison_results' in st.session_state:
            cdf  = st.session_state['comparison_results']
            best = cdf.iloc[0]['Model']

            # Table
            rows_html = ""
            for _, row in cdf.iterrows():
                is_best = row['Model'] == best
                badge_html = f'<span class="badge badge-teal" style="margin-left:0.4rem;">Best</span>' if is_best else ''
                rows_html += f"""
                <tr>
                  <td style="color:{'#2ABFBF' if is_best else '#C9D3E0'};font-weight:{'600' if is_best else '400'};">
                    {MODEL_META[row['Model']][0]} {row['Model']}{badge_html}
                  </td>
                  <td>${int(row['MAE ($)']):,}</td>
                  <td>${int(row['RMSE ($)']):,}</td>
                  <td style="color:{'#2ABFBF' if is_best else '#C9D3E0'};">{row['R²']:.4f}</td>
                </tr>
                """

            st.markdown(f"""
            <table class="cmp-table" style="width:100%;background:#1C2333;border-radius:10px;overflow:hidden;border:1px solid #252D3D;">
              <thead>
                <tr>
                  <th>Model</th><th>MAE ($)</th><th>RMSE ($)</th><th>R² Score</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

            # R² comparison chart
            st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
            fig, ax = dark_fig(9, 3.5)
            bar_clrs = [TEAL if m == best else ORANGE for m in cdf['Model']]
            bars = ax.barh(cdf['Model'], cdf['R²'], color=bar_clrs, edgecolor="none", height=0.6)
            ax.set_xlim(0, 1)
            ax.set_xlabel("R² Score (higher = better)")
            ax.set_title("Model R² Score Comparison")
            for bar, val in zip(bars, cdf['R²']):
                ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va='center', color=IRON, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.download_button("⬇  Export Comparison",
                               data=cdf.to_csv(index=False).encode(),
                               file_name="model_comparison.csv", mime="text/csv")
