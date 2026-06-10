import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
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
    page_title="SalaryIQ — ML Salary Intelligence",
    page_icon="💡",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# HIDE STREAMLIT TOOLBAR / GITHUB FORK BUTTON
# ──────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"]   { display: none !important; }
[data-testid="stDecoration"]{ display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TAILWIND + GSAP + CUSTOM CSS/JS
# ──────────────────────────────────────────────
st.markdown("""
<!-- Tailwind CSS CDN -->
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    theme: {
      extend: {
        colors: {
          midnight: '#0b0f1a',
          navy: '#111827',
          surface: '#161d2e',
          card: '#1c2537',
          border: '#1e2d45',
          violet: { DEFAULT: '#7c3aed', light: '#a78bfa', dark: '#5b21b6' },
          cyan: { DEFAULT: '#06b6d4', light: '#67e8f9' },
          emerald: { DEFAULT: '#10b981', light: '#6ee7b7' },
          amber: { DEFAULT: '#f59e0b', light: '#fcd34d' },
          rose: { DEFAULT: '#f43f5e', light: '#fda4af' },
        },
        fontFamily: {
          display: ['Space Grotesk', 'sans-serif'],
          body: ['Inter', 'sans-serif'],
          mono: ['JetBrains Mono', 'monospace'],
        },
        boxShadow: {
          glow: '0 0 20px rgba(124,58,237,0.35)',
          'glow-cyan': '0 0 20px rgba(6,182,212,0.3)',
          'glow-emerald': '0 0 20px rgba(16,185,129,0.3)',
        },
        animation: {
          'float': 'float 6s ease-in-out infinite',
          'pulse-slow': 'pulse 4s ease-in-out infinite',
          'shimmer': 'shimmer 2s linear infinite',
        },
      }
    }
  }
</script>

<!-- Google Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<!-- GSAP CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>

<style>
/* ── Base reset ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: #0b0f1a !important;
  color: #e2e8f0 !important;
}
.stApp { background: #0b0f1a !important; }
.main .block-container { padding-top: 1.5rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #111827 !important;
  border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Inputs ── */
.stSelectbox select, .stTextInput input, .stNumberInput input {
  background: #1c2537 !important;
  border: 1px solid #1e2d45 !important;
  color: #e2e8f0 !important;
  border-radius: 10px !important;
}
.stSlider > div > div { color: #a78bfa !important; }
.stRadio label { color: #94a3b8 !important; }
.stCheckbox label { color: #94a3b8 !important; }
.stFileUploader { background: #161d2e !important; border: 1px dashed #1e2d45 !important; border-radius: 12px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: #161d2e;
  border-radius: 14px;
  padding: 5px;
  gap: 4px;
  border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  color: #64748b !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
  color: #fff !important;
  box-shadow: 0 0 16px rgba(124,58,237,0.4) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.6rem 2rem !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.02em !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: #1c2537;
  border: 1px solid #1e2d45;
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
}
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── Expandable ── */
.streamlit-expanderHeader { background: #161d2e !important; border-radius: 10px !important; color: #a78bfa !important; }
.streamlit-expanderContent { background: #111827 !important; border: 1px solid #1e2d45 !important; border-radius: 0 0 10px 10px !important; }

/* ── Info / Warning / Success ── */
.stAlert { border-radius: 12px !important; }

/* ── Hero animated gradient ── */
@keyframes gradShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.hero-animate {
  background: linear-gradient(-45deg, #1e1b4b, #312e81, #1e1b4b, #2d1b69, #0f1117);
  background-size: 400% 400%;
  animation: gradShift 12s ease infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-10px); }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
.shimmer-text {
  background: linear-gradient(90deg, #a78bfa, #06b6d4, #a78bfa, #7c3aed);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 3s linear infinite;
}

/* ── Orbs ── */
.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(60px);
  opacity: 0.35;
  pointer-events: none;
}
.orb-1 { width: 300px; height: 300px; background: #7c3aed; top: -80px; right: -60px; animation: float 8s ease-in-out infinite; }
.orb-2 { width: 200px; height: 200px; background: #06b6d4; bottom: -60px; left: 20%; animation: float 10s ease-in-out infinite 2s; }

/* ── Card hover ── */
.hover-card {
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.hover-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 35px rgba(124,58,237,0.25);
  border-color: #7c3aed !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }

/* ── Prediction card ── */
@keyframes predReveal {
  from { opacity: 0; transform: scale(0.92) translateY(10px); }
  to   { opacity: 1; transform: scale(1) translateY(0); }
}
.pred-reveal { animation: predReveal 0.5s cubic-bezier(0.22, 1, 0.36, 1) forwards; }

/* ── Counter animation ── */
.counter { font-variant-numeric: tabular-nums; }

/* ── Tooltip ── */
.tooltip-wrapper { position: relative; display: inline-block; }
.tooltip-wrapper .tooltip-text {
  visibility: hidden; opacity: 0;
  background: #1c2537; border: 1px solid #1e2d45;
  color: #94a3b8; font-size: 0.72rem; padding: 6px 10px;
  border-radius: 8px; white-space: nowrap;
  position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%);
  transition: opacity 0.2s ease;
}
.tooltip-wrapper:hover .tooltip-text { visibility: visible; opacity: 1; }

/* ── Progress bar animation ── */
@keyframes barFill {
  from { width: 0%; }
}
.bar-animate { animation: barFill 1.2s cubic-bezier(0.22, 1, 0.36, 1) forwards; }

/* ── Pulse dot ── */
@keyframes pulseDot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(1.3); }
}
.pulse-dot { animation: pulseDot 2s ease-in-out infinite; }

/* ── Section divider ── */
.section-rule {
  height: 1px;
  background: linear-gradient(90deg, transparent, #1e2d45, transparent);
  margin: 2rem 0;
}

/* ── Tag / pill ── */
.tag {
  display: inline-flex; align-items: center; gap: 0.3rem;
  background: #1c2537; border: 1px solid #1e2d45;
  border-radius: 999px; padding: 0.25rem 0.85rem;
  font-size: 0.72rem; color: #64748b;
}
.tag-violet { border-color: rgba(124,58,237,0.4); color: #a78bfa; background: rgba(124,58,237,0.1); }
.tag-cyan   { border-color: rgba(6,182,212,0.4); color: #67e8f9; background: rgba(6,182,212,0.1); }
.tag-emerald{ border-color: rgba(16,185,129,0.4); color: #6ee7b7; background: rgba(16,185,129,0.1); }
.tag-amber  { border-color: rgba(245,158,11,0.4); color: #fcd34d; background: rgba(245,158,11,0.1); }

/* ── Comparison table ── */
.compare-table th { color: #a78bfa; font-size: 0.78rem; letter-spacing: 0.07em; text-transform: uppercase; padding: 0.75rem 1rem; border-bottom: 1px solid #1e2d45; }
.compare-table td { padding: 0.65rem 1rem; border-bottom: 1px solid #1a2234; font-size: 0.85rem; color: #cbd5e1; }
.compare-table tr:last-child td { border-bottom: none; }
.compare-table tr:hover td { background: rgba(124,58,237,0.06); }

/* ── GSAP fade-in targets ── */
.gsap-fadein { opacity: 0; }

/* ── Stepper ── */
.step-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg, #7c3aed, #6d28d9);
  color: #fff; font-weight: 700; font-size: 0.8rem;
  flex-shrink: 0; box-shadow: 0 0 12px rgba(124,58,237,0.4);
}
.section-header {
  display: flex; align-items: center; gap: 0.65rem;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.15rem; font-weight: 700; color: #a78bfa;
  border-bottom: 1px solid #1e2d45; padding-bottom: 0.55rem; margin: 2rem 0 1rem 0;
}

/* ── Model score indicator ── */
.score-ring {
  display: inline-flex; align-items: center; justify-content: center;
  width: 80px; height: 80px; border-radius: 50%;
  border: 3px solid #7c3aed;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.2rem; font-weight: 700; color: #a78bfa;
  box-shadow: 0 0 20px rgba(124,58,237,0.3);
}

/* ── Salary insight badge ── */
.insight-badge {
  display: flex; align-items: flex-start; gap: 0.75rem;
  background: #1c2537; border: 1px solid #1e2d45; border-radius: 12px;
  padding: 1rem 1.2rem; margin-bottom: 0.75rem;
  transition: border-color 0.2s;
}
.insight-badge:hover { border-color: #7c3aed; }
.insight-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 0.1rem; }
.insight-text { font-size: 0.85rem; color: #94a3b8; line-height: 1.55; }
.insight-title { font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; font-size: 0.875rem; }
</style>

<!-- GSAP init script (runs after DOM) -->
<script>
window.addEventListener('load', function() {
  if (typeof gsap !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger);

    // Fade in all hero children on load
    gsap.from('.hero-child', {
      opacity: 0, y: 30, stagger: 0.12, duration: 0.8,
      ease: 'power3.out', delay: 0.1
    });

    // Scroll-triggered fade for metric cards
    gsap.utils.toArray('.gsap-fadein').forEach(el => {
      gsap.to(el, {
        opacity: 1, y: 0,
        scrollTrigger: { trigger: el, start: 'top 88%', toggleActions: 'play none none none' },
        duration: 0.65, ease: 'power2.out'
      });
    });

    // Counter animate for metric cards
    document.querySelectorAll('[data-count]').forEach(el => {
      const target = parseFloat(el.dataset.count);
      const prefix = el.dataset.prefix || '';
      const suffix = el.dataset.suffix || '';
      const decimals = el.dataset.decimals || 0;
      gsap.from({ val: 0 }, {
        val: target, duration: 1.6, ease: 'power2.out',
        onUpdate: function() { el.textContent = prefix + this.targets()[0].val.toFixed(decimals) + suffix; },
        delay: 0.3
      });
    });
  }
});
</script>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ──────────────────────────────────────────────
BG      = "#0b0f1a"
SURFACE = "#161d2e"
CARD    = "#1c2537"
ACCENT  = "#7c3aed"
ACCENT2 = "#06b6d4"
ACCENT3 = "#10b981"
ACCENT4 = "#f59e0b"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

def dark_fig(w=10, h=4, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax_list = [axes] if nrows * ncols == 1 else axes.flatten()
    for ax in ax_list:
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d45")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
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
        cat_features  = preprocessor.transformers_[0][2]
        num_features  = preprocessor.transformers_[1][2]
        ohe = preprocessor.transformers_[0][1]
        cat_names = ohe.get_feature_names_out(cat_features).tolist()
        all_names = cat_names + list(num_features)
        fi_df = pd.DataFrame({'Feature': all_names, 'Importance': reg.feature_importances_})
        return fi_df.sort_values('Importance', ascending=False).head(15)
    except Exception:
        return None


def salary_tier(val):
    if val < 30000:   return "🔴 Entry Level", "#f43f5e"
    elif val < 55000: return "🟡 Mid Level", "#f59e0b"
    elif val < 75000: return "🟢 Senior Level", "#10b981"
    else:             return "💜 Executive", "#7c3aed"


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem; display:flex; align-items:center; gap:0.5rem;">
      <div style="width:8px;height:8px;border-radius:50%;background:#7c3aed;" class="pulse-dot"></div>
      <span style="font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:#e2e8f0;">SalaryIQ</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Model Config")

    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Extra Trees", "Ridge Regression", "Lasso Regression"],
        help="Tree-based models give feature importances."
    )

    test_size = st.slider("Test split (%)", 10, 40, 20, step=5) / 100

    st.markdown("### 🎛️ Advanced")
    show_confidence = st.checkbox("Show confidence intervals", value=True)
    enable_comparison = st.checkbox("Enable model comparison mode", value=False)
    show_shap_proxy = st.checkbox("Show feature attribution chart", value=True)

    st.divider()

    st.markdown("""
    <div style="font-size:0.78rem;color:#64748b;line-height:1.7;">
      <b style="color:#a78bfa;">How to use</b><br>
      1. Upload <code>adult.csv</code><br>
      2. Explore data in EDA tab<br>
      3. Train your model<br>
      4. Predict single or batch<br>
      5. Export results
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="tag tag-violet">adult.csv · UCI Census Income</span>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-animate" style="
  border: 1px solid rgba(124,58,237,0.35);
  border-radius: 20px; padding: 2.8rem 2.4rem;
  margin-bottom: 2rem; position: relative; overflow: hidden;
">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="hero-child" style="position:relative;z-index:1;">
    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;">
      <span style="font-size:2rem;">💡</span>
      <span class="shimmer-text" style="font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:700;line-height:1;">SalaryIQ</span>
    </div>
    <p style="color:#a5b4fc;font-size:1.05rem;max-width:560px;line-height:1.65;margin:0 0 1.2rem 0;">
      Machine-learning salary intelligence built on census data. Upload, train in one click, and uncover pay insights instantly.
    </p>
    <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
      <span class="tag tag-violet">5 ML Models</span>
      <span class="tag tag-cyan">EDA + Viz</span>
      <span class="tag tag-emerald">Batch Export</span>
      <span class="tag tag-amber">Salary Insights</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# IN-PAGE MODEL SWITCHER NAVBAR
# ──────────────────────────────────────────────
st.markdown("""
<style>
.model-btn-active {
  background: linear-gradient(135deg,#7c3aed,#6d28d9) !important;
  color: #fff !important;
  border: 1px solid #7c3aed !important;
  box-shadow: 0 0 14px rgba(124,58,237,0.4) !important;
}
/* Override default button style for navbar */
div[data-testid="column"] .stButton > button {
  border-radius: 999px !important;
  padding: 0.38rem 0.6rem !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  background: #1c2537 !important;
  color: #94a3b8 !important;
  border: 1px solid #1e2d45 !important;
  box-shadow: none !important;
  letter-spacing: 0.01em !important;
}
div[data-testid="column"] .stButton > button:hover {
  border-color: #7c3aed !important;
  color: #a78bfa !important;
  background: rgba(124,58,237,0.12) !important;
  transform: none !important;
}
</style>
""", unsafe_allow_html=True)

MODELS = ["Random Forest", "Gradient Boosting", "Extra Trees", "Ridge Regression", "Lasso Regression"]
MODEL_ICONS = {
    "Random Forest":     "🌲",
    "Gradient Boosting": "⚡",
    "Extra Trees":       "🌳",
    "Ridge Regression":  "📐",
    "Lasso Regression":  "🔗",
}
MODEL_DESCRIPTIONS = {
    "Random Forest":     "Ensemble · Feature importances · Best accuracy",
    "Gradient Boosting": "Boosted ensemble · Handles nonlinearity well",
    "Extra Trees":       "Faster than RF · Low variance",
    "Ridge Regression":  "Linear · L2 regularisation · Fast",
    "Lasso Regression":  "Linear · L1 · Sparse features",
}

if 'navbar_model' not in st.session_state:
    st.session_state['navbar_model'] = model_choice

st.markdown('<div style="margin-bottom:0.5rem;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:#475569;">🎛️ Select Model</div>', unsafe_allow_html=True)

nav_cols = st.columns(len(MODELS))
for i, m in enumerate(MODELS):
    with nav_cols[i]:
        if st.button(f"{MODEL_ICONS[m]} {m}", key=f"nav_model_{i}",
                     use_container_width=True, help=MODEL_DESCRIPTIONS[m]):
            st.session_state['navbar_model'] = m
            st.rerun()

active_model = st.session_state['navbar_model']

# Highlight active button via JS
active_idx = MODELS.index(active_model)
st.markdown(f"""
<script>
(function() {{
  const attempt = () => {{
    const btns = document.querySelectorAll('[data-testid="column"] .stButton > button');
    if (btns.length < {len(MODELS)}) {{ setTimeout(attempt, 150); return; }}
    btns.forEach((b, i) => {{
      b.style.background = '';
      b.style.color = '';
      b.style.borderColor = '';
      b.style.boxShadow = '';
    }});
    const active = btns[{active_idx}];
    if (active) {{
      active.style.background = 'linear-gradient(135deg,#7c3aed,#6d28d9)';
      active.style.color = '#fff';
      active.style.borderColor = '#7c3aed';
      active.style.boxShadow = '0 0 14px rgba(124,58,237,0.45)';
    }}
  }};
  attempt();
}})();
</script>
""", unsafe_allow_html=True)

# Active model info bar
st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;
  background:#1c2537;border:1px solid rgba(124,58,237,0.3);
  border-radius:12px;padding:0.75rem 1.2rem;margin:0.75rem 0 1.5rem 0;">
  <span style="font-size:1.4rem;">{MODEL_ICONS[active_model]}</span>
  <div>
    <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
      color:#a78bfa;font-size:0.9rem;">{active_model}</div>
    <div style="font-size:0.78rem;color:#64748b;margin-top:0.1rem;">
      {MODEL_DESCRIPTIONS[active_model]}
    </div>
  </div>
  <div style="margin-left:auto;">
    <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
      border-radius:999px;padding:3px 12px;font-size:0.72rem;color:#a78bfa;">● Active</span>
  </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,#1e2d45,transparent);margin-bottom:1.5rem;"></div>
""", unsafe_allow_html=True)

# Override model_choice with navbar selection
model_choice = active_model

# ──────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="step-num">1</span> Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload adult.csv", type="csv", label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="background:#1c2537;border:1px dashed #1e2d45;border-radius:14px;padding:2rem;text-align:center;color:#64748b;">
      <div style="font-size:2.5rem;margin-bottom:0.8rem;">📂</div>
      <div style="font-size:0.95rem;">Drop your <code style="background:#111827;padding:2px 7px;border-radius:5px;color:#a78bfa;">adult.csv</code> file above to get started</div>
      <div style="font-size:0.78rem;margin-top:0.5rem;color:#475569;">UCI Census Income dataset · 14 features · ~48,000 records</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading and preprocessing data…"):
    df = load_and_preprocess_data(uploaded_file)

if df is None:
    st.error("Could not process the file. Please check its format.")
    st.stop()

n_rows, n_cols = df.shape
high_earners = (df['income_numeric'] == 75000).sum() if 'income_numeric' in df.columns else 0
pct_high = high_earners / n_rows * 100
avg_salary = df['income_numeric'].mean() if 'income_numeric' in df.columns else 0

# ── Dataset stat cards ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{n_rows:,}", "after cleaning")
c2.metric("Features", f"{n_cols - 2}", "input columns")
c3.metric("High Earners (>50K)", f"{pct_high:.1f}%", f"{high_earners:,} records")
c4.metric("Algorithm", model_choice, "selected")

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_eda, tab_train, tab_predict, tab_batch, tab_insights, tab_compare = st.tabs([
    "📊 Data Explorer",
    "🤖 Train Model",
    "🔍 Single Prediction",
    "📦 Batch Prediction",
    "💡 Salary Insights",
    "⚖️ Compare Models",
])


# ═══════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header"><span class="step-num">2</span> Explore Your Data</div>', unsafe_allow_html=True)

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["Preview & Stats", "Distributions", "Correlation", "Income Breakdown"])

    with eda_tab1:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown("**Dataset Preview** (first 50 rows)")
            st.dataframe(df.drop(columns=['income_numeric'], errors='ignore').head(50), use_container_width=True)
        with col_b:
            st.markdown("**Quick Stats**")
            num_df = df.select_dtypes(include=np.number)
            for col in num_df.columns[:6]:
                st.markdown(f"""
                <div class="insight-badge" style="padding:0.65rem 0.9rem;margin-bottom:0.4rem;">
                  <div>
                    <div class="insight-title" style="font-size:0.78rem;">{col}</div>
                    <div class="insight-text" style="font-size:0.72rem;">
                      μ {num_df[col].mean():,.1f} · σ {num_df[col].std():,.1f}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with eda_tab2:
        col_sel, col_plot = st.columns([1, 3])
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        with col_sel:
            chart_type = st.radio("Chart type", ["Numerical histogram", "Categorical bar", "Box plot"])
            if chart_type in ["Numerical histogram", "Box plot"]:
                col_to_plot = st.selectbox("Column", num_cols)
                color_by_income = st.checkbox("Split by income group", value=True)
            else:
                col_to_plot = st.selectbox("Column", cat_cols)
                color_by_income = False

        with col_plot:
            if chart_type == "Numerical histogram":
                fig, ax = dark_fig(9, 4)
                if color_by_income and 'income' in df.columns:
                    for grp, col in zip(df['income'].unique(), [ACCENT, ACCENT2]):
                        vals = df.loc[df['income'] == grp, col_to_plot].dropna()
                        ax.hist(vals, bins=40, alpha=0.6, color=col, label=grp, edgecolor="none")
                    ax.legend(facecolor=SURFACE, edgecolor="#1e2d45", labelcolor=TEXT, fontsize=9)
                else:
                    ax.hist(df[col_to_plot].dropna(), bins=40, color=ACCENT, edgecolor="none", alpha=0.85)
                ax.set_xlabel(col_to_plot); ax.set_ylabel("Count")
                ax.set_title(f"Distribution of {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_type == "Box plot":
                fig, ax = dark_fig(9, 4)
                if color_by_income and 'income' in df.columns:
                    groups = [df.loc[df['income'] == g, col_to_plot].dropna() for g in df['income'].unique()]
                    labels = df['income'].unique().tolist()
                    bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=True,
                                    medianprops=dict(color=ACCENT2, linewidth=2))
                    colors_bp = [ACCENT, ACCENT3]
                    for patch, c in zip(bp['boxes'], colors_bp):
                        patch.set_facecolor(c); patch.set_alpha(0.5)
                    for element in ['whiskers','caps','fliers']:
                        for item in bp[element]: item.set_color(MUTED)
                else:
                    bp = ax.boxplot(df[col_to_plot].dropna(), patch_artist=True, notch=True,
                                    medianprops=dict(color=ACCENT2, linewidth=2))
                    bp['boxes'][0].set_facecolor(ACCENT); bp['boxes'][0].set_alpha(0.5)
                ax.set_ylabel(col_to_plot); ax.set_title(f"Box Plot — {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            else:
                counts = df[col_to_plot].value_counts()
                fig, ax = dark_fig(9, max(3, len(counts) * 0.45))
                colors = plt.cm.get_cmap('Purples')(np.linspace(0.35, 0.9, len(counts)))
                bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none")
                ax.set_xlabel("Count"); ax.set_title(f"Counts by {col_to_plot}"); ax.invert_yaxis()
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_width() + counts.values.max() * 0.01,
                            bar.get_y() + bar.get_height()/2,
                            f"{val:,}", va='center', color=MUTED, fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with eda_tab3:
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        n = len(corr)
        fig, ax = dark_fig(8, 6)
        im = ax.imshow(corr.values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center',
                        color='white' if abs(corr.values[i, j]) > 0.5 else TEXT, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03)
        cbar.ax.tick_params(colors=MUTED)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with eda_tab4:
        # ── New: income breakdown charts ──
        if 'income' in df.columns and 'occupation' in df.columns:
            fig, axes = dark_fig(12, 9, nrows=2, ncols=2)
            ax1, ax2, ax3, ax4 = axes.flatten()

            # Occupation income split
            occ_income = df.groupby('occupation')['income_numeric'].mean().sort_values()
            colors_occ = [ACCENT3 if v >= 50000 else ACCENT for v in occ_income.values]
            ax1.barh(occ_income.index, occ_income.values, color=colors_occ, edgecolor="none")
            ax1.set_xlabel("Avg Income ($)")
            ax1.set_title("Avg Income by Occupation")
            ax1.axvline(occ_income.mean(), color=ACCENT2, linestyle='--', linewidth=1, label='Mean')
            ax1.legend(facecolor=SURFACE, edgecolor="#1e2d45", labelcolor=TEXT, fontsize=8)

            # Age vs income scatter
            sample = df.sample(min(2000, len(df)), random_state=42)
            colors_scatter = [ACCENT if v == 75000 else ACCENT2 for v in sample['income_numeric']]
            ax2.scatter(sample['age'], sample['hours-per-week'], alpha=0.25, s=10, c=colors_scatter, edgecolors='none')
            ax2.set_xlabel("Age"); ax2.set_ylabel("Hours / Week")
            ax2.set_title("Age vs. Hours (colour = income tier)")
            p1 = mpatches.Patch(color=ACCENT, label='>50K')
            p2 = mpatches.Patch(color=ACCENT2, label='≤50K')
            ax2.legend(handles=[p1, p2], facecolor=SURFACE, edgecolor="#1e2d45", labelcolor=TEXT, fontsize=8)

            # Education vs income
            if 'educational-num' in df.columns:
                edu_income = df.groupby('educational-num')['income_numeric'].mean()
                ax3.bar(edu_income.index, edu_income.values,
                        color=plt.cm.get_cmap('Purples')(np.linspace(0.35, 0.9, len(edu_income))),
                        edgecolor="none")
                ax3.set_xlabel("Education Level (1–16)"); ax3.set_ylabel("Avg Income ($)")
                ax3.set_title("Education Level vs. Avg Income")

            # Gender pay gap bar
            if 'gender' in df.columns:
                gender_income = df.groupby('gender')['income_numeric'].mean()
                bars_g = ax4.bar(gender_income.index, gender_income.values,
                                 color=[ACCENT, ACCENT2], edgecolor="none", width=0.4)
                ax4.set_ylabel("Avg Income ($)"); ax4.set_title("Income by Gender")
                for bar in bars_g:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                             f"${bar.get_height():,.0f}", ha='center', va='bottom', color=TEXT, fontsize=9)

            plt.tight_layout(pad=2.0); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Income breakdown charts require income and occupation columns.")


# ═══════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ═══════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="section-header"><span class="step-num">3</span> Train the ML Model</div>', unsafe_allow_html=True)

    X = df.drop(columns=[c for c in ['income', 'income_numeric'] if c in df.columns])
    y = df['income_numeric']

    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numerical_features   = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor_transformer = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    col_tr, col_te = st.columns(2)
    col_tr.metric("Training samples", f"{len(X_train):,}")
    col_te.metric("Testing samples",  f"{len(X_test):,}")

    st.markdown(f"""
    <div style="background:#1c2537;border:1px solid #1e2d45;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
      <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.08em;">Selected model</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#a78bfa;">{model_choice}</div>
      <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">
        {len(categorical_features)} categorical · {len(numerical_features)} numerical features · {test_size*100:.0f}% test split
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Train Model Now", use_container_width=True):
        with st.spinner(f"Training {model_choice}… this may take up to 30 s"):
            model = train_model(X_train, y_train, preprocessor_transformer, model_type=model_choice)

        st.session_state['model']     = model
        st.session_state['X_columns'] = X.columns.tolist()
        st.session_state['df']        = df
        st.session_state['X_test']    = X_test
        st.session_state['y_test']    = y_test

        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        st.session_state['metrics'] = dict(mae=mae, rmse=rmse, r2=r2)
        st.success(f"✅ {model_choice} trained successfully!")

    if 'metrics' in st.session_state:
        m = st.session_state['metrics']

        # Model metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",      f"${m['mae']:,.0f}",   "avg absolute error")
        c2.metric("RMSE",     f"${m['rmse']:,.0f}",  "root mean squared")
        c3.metric("R² Score", f"{m['r2']:.3f}",      "1.0 = perfect fit")

        # Animated score ring
        r2_pct = int(m['r2'] * 100)
        ring_color = "#10b981" if r2_pct >= 80 else ("#f59e0b" if r2_pct >= 60 else "#f43f5e")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1.5rem;margin:1.2rem 0;
                    background:#1c2537;border:1px solid #1e2d45;border-radius:14px;padding:1.2rem 1.5rem;">
          <div style="width:80px;height:80px;border-radius:50%;border:3px solid {ring_color};
                      display:flex;align-items:center;justify-content:center;
                      font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;color:{ring_color};
                      box-shadow:0 0 20px {ring_color}44;flex-shrink:0;">
            {r2_pct}%
          </div>
          <div>
            <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.25rem;">Model Fit Score</div>
            <div style="font-size:0.82rem;color:#64748b;">
              {"Excellent fit — model explains most variance." if r2_pct >= 80 else
               "Good fit — reasonable predictive power." if r2_pct >= 60 else
               "Moderate fit — consider a different algorithm."}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance
        fi_df = get_feature_importance(st.session_state['model'], X)
        if fi_df is not None and show_shap_proxy:
            st.markdown("**Top 15 Feature Importances**")
            fig, ax = dark_fig(10, 5)
            cmap_colors = plt.cm.get_cmap('Purples')(np.linspace(0.35, 0.95, len(fi_df)))
            bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                           color=cmap_colors[::-1], edgecolor="none")
            ax.invert_yaxis()
            ax.set_xlabel("Importance"); ax.set_title("Feature Importances (Top 15)")
            for bar, val in zip(bars, fi_df['Importance']):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va='center', color=MUTED, fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Actual vs Predicted + Residuals side-by-side (NEW)
        model_eval = st.session_state['model']
        X_test_s   = st.session_state['X_test']
        y_test_s   = st.session_state['y_test']
        y_pred_s   = model_eval.predict(X_test_s)
        residuals  = y_test_s - y_pred_s

        col_left, col_right = st.columns(2)
        with col_left:
            fig, ax = dark_fig(6, 4)
            jitter = np.random.RandomState(0).uniform(-1500, 1500, len(y_test_s))
            ax.scatter(y_test_s + jitter, y_pred_s, alpha=0.2, s=10, color=ACCENT, edgecolors="none")
            mn = min(y_test_s.min(), y_pred_s.min())
            mx = max(y_test_s.max(), y_pred_s.max())
            ax.plot([mn, mx], [mn, mx], color="#a78bfa", linewidth=1.5, linestyle="--", label="Perfect fit")
            ax.set_xlabel("Actual Salary ($)"); ax.set_ylabel("Predicted Salary ($)")
            ax.set_title("Actual vs. Predicted")
            ax.legend(facecolor=SURFACE, edgecolor="#1e2d45", labelcolor=TEXT, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with col_right:
            fig, ax = dark_fig(6, 4)
            ax.hist(residuals, bins=40, color=ACCENT2, edgecolor="none", alpha=0.8)
            ax.axvline(0, color="#a78bfa", linestyle="--", linewidth=1.5, label="Zero error")
            ax.set_xlabel("Residual ($)"); ax.set_ylabel("Count")
            ax.set_title("Residual Distribution")
            ax.legend(facecolor=SURFACE, edgecolor="#1e2d45", labelcolor=TEXT, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Download model summary as CSV (NEW)
        summary_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²', 'Model', 'Test Size', 'Train Samples', 'Test Samples'],
            'Value':  [f"${m['mae']:,.0f}", f"${m['rmse']:,.0f}", f"{m['r2']:.4f}",
                       model_choice, f"{test_size*100:.0f}%",
                       f"{len(X_train):,}", f"{len(X_test):,}"]
        })
        st.download_button("⬇️ Download Model Summary", data=summary_df.to_csv(index=False).encode(),
                           file_name="model_summary.csv", mime="text/csv")
    else:
        st.markdown("""
        <div style="background:#1c2537;border:1px solid #1e2d45;border-radius:14px;padding:2rem;text-align:center;">
          <div style="font-size:2rem;margin-bottom:0.6rem;">🤖</div>
          <div style="color:#64748b;font-size:0.9rem;">Click <b style="color:#a78bfa;">Train Model Now</b> to begin.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 3 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-header"><span class="step-num">4</span> Single Employee Prediction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("⚠️ Train the model first (Train Model tab) before making predictions.")
    else:
        model     = st.session_state['model']
        X_columns = st.session_state['X_columns']
        df_ref    = st.session_state.get('df', df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:0.75rem;">👤 Personal Info</div>""", unsafe_allow_html=True)
            age            = st.slider("Age", int(df_ref['age'].min()), int(df_ref['age'].max()), 30)
            gender         = st.radio("Gender", df_ref['gender'].unique().tolist())
            race           = st.selectbox("Race", df_ref['race'].unique().tolist())
            native_country = st.selectbox("Native Country", sorted(df_ref['native-country'].unique().tolist()))

        with col2:
            st.markdown("""<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:0.75rem;">🎓 Education & Work</div>""", unsafe_allow_html=True)
            educational_num = st.slider("Education Level (1–16)",
                                        int(df_ref['educational-num'].min()),
                                        int(df_ref['educational-num'].max()), 10,
                                        help="1=low, 16=Doctorate")
            workclass  = st.selectbox("Work Class", df_ref['workclass'].unique().tolist())
            occupation = st.selectbox("Occupation", df_ref['occupation'].unique().tolist())
            hours_per_week = st.slider("Hours / Week",
                                       int(df_ref['hours-per-week'].min()),
                                       int(df_ref['hours-per-week'].max()), 40)

        with col3:
            st.markdown("""<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:0.75rem;">🏠 Household & Capital</div>""", unsafe_allow_html=True)
            marital_status = st.selectbox("Marital Status", df_ref['marital-status'].unique().tolist())
            relationship   = st.selectbox("Relationship",   df_ref['relationship'].unique().tolist())
            capital_gain   = st.number_input("Capital Gain ($)",  min_value=0, max_value=int(df_ref['capital-gain'].max()), value=0)
            capital_loss   = st.number_input("Capital Loss ($)",  min_value=0, max_value=int(df_ref['capital-loss'].max()), value=0)
            fnlwgt         = st.number_input("Final Weight (fnlwgt)",
                                             min_value=int(df_ref['fnlwgt'].min()),
                                             max_value=int(df_ref['fnlwgt'].max()), value=200000)

        if st.button("💡 Predict Salary", use_container_width=True):
            new_data = pd.DataFrame([{
                'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt,
                'educational-num': educational_num, 'marital-status': marital_status,
                'occupation': occupation, 'relationship': relationship,
                'race': race, 'gender': gender,
                'capital-gain': capital_gain, 'capital-loss': capital_loss,
                'hours-per-week': hours_per_week, 'native-country': native_country
            }])[X_columns]

            try:
                predicted = model.predict(new_data)[0]
                low, high = predicted * 0.85, predicted * 1.15
                nat_avg   = df_ref['income_numeric'].mean()
                pct_above = (df_ref['income_numeric'] < predicted).mean() * 100
                tier_label, tier_color = salary_tier(predicted)

                # Result card (GSAP animated via pred-reveal class)
                st.markdown(f"""
                <div class="pred-reveal" style="
                  background: linear-gradient(135deg, #1e1b4b, #2d1b69);
                  border: 1px solid #7c3aed; border-radius: 18px;
                  padding: 2.2rem; text-align: center; margin-top: 1.5rem;
                  box-shadow: 0 0 40px rgba(124,58,237,0.3);
                ">
                  <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.14em;color:#a78bfa;margin-bottom:0.5rem;">Estimated Annual Salary</div>
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:3rem;font-weight:700;color:#fff;line-height:1.1;">${predicted:,.0f}</div>
                  <div style="font-size:0.85rem;color:#a5b4fc;margin-top:0.5rem;">
                    {"Confidence range: " if show_confidence else ""}${low:,.0f} – ${high:,.0f}
                  </div>
                  <div style="margin-top:0.8rem;">
                    <span style="background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);
                      border-radius:999px;padding:4px 14px;font-size:0.78rem;color:#a78bfa;">
                      {tier_label}
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Stats row
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Dataset Average", f"${nat_avg:,.0f}")
                col_b.metric("Percentile",       f"{pct_above:.0f}th")
                col_c.metric("vs. Average",      f"${predicted - nat_avg:+,.0f}")

                # Percentile gauge bar
                fig, ax = dark_fig(7, 0.9)
                ax.barh([0], [100], color="#1e2d45", height=0.5, edgecolor="none")
                fill_color = ACCENT3 if pct_above >= 70 else (ACCENT4 if pct_above >= 40 else ACCENT)
                ax.barh([0], [pct_above], color=fill_color, height=0.5, edgecolor="none")
                ax.set_xlim(0, 100); ax.set_yticks([])
                ax.set_xlabel("Salary Percentile in Dataset")
                ax.set_title(f"This profile is in the {pct_above:.0f}th percentile", fontsize=10)
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                # Save to comparison history (NEW)
                if 'pred_history' not in st.session_state:
                    st.session_state['pred_history'] = []
                st.session_state['pred_history'].append({
                    'Age': age, 'Occupation': occupation, 'Hours/Wk': hours_per_week,
                    'Predicted ($)': int(predicted), 'Percentile': f"{pct_above:.0f}th"
                })

                st.markdown(f'<div style="margin-top:0.5rem;font-size:0.75rem;color:#475569;text-align:center;">Prediction saved to history ({len(st.session_state["pred_history"])} records)</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Prediction history table (NEW)
        if 'pred_history' in st.session_state and len(st.session_state['pred_history']) > 0:
            with st.expander(f"📋 Prediction History ({len(st.session_state['pred_history'])} runs)"):
                hist_df = pd.DataFrame(st.session_state['pred_history'])
                st.dataframe(hist_df, use_container_width=True)
                st.download_button("⬇️ Export History", data=hist_df.to_csv(index=False).encode(),
                                   file_name="prediction_history.csv", mime="text/csv")
                if st.button("🗑️ Clear History"):
                    st.session_state['pred_history'] = []
                    st.rerun()


# ═══════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ═══════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-header"><span class="step-num">5</span> Batch Prediction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("⚠️ Train the model first before running batch predictions.")
    else:
        model     = st.session_state['model']
        X_columns = st.session_state['X_columns']

        with st.expander("📄 View expected CSV format"):
            sample_csv = (
                "age,workclass,fnlwgt,educational-num,marital-status,occupation,"
                "relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country\n"
                "35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States\n"
                "28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States\n"
                "50,Self-emp-inc,180000,14,Married-civ-spouse,Prof-specialty,Wife,Asian-Pac-Islander,Female,10000,0,50,India"
            )
            st.code(sample_csv, language="csv")
            st.download_button("⬇️ Download Sample CSV", data=sample_csv,
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

                    st.success(f"✅ Predicted salaries for {len(batch_data):,} records.")

                    # Summary metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Average",  f"${preds.mean():,.0f}")
                    c2.metric("Median",   f"${np.median(preds):,.0f}")
                    c3.metric("Min",      f"${preds.min():,.0f}")
                    c4.metric("Max",      f"${preds.max():,.0f}")

                    # Distribution chart + tier breakdown (NEW)
                    col_left, col_right = st.columns([2, 1])
                    with col_left:
                        fig, ax = dark_fig(7, 3.5)
                        ax.hist(preds, bins=30, color=ACCENT, edgecolor="none", alpha=0.85)
                        ax.axvline(preds.mean(), color=ACCENT2, linewidth=1.5, linestyle="--",
                                   label=f"Mean ${preds.mean():,.0f}")
                        ax.axvline(np.median(preds), color=ACCENT3, linewidth=1.5, linestyle=":",
                                   label=f"Median ${np.median(preds):,.0f}")
                        ax.set_xlabel("Predicted Salary ($)"); ax.set_ylabel("Count")
                        ax.set_title("Salary Distribution — Batch Results")
                        ax.legend(facecolor=SURFACE, edgecolor="#1e2d45", labelcolor=TEXT, fontsize=9)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                    with col_right:
                        st.markdown("**Tier Breakdown**")
                        tiers = batch_data['Salary_Tier'].value_counts()
                        for tier, count in tiers.items():
                            pct = count / len(batch_data) * 100
                            st.markdown(f"""
                            <div style="background:#1c2537;border:1px solid #1e2d45;border-radius:10px;
                                        padding:0.7rem 1rem;margin-bottom:0.5rem;">
                              <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                                <span style="font-size:0.82rem;color:#e2e8f0;">{tier}</span>
                                <span style="font-size:0.78rem;color:#a78bfa;">{count} ({pct:.0f}%)</span>
                              </div>
                              <div style="height:4px;background:#111827;border-radius:999px;">
                                <div style="height:4px;background:#7c3aed;border-radius:999px;width:{pct}%;"></div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.dataframe(batch_data, use_container_width=True)

                    csv_out = batch_data.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Full Predictions CSV", data=csv_out,
                                       file_name='predicted_salaries.csv', mime='text/csv')

                except KeyError as ke:
                    st.error(f"Missing column: {ke}. Ensure all required columns are present.")
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")


# ═══════════════════════════════════════════════
# TAB 5 — SALARY INSIGHTS (NEW)
# ═══════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="section-header"><span class="step-num">6</span> Salary Insights & Intelligence</div>', unsafe_allow_html=True)

    if 'income_numeric' not in df.columns:
        st.info("Load a dataset with income data to see insights.")
    else:
        # Summary insights
        high_df = df[df['income_numeric'] == 75000]
        low_df  = df[df['income_numeric'] == 25000]

        # Derived facts
        top_occupation  = high_df['occupation'].mode()[0]  if 'occupation' in df.columns else "N/A"
        top_workclass   = high_df['workclass'].mode()[0]   if 'workclass' in df.columns else "N/A"
        avg_edu_high    = high_df['educational-num'].mean() if 'educational-num' in df.columns else 0
        avg_edu_low     = low_df['educational-num'].mean()  if 'educational-num' in df.columns else 0
        avg_hours_high  = high_df['hours-per-week'].mean()  if 'hours-per-week' in df.columns else 0
        avg_hours_low   = low_df['hours-per-week'].mean()   if 'hours-per-week' in df.columns else 0

        # ── Insight cards ──
        insights = [
            ("💼", "Top High-Earning Occupation",
             f"<b style='color:#e2e8f0'>{top_occupation}</b> is the most common occupation among high earners in this dataset."),
            ("🏢", "Dominant Work Class",
             f"High earners are most frequently employed in <b style='color:#e2e8f0'>{top_workclass}</b>."),
            ("🎓", "Education Premium",
             f"High earners average education level <b style='color:#a78bfa'>{avg_edu_high:.1f}/16</b> vs "
             f"<b style='color:#64748b'>{avg_edu_low:.1f}/16</b> for low earners — a <b style='color:#10b981'>"
             f"+{avg_edu_high - avg_edu_low:.1f} level gap</b>."),
            ("⏱️", "Hours Worked",
             f"High earners average <b style='color:#a78bfa'>{avg_hours_high:.0f} hrs/week</b> vs "
             f"<b style='color:#64748b'>{avg_hours_low:.0f} hrs/week</b> for low earners."),
            ("👥", "Dataset Composition",
             f"This dataset has <b style='color:#10b981'>{pct_high:.1f}%</b> high earners "
             f"({high_earners:,} records) and <b style='color:#64748b'>{100-pct_high:.1f}%</b> low earners."),
        ]

        for icon, title, body in insights:
            st.markdown(f"""
            <div class="insight-badge hover-card">
              <div class="insight-icon">{icon}</div>
              <div>
                <div class="insight-title">{title}</div>
                <div class="insight-text">{body}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

        # ── Salary rank by occupation (interactive select) ──
        st.markdown("#### 📊 Salary Ranking by Category")
        rank_col = st.selectbox("Rank by category", [c for c in ['occupation', 'workclass', 'marital-status', 'race', 'gender'] if c in df.columns])

        fig, ax = dark_fig(10, max(4, df[rank_col].nunique() * 0.4))
        grp = df.groupby(rank_col)['income_numeric'].agg(['mean', 'std', 'count']).sort_values('mean')
        colors_rank = plt.cm.get_cmap('Purples')(np.linspace(0.3, 0.9, len(grp)))
        bars = ax.barh(grp.index, grp['mean'], color=colors_rank, edgecolor="none")
        ax.axvline(df['income_numeric'].mean(), color=ACCENT2, linestyle='--', linewidth=1.2, label="Dataset mean")
        ax.set_xlabel("Average Salary ($)"); ax.set_title(f"Average Salary by {rank_col.title()}")
        ax.legend(facecolor=SURFACE, edgecolor="#1e2d45", labelcolor=TEXT, fontsize=9)
        for bar, (_, row) in zip(bars, grp.iterrows()):
            ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                    f"${row['mean']:,.0f} (n={int(row['count'])})",
                    va='center', color=MUTED, fontsize=7.5)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════
# TAB 6 — MODEL COMPARISON (NEW)
# ═══════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-header"><span class="step-num">7</span> Compare Models</div>', unsafe_allow_html=True)

    if not enable_comparison:
        st.markdown("""
        <div style="background:#1c2537;border:1px solid #1e2d45;border-radius:14px;padding:2rem;text-align:center;">
          <div style="font-size:1.8rem;margin-bottom:0.6rem;">⚖️</div>
          <div style="color:#64748b;font-size:0.9rem;">Enable <b style="color:#a78bfa;">Model Comparison Mode</b> in the sidebar to compare all algorithms.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Training all 5 models for comparison — this may take 1–2 minutes.")

        if st.button("🔬 Run Full Comparison", use_container_width=True):
            X_cmp = df.drop(columns=[c for c in ['income', 'income_numeric'] if c in df.columns])
            y_cmp = df['income_numeric']

            cat_f = X_cmp.select_dtypes(include='object').columns.tolist()
            num_f = X_cmp.select_dtypes(include=np.number).columns.tolist()
            prep  = ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_f),
                ('num', 'passthrough', num_f)
            ])

            X_tr, X_te, y_tr, y_te = train_test_split(X_cmp, y_cmp, test_size=test_size, random_state=42)

            results = []
            progress = st.progress(0)
            models_to_test = ["Random Forest", "Gradient Boosting", "Extra Trees", "Ridge Regression", "Lasso Regression"]

            for i, mname in enumerate(models_to_test):
                with st.spinner(f"Training {mname}…"):
                    m = train_model(X_tr, y_tr, prep, model_type=mname)
                    yp = m.predict(X_te)
                    results.append({
                        'Model': mname,
                        'MAE ($)':  int(mean_absolute_error(y_te, yp)),
                        'RMSE ($)': int(np.sqrt(mean_squared_error(y_te, yp))),
                        'R² Score': round(r2_score(y_te, yp), 4),
                    })
                progress.progress((i + 1) / len(models_to_test))

            progress.empty()
            results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
            st.session_state['comparison_results'] = results_df
            st.success("✅ Comparison complete!")

        if 'comparison_results' in st.session_state:
            cdf = st.session_state['comparison_results']
            best = cdf.iloc[0]['Model']

            # Table
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
              <span class="tag tag-emerald">🏆 Best: {best}</span>
            </div>
            <table class="compare-table" style="width:100%;border-collapse:collapse;background:#1c2537;border-radius:14px;overflow:hidden;">
              <tr>
                {''.join(f'<th>{col}</th>' for col in cdf.columns)}
              </tr>
              {''.join(
                f'<tr>{"".join(f"<td style=\"color:{"#10b981" if row["Model"] == best else "#e2e8f0"}\">{row[col]}</td>" for col in cdf.columns)}</tr>'
                for _, row in cdf.iterrows()
              )}
            </table>
            """, unsafe_allow_html=True)

            # R² comparison chart
            fig, ax = dark_fig(9, 3.5)
            bar_colors = [ACCENT3 if m == best else ACCENT for m in cdf['Model']]
            bars = ax.barh(cdf['Model'], cdf['R² Score'], color=bar_colors, edgecolor="none")
            ax.set_xlim(0, 1); ax.set_xlabel("R² Score (higher = better)")
            ax.set_title("Model R² Score Comparison")
            for bar, val in zip(bars, cdf['R² Score']):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va='center', color=TEXT, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.download_button("⬇️ Download Comparison CSV",
                               data=cdf.to_csv(index=False).encode(),
                               file_name="model_comparison.csv", mime="text/csv")
