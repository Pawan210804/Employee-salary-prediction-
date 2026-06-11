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

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="SalaryIQ — ML Salary Intelligence",
    page_icon="💡",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# HIDE STREAMLIT CHROME
# ─────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu{visibility:hidden;}
header{visibility:hidden;}
footer{visibility:hidden;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stDecoration"]{display:none!important;}
[data-testid="stStatusWidget"]{display:none!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FULL THEME — GLASSY CREAM VINTAGE
# ─────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── TOKENS ── */
:root {
  --cream:        #F5F0E8;
  --cream-dark:   #EDE6D6;
  --cream-deeper: #E3D9C6;
  --parchment:    #D4C9B0;
  --ink:          #2C2416;
  --ink-mid:      #5A5040;
  --ink-muted:    #8A7D6A;
  --ink-ghost:    #B8AA96;
  --sienna:       #8B4A2B;
  --sienna-light: #C47B52;
  --sienna-pale:  #F0DDD1;
  --sage:         #5C7A5E;
  --sage-pale:    #D6E5D7;
  --amber-warm:   #B87333;
  --amber-pale:   #F2E8D0;
  --border:       rgba(90,70,50,0.14);
  --border-warm:  rgba(139,74,43,0.22);
  --glass:        rgba(245,240,232,0.72);
  --glass-dark:   rgba(237,230,214,0.55);
  --shadow:       0 2px 24px rgba(44,36,22,0.09);
  --shadow-lift:  0 8px 40px rgba(44,36,22,0.13);
  --radius-sm:    8px;
  --radius-md:    14px;
  --radius-lg:    20px;
  --font-display: 'Playfair Display', Georgia, serif;
  --font-body:    'DM Sans', system-ui, sans-serif;
  --font-mono:    'DM Mono', monospace;
}

/* ── BASE ── */
html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  background: var(--cream) !important;
  color: var(--ink) !important;
}
.stApp { background: var(--cream) !important; }
.main .block-container {
  padding-top: 1.5rem !important;
  max-width: 1340px;
}

/* ── ANIMATED BACKGROUND CANVAS ── */
#bg-canvas {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  z-index: 0;
  opacity: 0.55;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--cream-dark) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stTextInput input {
  background: var(--cream) !important;
  border: 1px solid var(--border) !important;
  color: var(--ink) !important;
}

/* ── INPUTS ── */
.stSelectbox select, .stTextInput input, .stNumberInput input {
  background: var(--cream) !important;
  border: 1px solid var(--border) !important;
  color: var(--ink) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-body) !important;
}
.stSlider > div > div { color: var(--sienna) !important; }
.stRadio label { color: var(--ink-mid) !important; }
.stCheckbox label { color: var(--ink-mid) !important; }
.stFileUploader {
  background: var(--cream-dark) !important;
  border: 1.5px dashed var(--parchment) !important;
  border-radius: var(--radius-md) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--cream-dark);
  border-radius: var(--radius-md);
  padding: 5px;
  gap: 4px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: var(--radius-sm) !important;
  color: var(--ink-muted) !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  font-family: var(--font-body) !important;
  transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
  background: var(--sienna) !important;
  color: #fff !important;
  box-shadow: 0 2px 12px rgba(139,74,43,0.3) !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background: var(--sienna) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.6rem 2rem !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  font-family: var(--font-body) !important;
  letter-spacing: 0.02em !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 3px 14px rgba(139,74,43,0.28) !important;
}
.stButton > button:hover {
  background: #7A3D22 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 22px rgba(139,74,43,0.38) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.1rem 1.3rem;
  backdrop-filter: blur(8px);
}
[data-testid="stMetricValue"] {
  color: var(--ink) !important;
  font-family: var(--font-display) !important;
  font-size: 1.65rem !important;
}
[data-testid="stMetricLabel"] {
  color: var(--ink-muted) !important;
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}

/* ── DATA TABLE ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-md) !important; overflow: hidden !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
  background: var(--cream-dark) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--sienna) !important;
  font-family: var(--font-body) !important;
}
.streamlit-expanderContent {
  background: var(--cream) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

/* ── ALERTS ── */
.stAlert { border-radius: var(--radius-md) !important; }

/* ── GLASS CARD ── */
.glass-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.8rem;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: var(--shadow);
  transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.glass-card:hover {
  box-shadow: var(--shadow-lift);
  transform: translateY(-2px);
}

/* ── SECTION HEADER ── */
.section-header {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  font-family: var(--font-display);
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--sienna);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.6rem;
  margin: 2rem 0 1rem 0;
}
.step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px; height: 28px;
  border-radius: 50%;
  background: var(--sienna);
  color: #fff;
  font-family: var(--font-body);
  font-weight: 700;
  font-size: 0.78rem;
  flex-shrink: 0;
}

/* ── TAGS / PILLS ── */
.tag {
  display: inline-flex; align-items: center; gap: 0.3rem;
  background: var(--cream-dark);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.22rem 0.75rem;
  font-size: 0.72rem;
  color: var(--ink-muted);
  font-family: var(--font-body);
}
.tag-sienna { border-color: var(--border-warm); color: var(--sienna); background: var(--sienna-pale); }
.tag-sage   { border-color: rgba(92,122,94,0.3); color: var(--sage); background: var(--sage-pale); }
.tag-amber  { border-color: rgba(184,115,51,0.3); color: var(--amber-warm); background: var(--amber-pale); }

/* ── INSIGHT BADGE ── */
.insight-badge {
  display: flex; align-items: flex-start; gap: 0.8rem;
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem 1.2rem;
  margin-bottom: 0.65rem;
  backdrop-filter: blur(8px);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.insight-badge:hover { border-color: var(--sienna-light); box-shadow: var(--shadow); }
.insight-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 0.1rem; }
.insight-title { font-weight: 600; color: var(--ink); margin-bottom: 0.2rem; font-size: 0.875rem; }
.insight-text  { font-size: 0.83rem; color: var(--ink-mid); line-height: 1.55; }

/* ── SECTION RULE ── */
.section-rule {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--parchment), transparent);
  margin: 2rem 0;
}

/* ── PREDICTION CARD ── */
@keyframes predReveal {
  from { opacity:0; transform: scale(0.93) translateY(12px); }
  to   { opacity:1; transform: scale(1) translateY(0); }
}
.pred-reveal { animation: predReveal 0.5s cubic-bezier(0.22,1,0.36,1) forwards; }

/* ── HAMBURGER MENU ── */
.model-drawer {
  position: fixed;
  top: 0; right: 0;
  height: 100vh;
  width: 300px;
  background: var(--cream-dark);
  border-left: 1px solid var(--border);
  backdrop-filter: blur(20px);
  z-index: 9999;
  transform: translateX(100%);
  transition: transform 0.35s cubic-bezier(0.22,1,0.36,1);
  padding: 2rem 1.5rem;
  box-shadow: -8px 0 40px rgba(44,36,22,0.12);
}
.model-drawer.open { transform: translateX(0); }
.drawer-overlay {
  position: fixed;
  inset: 0;
  background: rgba(44,36,22,0.18);
  backdrop-filter: blur(3px);
  z-index: 9998;
  display: none;
  cursor: pointer;
}
.drawer-overlay.open { display: block; }
.drawer-item {
  display: flex; align-items: center; gap: 0.8rem;
  padding: 0.85rem 1rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.18s;
  margin-bottom: 0.35rem;
  border: 1px solid transparent;
  font-family: var(--font-body);
  font-size: 0.88rem;
  color: var(--ink-mid);
}
.drawer-item:hover { background: var(--cream-deeper); border-color: var(--border); }
.drawer-item.active {
  background: var(--sienna-pale);
  border-color: var(--border-warm);
  color: var(--sienna);
  font-weight: 600;
}
.drawer-icon { font-size: 1.1rem; width: 26px; text-align: center; }
.hamburger-btn {
  position: fixed;
  top: 1rem; right: 1rem;
  z-index: 10000;
  width: 42px; height: 42px;
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
  transition: all 0.2s;
}
.hamburger-btn:hover { background: var(--sienna-pale); border-color: var(--border-warm); }
.ham-line {
  width: 18px; height: 1.5px;
  background: var(--ink);
  display: block;
  margin: 2.5px auto;
  transition: all 0.25s;
  border-radius: 2px;
}
.active-model-badge {
  position: fixed;
  top: 1rem; right: 4rem;
  z-index: 9997;
  display: flex; align-items: center; gap: 0.5rem;
  background: var(--glass);
  border: 1px solid var(--border-warm);
  border-radius: 999px;
  padding: 0.3rem 0.9rem 0.3rem 0.55rem;
  font-size: 0.75rem;
  font-family: var(--font-body);
  color: var(--sienna);
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
  font-weight: 500;
}

/* ── EMPTY STATE ── */
.empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center;
  padding: 3rem 2rem;
  background: var(--glass);
  border: 1.5px dashed var(--parchment);
  border-radius: var(--radius-lg);
  gap: 0.75rem;
}
.empty-state-icon {
  font-size: 2.5rem; margin-bottom: 0.4rem;
  opacity: 0.6;
}
.empty-state-title {
  font-family: var(--font-display);
  font-size: 1.15rem;
  color: var(--ink-mid);
  font-weight: 600;
}
.empty-state-sub {
  font-size: 0.83rem; color: var(--ink-muted); max-width: 360px; line-height: 1.6;
}
.empty-tip {
  background: var(--sienna-pale);
  border: 1px solid var(--border-warm);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.9rem;
  font-size: 0.78rem;
  color: var(--sienna);
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--cream-dark); }
::-webkit-scrollbar-thumb { background: var(--parchment); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--sienna-light); }

/* ── COMPARE TABLE ── */
.compare-table th {
  color: var(--sienna); font-size: 0.76rem; letter-spacing: 0.07em;
  text-transform: uppercase; padding: 0.7rem 1rem;
  border-bottom: 1px solid var(--border);
  font-family: var(--font-body);
}
.compare-table td {
  padding: 0.6rem 1rem; border-bottom: 1px solid rgba(90,70,50,0.07);
  font-size: 0.84rem; color: var(--ink-mid);
}
.compare-table tr:last-child td { border-bottom: none; }
.compare-table tr:hover td { background: var(--sienna-pale); }

/* ── MODEL SCORE RING ── */
.score-ring {
  display: inline-flex; align-items: center; justify-content: center;
  width: 78px; height: 78px; border-radius: 50%;
  border: 2.5px solid var(--sienna);
  font-family: var(--font-display);
  font-size: 1.2rem; font-weight: 700; color: var(--sienna);
  box-shadow: 0 0 18px rgba(139,74,43,0.2);
}

/* ── PULSE DOT ── */
@keyframes pulseDot {
  0%,100% { opacity:1; transform:scale(1); }
  50% { opacity:0.5; transform:scale(1.4); }
}
.pulse-dot { animation: pulseDot 2.5s ease-in-out infinite; }

/* ── HERO ── */
@keyframes heroFadeUp {
  from { opacity:0; transform:translateY(18px); }
  to   { opacity:1; transform:translateY(0); }
}
.hero-child { animation: heroFadeUp 0.7s cubic-bezier(0.22,1,0.36,1) both; }
.hero-child:nth-child(2) { animation-delay:0.1s; }
.hero-child:nth-child(3) { animation-delay:0.2s; }
.hero-child:nth-child(4) { animation-delay:0.3s; }
</style>

<!-- ANIMATED BACKGROUND: soft floating orbs in cream/sienna palette -->
<canvas id="bg-canvas"></canvas>
<script>
(function() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, orbs = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  const palette = [
    'rgba(196,123,82,0.13)',
    'rgba(212,201,176,0.18)',
    'rgba(92,122,94,0.09)',
    'rgba(184,115,51,0.10)',
    'rgba(240,221,209,0.20)',
    'rgba(139,74,43,0.08)',
  ];

  function makeOrb() {
    return {
      x:    Math.random() * W,
      y:    Math.random() * H,
      r:    120 + Math.random() * 220,
      vx:   (Math.random() - 0.5) * 0.22,
      vy:   (Math.random() - 0.5) * 0.18,
      col:  palette[Math.floor(Math.random() * palette.length)],
      phase: Math.random() * Math.PI * 2,
      freq:  0.0003 + Math.random() * 0.0004,
    };
  }

  resize();
  window.addEventListener('resize', resize);
  for (let i = 0; i < 9; i++) orbs.push(makeOrb());

  function tick(t) {
    ctx.clearRect(0, 0, W, H);
    orbs.forEach(o => {
      const pulse = 1 + 0.07 * Math.sin(t * o.freq + o.phase);
      const r = o.r * pulse;
      const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, r);
      g.addColorStop(0, o.col);
      g.addColorStop(1, 'rgba(245,240,232,0)');
      ctx.beginPath();
      ctx.arc(o.x, o.y, r, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
      o.x += o.vx;
      o.y += o.vy;
      if (o.x < -o.r) o.x = W + o.r;
      if (o.x > W + o.r) o.x = -o.r;
      if (o.y < -o.r) o.y = H + o.r;
      if (o.y > H + o.r) o.y = -o.r;
    });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>

<!-- HAMBURGER MENU -->
<div class="active-model-badge" id="active-badge">
  <span id="badge-icon">🌲</span>
  <span id="badge-name">Random Forest</span>
</div>

<div class="hamburger-btn" id="ham-btn" onclick="toggleDrawer()" title="Switch model">
  <div>
    <span class="ham-line"></span>
    <span class="ham-line"></span>
    <span class="ham-line"></span>
  </div>
</div>

<div class="drawer-overlay" id="drawer-overlay" onclick="closeDrawer()"></div>

<div class="model-drawer" id="model-drawer">
  <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid rgba(90,70,50,0.12);">
    <div class="pulse-dot" style="width:7px;height:7px;background:var(--sienna);border-radius:50%;flex-shrink:0;"></div>
    <span style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:600;color:var(--ink);">SalaryIQ</span>
    <span style="margin-left:auto;font-size:0.72rem;color:var(--ink-muted);">Select Model</span>
  </div>

  <div id="model-list">
    <div class="drawer-item active" onclick="selectModel('Random Forest','🌲')" data-model="Random Forest">
      <span class="drawer-icon">🌲</span>
      <div>
        <div style="font-weight:600;">Random Forest</div>
        <div style="font-size:0.73rem;color:var(--ink-muted);margin-top:1px;">Ensemble · Best accuracy</div>
      </div>
    </div>
    <div class="drawer-item" onclick="selectModel('Gradient Boosting','⚡')" data-model="Gradient Boosting">
      <span class="drawer-icon">⚡</span>
      <div>
        <div style="font-weight:600;">Gradient Boosting</div>
        <div style="font-size:0.73rem;color:var(--ink-muted);margin-top:1px;">Boosted · Nonlinear</div>
      </div>
    </div>
    <div class="drawer-item" onclick="selectModel('Extra Trees','🌳')" data-model="Extra Trees">
      <span class="drawer-icon">🌳</span>
      <div>
        <div style="font-weight:600;">Extra Trees</div>
        <div style="font-size:0.73rem;color:var(--ink-muted);margin-top:1px;">Fast · Low variance</div>
      </div>
    </div>
    <div class="drawer-item" onclick="selectModel('Ridge Regression','📐')" data-model="Ridge Regression">
      <span class="drawer-icon">📐</span>
      <div>
        <div style="font-weight:600;">Ridge Regression</div>
        <div style="font-size:0.73rem;color:var(--ink-muted);margin-top:1px;">Linear · L2 regularisation</div>
      </div>
    </div>
    <div class="drawer-item" onclick="selectModel('Lasso Regression','🔗')" data-model="Lasso Regression">
      <span class="drawer-icon">🔗</span>
      <div>
        <div style="font-weight:600;">Lasso Regression</div>
        <div style="font-size:0.73rem;color:var(--ink-muted);margin-top:1px;">Linear · L1 · Sparse</div>
      </div>
    </div>
  </div>

  <div style="margin-top:auto;padding-top:1.5rem;border-top:1px solid rgba(90,70,50,0.12);margin-top:1.5rem;">
    <div style="font-size:0.72rem;color:var(--ink-muted);line-height:1.7;">
      <b style="color:var(--sienna);">How to use</b><br>
      1. Upload <code style="background:var(--cream-deeper);padding:1px 5px;border-radius:4px;">adult.csv</code><br>
      2. Select a model above<br>
      3. Explore data in EDA tab<br>
      4. Train &amp; evaluate<br>
      5. Predict single or batch
    </div>
  </div>
</div>

<script>
let drawerOpen = false;
function toggleDrawer() {
  drawerOpen = !drawerOpen;
  document.getElementById('model-drawer').classList.toggle('open', drawerOpen);
  document.getElementById('drawer-overlay').classList.toggle('open', drawerOpen);
}
function closeDrawer() {
  drawerOpen = false;
  document.getElementById('model-drawer').classList.remove('open');
  document.getElementById('drawer-overlay').classList.remove('open');
}
function selectModel(name, icon) {
  document.querySelectorAll('.drawer-item').forEach(el => {
    el.classList.toggle('active', el.dataset.model === name);
  });
  document.getElementById('badge-name').textContent = name;
  document.getElementById('badge-icon').textContent = icon;
  closeDrawer();
  // Store in sessionStorage for Streamlit interaction hint
  sessionStorage.setItem('selectedModel', name);
}
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB CREAM THEME
# ─────────────────────────────────────────────
BG      = "#F5F0E8"
SURFACE = "#EDE6D6"
CARD    = "#E3D9C6"
ACCENT  = "#8B4A2B"
ACCENT2 = "#5C7A5E"
ACCENT3 = "#B87333"
ACCENT4 = "#C47B52"
TEXT    = "#2C2416"
MUTED   = "#8A7D6A"
BORDER  = "#D4C9B0"

def dark_fig(w=10, h=4, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax_list = [axes] if nrows * ncols == 1 else axes.flatten()
    for ax in ax_list:
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(ACCENT)
        ax.title.set_fontsize(11)
    return fig, axes

# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
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
    if val < 30000:   return "Entry Level", ACCENT
    elif val < 55000: return "Mid Level", ACCENT3
    elif val < 75000: return "Senior Level", ACCENT2
    else:             return "Executive", ACCENT

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 0.4rem;display:flex;align-items:center;gap:0.55rem;">
      <div style="width:7px;height:7px;border-radius:50%;background:var(--sienna);" class="pulse-dot"></div>
      <span style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:600;color:var(--ink);">SalaryIQ</span>
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
    show_confidence  = st.checkbox("Show confidence intervals", value=True)
    enable_comparison = st.checkbox("Enable model comparison mode", value=False)
    show_shap_proxy  = st.checkbox("Show feature attribution chart", value=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.78rem;color:var(--ink-muted);line-height:1.75;">
      <b style="color:var(--sienna);">Quick guide</b><br>
      1. Upload <code>adult.csv</code><br>
      2. Open ☰ menu to pick model<br>
      3. Explore data in EDA tab<br>
      4. Train your model<br>
      5. Predict single or batch<br>
      6. Export results
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<span class="tag tag-sienna">adult.csv · UCI Census Income</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="glass-card hero-child" style="margin-bottom:2rem;border-radius:var(--radius-lg);position:relative;overflow:hidden;">
  <!-- decorative ruled lines -->
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,transparent,var(--sienna-light),transparent);opacity:0.5;"></div>
  <div style="position:relative;z-index:1;">
    <div class="hero-child" style="display:flex;align-items:baseline;gap:0.7rem;margin-bottom:0.5rem;">
      <span style="font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;
        color:var(--sienna);letter-spacing:-0.01em;line-height:1;">SalaryIQ</span>
      <span style="font-size:1.4rem;opacity:0.7;">💡</span>
    </div>
    <p class="hero-child" style="color:var(--ink-mid);font-size:1rem;max-width:540px;
      line-height:1.65;margin:0 0 1.2rem;font-weight:300;">
      Machine-learning salary intelligence built on census data.<br>
      Upload, train in one click, and uncover pay insights instantly.
    </p>
    <div class="hero-child" style="display:flex;gap:0.5rem;flex-wrap:wrap;">
      <span class="tag tag-sienna">5 ML Models</span>
      <span class="tag tag-sage">EDA + Visualisations</span>
      <span class="tag tag-amber">Batch Export</span>
      <span class="tag">Salary Insights</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="step-num">1</span> Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload adult.csv", type="csv", label_visibility="collapsed")

if uploaded_file is None:
    # ── Rich empty state ──
    st.markdown("""
    <div class="empty-state" style="margin-bottom:1.5rem;">
      <div class="empty-state-icon">📂</div>
      <div class="empty-state-title">No dataset loaded yet</div>
      <div class="empty-state-sub">
        Drop your <code style="background:var(--cream-deeper);padding:2px 6px;border-radius:4px;
        font-family:'DM Mono',monospace;font-size:0.82em;">adult.csv</code> file in the uploader above to begin.
        The app will automatically clean, preprocess, and prepare the data for training.
      </div>
      <div class="empty-tip">UCI Census Income dataset · 14 features · ~48,000 records</div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:0.5rem;">
      <div class="glass-card" style="text-align:center;padding:1.5rem 1rem;">
        <div style="font-size:1.8rem;margin-bottom:0.5rem;">🔬</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.95rem;font-weight:600;color:var(--ink);margin-bottom:0.35rem;">Explore &amp; Visualise</div>
        <div style="font-size:0.78rem;color:var(--ink-muted);line-height:1.55;">
          Histograms, box plots, correlation matrices, and income breakdowns across all features.
        </div>
      </div>
      <div class="glass-card" style="text-align:center;padding:1.5rem 1rem;">
        <div style="font-size:1.8rem;margin-bottom:0.5rem;">🤖</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.95rem;font-weight:600;color:var(--ink);margin-bottom:0.35rem;">Train Any Model</div>
        <div style="font-size:0.78rem;color:var(--ink-muted);line-height:1.55;">
          Choose from 5 ML algorithms. One-click training with full evaluation — MAE, RMSE, R².
        </div>
      </div>
      <div class="glass-card" style="text-align:center;padding:1.5rem 1rem;">
        <div style="font-size:1.8rem;margin-bottom:0.5rem;">📦</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.95rem;font-weight:600;color:var(--ink);margin-bottom:0.35rem;">Predict &amp; Export</div>
        <div style="font-size:0.78rem;color:var(--ink-muted);line-height:1.55;">
          Single-record predictions with percentile gauge, or batch CSV uploads with tier breakdown.
        </div>
      </div>
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
pct_high  = high_earners / n_rows * 100
avg_salary = df['income_numeric'].mean() if 'income_numeric' in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records",           f"{n_rows:,}",          "after cleaning")
c2.metric("Features",          f"{n_cols - 2}",         "input columns")
c3.metric("High Earners >50K", f"{pct_high:.1f}%",      f"{high_earners:,} records")
c4.metric("Algorithm",         model_choice,            "selected")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
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
                <div class="insight-badge" style="padding:0.55rem 0.85rem;margin-bottom:0.35rem;">
                  <div>
                    <div class="insight-title" style="font-size:0.76rem;">{col}</div>
                    <div class="insight-text" style="font-size:0.7rem;">
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
                    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
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
                    for patch, c in zip(bp['boxes'], [ACCENT, ACCENT2]):
                        patch.set_facecolor(c); patch.set_alpha(0.45)
                    for element in ['whiskers','caps','fliers']:
                        for item in bp[element]: item.set_color(MUTED)
                else:
                    bp = ax.boxplot(df[col_to_plot].dropna(), patch_artist=True, notch=True,
                                    medianprops=dict(color=ACCENT2, linewidth=2))
                    bp['boxes'][0].set_facecolor(ACCENT); bp['boxes'][0].set_alpha(0.45)
                ax.set_ylabel(col_to_plot); ax.set_title(f"Box Plot — {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            else:
                counts = df[col_to_plot].value_counts()
                fig, ax = dark_fig(9, max(3, len(counts) * 0.45))
                colors = plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.85, len(counts)))
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
        im = ax.imshow(corr.values, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center',
                        color=TEXT if abs(corr.values[i, j]) < 0.5 else '#fff', fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03)
        cbar.ax.tick_params(colors=MUTED)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with eda_tab4:
        if 'income' in df.columns and 'occupation' in df.columns:
            fig, axes = dark_fig(12, 9, nrows=2, ncols=2)
            ax1, ax2, ax3, ax4 = axes.flatten()

            occ_income = df.groupby('occupation')['income_numeric'].mean().sort_values()
            colors_occ = [ACCENT2 if v >= 50000 else ACCENT for v in occ_income.values]
            ax1.barh(occ_income.index, occ_income.values, color=colors_occ, edgecolor="none")
            ax1.set_xlabel("Avg Income ($)"); ax1.set_title("Avg Income by Occupation")
            ax1.axvline(occ_income.mean(), color=ACCENT3, linestyle='--', linewidth=1, label='Mean')
            ax1.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

            sample = df.sample(min(2000, len(df)), random_state=42)
            colors_scatter = [ACCENT if v == 75000 else ACCENT2 for v in sample['income_numeric']]
            ax2.scatter(sample['age'], sample['hours-per-week'], alpha=0.25, s=10,
                        c=colors_scatter, edgecolors='none')
            ax2.set_xlabel("Age"); ax2.set_ylabel("Hours / Week")
            ax2.set_title("Age vs Hours (colour = income tier)")
            p1 = mpatches.Patch(color=ACCENT,  label='>50K')
            p2 = mpatches.Patch(color=ACCENT2, label='≤50K')
            ax2.legend(handles=[p1,p2], facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

            if 'educational-num' in df.columns:
                edu_income = df.groupby('educational-num')['income_numeric'].mean()
                ax3.bar(edu_income.index, edu_income.values,
                        color=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.85, len(edu_income))),
                        edgecolor="none")
                ax3.set_xlabel("Education Level (1–16)"); ax3.set_ylabel("Avg Income ($)")
                ax3.set_title("Education Level vs Avg Income")

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
    <div class="glass-card" style="margin-bottom:1rem;padding:1rem 1.3rem;">
      <div style="font-size:0.72rem;color:var(--ink-muted);margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.09em;">Selected model</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:var(--sienna);">{model_choice}</div>
      <div style="font-size:0.78rem;color:var(--ink-muted);margin-top:0.25rem;">
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

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",      f"${m['mae']:,.0f}",  "avg absolute error")
        c2.metric("RMSE",     f"${m['rmse']:,.0f}", "root mean squared")
        c3.metric("R² Score", f"{m['r2']:.3f}",     "1.0 = perfect fit")

        r2_pct = int(m['r2'] * 100)
        ring_color = ACCENT2 if r2_pct >= 80 else (ACCENT3 if r2_pct >= 60 else ACCENT)
        quality_label = (
            "Excellent fit — model explains most variance." if r2_pct >= 80 else
            "Good fit — reasonable predictive power."       if r2_pct >= 60 else
            "Moderate fit — consider a different algorithm."
        )
        st.markdown(f"""
        <div class="glass-card" style="display:flex;align-items:center;gap:1.5rem;margin:1.2rem 0;">
          <div style="width:78px;height:78px;border-radius:50%;border:2.5px solid {ring_color};
            display:flex;align-items:center;justify-content:center;
            font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:{ring_color};
            box-shadow:0 0 18px {ring_color}33;flex-shrink:0;">
            {r2_pct}%
          </div>
          <div>
            <div style="font-family:'Playfair Display',serif;font-weight:600;color:var(--ink);margin-bottom:0.2rem;">Model Fit Score</div>
            <div style="font-size:0.83rem;color:var(--ink-mid);">{quality_label}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        fi_df = get_feature_importance(st.session_state['model'], X)
        if fi_df is not None and show_shap_proxy:
            st.markdown("**Top 15 Feature Importances**")
            fig, ax = dark_fig(10, 5)
            cmap_colors = plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.9, len(fi_df)))
            bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                           color=cmap_colors[::-1], edgecolor="none")
            ax.invert_yaxis()
            ax.set_xlabel("Importance"); ax.set_title("Feature Importances (Top 15)")
            for bar, val in zip(bars, fi_df['Importance']):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va='center', color=MUTED, fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

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
            ax.plot([mn, mx], [mn, mx], color=ACCENT3, linewidth=1.5, linestyle="--", label="Perfect fit")
            ax.set_xlabel("Actual Salary ($)"); ax.set_ylabel("Predicted Salary ($)")
            ax.set_title("Actual vs. Predicted")
            ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with col_right:
            fig, ax = dark_fig(6, 4)
            ax.hist(residuals, bins=40, color=ACCENT2, edgecolor="none", alpha=0.8)
            ax.axvline(0, color=ACCENT, linestyle="--", linewidth=1.5, label="Zero error")
            ax.set_xlabel("Residual ($)"); ax.set_ylabel("Count")
            ax.set_title("Residual Distribution")
            ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        summary_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²', 'Model', 'Test Size', 'Train Samples', 'Test Samples'],
            'Value':  [f"${m['mae']:,.0f}", f"${m['rmse']:,.0f}", f"{m['r2']:.4f}",
                       model_choice, f"{test_size*100:.0f}%",
                       f"{len(X_train):,}", f"{len(X_test):,}"]
        })
        st.download_button("⬇️ Download Model Summary",
                           data=summary_df.to_csv(index=False).encode(),
                           file_name="model_summary.csv", mime="text/csv")
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">🤖</div>
          <div class="empty-state-title">No model trained yet</div>
          <div class="empty-state-sub">
            Click <b>Train Model Now</b> above to begin. Training takes 10–30 seconds depending on the chosen algorithm.
          </div>
          <div class="empty-tip">Tip: Random Forest typically gives the best accuracy on this dataset</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 3 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-header"><span class="step-num">4</span> Single Employee Prediction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">🔍</div>
          <div class="empty-state-title">Train a model first</div>
          <div class="empty-state-sub">
            Head to the <b>Train Model</b> tab, click <b>Train Model Now</b>, and come back here
            to predict salary for any employee profile.
          </div>
          <div class="empty-tip">Predictions include confidence interval and percentile ranking</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        model     = st.session_state['model']
        X_columns = st.session_state['X_columns']
        df_ref    = st.session_state.get('df', df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-muted);margin-bottom:0.75rem;">👤 Personal Info</div>', unsafe_allow_html=True)
            age            = st.slider("Age", int(df_ref['age'].min()), int(df_ref['age'].max()), 30)
            gender         = st.radio("Gender", df_ref['gender'].unique().tolist())
            race           = st.selectbox("Race", df_ref['race'].unique().tolist())
            native_country = st.selectbox("Native Country", sorted(df_ref['native-country'].unique().tolist()))

        with col2:
            st.markdown('<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-muted);margin-bottom:0.75rem;">🎓 Education & Work</div>', unsafe_allow_html=True)
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
            st.markdown('<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-muted);margin-bottom:0.75rem;">🏠 Household & Capital</div>', unsafe_allow_html=True)
            marital_status = st.selectbox("Marital Status", df_ref['marital-status'].unique().tolist())
            relationship   = st.selectbox("Relationship",   df_ref['relationship'].unique().tolist())
            capital_gain   = st.number_input("Capital Gain ($)", min_value=0, max_value=int(df_ref['capital-gain'].max()), value=0)
            capital_loss   = st.number_input("Capital Loss ($)", min_value=0, max_value=int(df_ref['capital-loss'].max()), value=0)
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

                st.markdown(f"""
                <div class="pred-reveal glass-card" style="text-align:center;margin-top:1.5rem;
                  border-color:rgba(139,74,43,0.3);padding:2rem;">
                  <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.14em;
                    color:var(--ink-muted);margin-bottom:0.5rem;">Estimated Annual Salary</div>
                  <div style="font-family:'Playfair Display',serif;font-size:3rem;font-weight:700;
                    color:var(--sienna);line-height:1.1;">${predicted:,.0f}</div>
                  <div style="font-size:0.84rem;color:var(--ink-mid);margin-top:0.45rem;">
                    {"Confidence range: " if show_confidence else ""}${low:,.0f} – ${high:,.0f}
                  </div>
                  <div style="margin-top:0.8rem;">
                    <span style="background:var(--sienna-pale);border:1px solid var(--border-warm);
                      border-radius:999px;padding:4px 14px;font-size:0.76rem;color:var(--sienna);">
                      {tier_label}
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Dataset Average", f"${nat_avg:,.0f}")
                col_b.metric("Percentile",       f"{pct_above:.0f}th")
                col_c.metric("vs. Average",      f"${predicted - nat_avg:+,.0f}")

                fig, ax = dark_fig(7, 0.9)
                ax.barh([0], [100], color=CARD, height=0.5, edgecolor="none")
                fill_color = ACCENT2 if pct_above >= 70 else (ACCENT3 if pct_above >= 40 else ACCENT)
                ax.barh([0], [pct_above], color=fill_color, height=0.5, edgecolor="none")
                ax.set_xlim(0, 100); ax.set_yticks([])
                ax.set_xlabel("Salary Percentile in Dataset")
                ax.set_title(f"This profile is in the {pct_above:.0f}th percentile", fontsize=10)
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                if 'pred_history' not in st.session_state:
                    st.session_state['pred_history'] = []
                st.session_state['pred_history'].append({
                    'Age': age, 'Occupation': occupation, 'Hours/Wk': hours_per_week,
                    'Predicted ($)': int(predicted), 'Percentile': f"{pct_above:.0f}th"
                })

            except Exception as e:
                st.error(f"Prediction error: {e}")

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
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">📦</div>
          <div class="empty-state-title">No model available</div>
          <div class="empty-state-sub">
            Train a model in the <b>Train Model</b> tab first, then upload a batch CSV here
            to predict salaries for multiple employees at once.
          </div>
          <div class="empty-tip">Batch results include salary tier labels and percentile rankings</div>
        </div>
        """, unsafe_allow_html=True)
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
                        f"{(df['income_numeric'] < p).mean()*100:.0f}th" for p in preds
                    ]

                    st.success(f"✅ Predicted salaries for {len(batch_data):,} records.")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Average",  f"${preds.mean():,.0f}")
                    c2.metric("Median",   f"${np.median(preds):,.0f}")
                    c3.metric("Min",      f"${preds.min():,.0f}")
                    c4.metric("Max",      f"${preds.max():,.0f}")

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
                        ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                    with col_right:
                        st.markdown("**Tier Breakdown**")
                        tiers = batch_data['Salary_Tier'].value_counts()
                        for tier, count in tiers.items():
                            pct = count / len(batch_data) * 100
                            st.markdown(f"""
                            <div class="glass-card" style="padding:0.65rem 0.9rem;margin-bottom:0.4rem;">
                              <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                                <span style="font-size:0.82rem;">{tier}</span>
                                <span style="font-size:0.76rem;color:var(--sienna);">{count} ({pct:.0f}%)</span>
                              </div>
                              <div style="height:3px;background:var(--cream-deeper);border-radius:999px;">
                                <div style="height:3px;background:var(--sienna);border-radius:999px;width:{pct}%;"></div>
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

        else:
            st.markdown("""
            <div class="empty-state" style="margin-top:1rem;">
              <div class="empty-state-icon">🗂️</div>
              <div class="empty-state-title">Upload a batch file</div>
              <div class="empty-state-sub">
                Drop a CSV above using the same column format as the training data.
                Click <b>View expected CSV format</b> for a sample template you can download.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 5 — SALARY INSIGHTS
# ═══════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="section-header"><span class="step-num">6</span> Salary Insights & Intelligence</div>', unsafe_allow_html=True)

    if 'income_numeric' not in df.columns:
        st.info("Load a dataset with income data to see insights.")
    else:
        high_df = df[df['income_numeric'] == 75000]
        low_df  = df[df['income_numeric'] == 25000]

        top_occupation  = high_df['occupation'].mode()[0]  if 'occupation' in df.columns else "N/A"
        top_workclass   = high_df['workclass'].mode()[0]   if 'workclass' in df.columns else "N/A"
        avg_edu_high    = high_df['educational-num'].mean() if 'educational-num' in df.columns else 0
        avg_edu_low     = low_df['educational-num'].mean()  if 'educational-num' in df.columns else 0
        avg_hours_high  = high_df['hours-per-week'].mean()  if 'hours-per-week' in df.columns else 0
        avg_hours_low   = low_df['hours-per-week'].mean()   if 'hours-per-week' in df.columns else 0

        insights = [
            ("💼", "Top High-Earning Occupation",
             f"<b style='color:var(--ink)'>{top_occupation}</b> is the most common occupation among high earners in this dataset."),
            ("🏢", "Dominant Work Class",
             f"High earners are most frequently employed in <b style='color:var(--ink)'>{top_workclass}</b>."),
            ("🎓", "Education Premium",
             f"High earners average education level <b style='color:var(--sienna)'>{avg_edu_high:.1f}/16</b> vs "
             f"<b style='color:var(--ink-muted)'>{avg_edu_low:.1f}/16</b> for low earners — a "
             f"<b style='color:var(--sage)'>+{avg_edu_high - avg_edu_low:.1f} level gap</b>."),
            ("⏱️", "Hours Worked",
             f"High earners average <b style='color:var(--sienna)'>{avg_hours_high:.0f} hrs/week</b> vs "
             f"<b style='color:var(--ink-muted)'>{avg_hours_low:.0f} hrs/week</b> for low earners."),
            ("👥", "Dataset Composition",
             f"This dataset has <b style='color:var(--sage)'>{pct_high:.1f}%</b> high earners "
             f"({high_earners:,} records) and <b style='color:var(--ink-muted)'>{100-pct_high:.1f}%</b> low earners."),
        ]

        for icon, title, body in insights:
            st.markdown(f"""
            <div class="insight-badge">
              <div class="insight-icon">{icon}</div>
              <div>
                <div class="insight-title">{title}</div>
                <div class="insight-text">{body}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 Salary Ranking by Category")
        rank_col = st.selectbox("Rank by category",
            [c for c in ['occupation','workclass','marital-status','race','gender'] if c in df.columns])

        fig, ax = dark_fig(10, max(4, df[rank_col].nunique() * 0.4))
        grp = df.groupby(rank_col)['income_numeric'].agg(['mean','std','count']).sort_values('mean')
        colors_rank = plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.88, len(grp)))
        bars = ax.barh(grp.index, grp['mean'], color=colors_rank, edgecolor="none")
        ax.axvline(df['income_numeric'].mean(), color=ACCENT2, linestyle='--', linewidth=1.2, label="Dataset mean")
        ax.set_xlabel("Average Salary ($)"); ax.set_title(f"Average Salary by {rank_col.title()}")
        ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        for bar, (_, row) in zip(bars, grp.iterrows()):
            ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                    f"${row['mean']:,.0f} (n={int(row['count'])})",
                    va='center', color=MUTED, fontsize=7.5)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════
# TAB 6 — MODEL COMPARISON
# ═══════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-header"><span class="step-num">7</span> Compare Models</div>', unsafe_allow_html=True)

    if not enable_comparison:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">⚖️</div>
          <div class="empty-state-title">Model comparison is off</div>
          <div class="empty-state-sub">
            Enable <b>Model Comparison Mode</b> in the sidebar to benchmark all 5 algorithms
            on the same dataset and see which performs best.
          </div>
          <div class="empty-tip">Comparison trains all 5 models — may take 1–2 minutes</div>
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
            models_to_test = ["Random Forest","Gradient Boosting","Extra Trees","Ridge Regression","Lasso Regression"]

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

            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
              <span class="tag tag-sage">🏆 Best: {best}</span>
            </div>
            <div class="glass-card" style="padding:0;overflow:hidden;">
            <table class="compare-table" style="width:100%;border-collapse:collapse;">
              <tr>{''.join(f'<th>{col}</th>' for col in cdf.columns)}</tr>
              {''.join(
                f'<tr>{"".join(f"<td style=\"color:{"var(--sage)" if row["Model"] == best else "var(--ink-mid)"}\">{row[col]}</td>" for col in cdf.columns)}</tr>'
                for _, row in cdf.iterrows()
              )}
            </table>
            </div>
            """, unsafe_allow_html=True)

            fig, ax = dark_fig(9, 3.5)
            bar_colors = [ACCENT2 if m == best else ACCENT for m in cdf['Model']]
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
