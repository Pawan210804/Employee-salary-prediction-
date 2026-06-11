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
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="SalaryIQ — ML Salary Intelligence",
    page_icon="💡",
    initial_sidebar_state="collapsed",
)

# ── SESSION DEFAULTS ──────────────────────────────
if "active_model" not in st.session_state:
    st.session_state.active_model = "Random Forest"
if "nav_open" not in st.session_state:
    st.session_state.nav_open = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dm = st.session_state.dark_mode

# ── THEME VARS ──────────────────────────────
if dm:
    CSS_BG     = "#1A1612"
    CSS_BG_D   = "#221E18"
    CSS_BG_DD  = "#2A2420"
    CSS_PARCH  = "#3D342A"
    CSS_INK    = "#F0EBE0"
    CSS_INK_M  = "#C8BFB0"
    CSS_INK_MU = "#9A8F80"
    CSS_INK_G  = "#6A6058"
    CSS_GLASS  = "rgba(26,22,18,0.88)"
    CSS_BORDER = "rgba(120,100,80,0.22)"
    CSS_BORDER_W = "rgba(180,120,80,0.3)"
    MUTED_PY   = "#9A8F80"
    SURFACE_PY = "#221E18"
    BG_PY      = "#1A1612"
    CARD_PY    = "#2A2420"
else:
    CSS_BG     = "#F5F0E8"
    CSS_BG_D   = "#EDE6D6"
    CSS_BG_DD  = "#E3D9C6"
    CSS_PARCH  = "#D4C9B0"
    CSS_INK    = "#2C2416"
    CSS_INK_M  = "#5A5040"
    CSS_INK_MU = "#8A7D6A"
    CSS_INK_G  = "#B8AA96"
    CSS_GLASS  = "rgba(245,240,232,0.88)"
    CSS_BORDER = "rgba(90,70,50,0.14)"
    CSS_BORDER_W = "rgba(139,74,43,0.22)"
    MUTED_PY   = "#8A7D6A"
    SURFACE_PY = "#EDE6D6"
    BG_PY      = "#F5F0E8"
    CARD_PY    = "#E3D9C6"

ACC="#8B4A2B"; ACC2="#5C7A5E"; ACC3="#B87333"; ACC4="#C47B52"
TEXT_PY = CSS_INK
BORDER_PY = CSS_PARCH

st.html(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
#MainMenu{{visibility:hidden;}}header{{visibility:hidden;}}footer{{visibility:hidden;}}
[data-testid="stToolbar"]{{display:none!important;}}
[data-testid="stDecoration"]{{display:none!important;}}
[data-testid="stStatusWidget"]{{display:none!important;}}
[data-testid="stSidebar"]{{display:none!important;}}

:root{{
  --cr:{CSS_BG};--cr-d:{CSS_BG_D};--cr-dd:{CSS_BG_DD};
  --parch:{CSS_PARCH};--ink:{CSS_INK};--ink-m:{CSS_INK_M};
  --ink-mu:{CSS_INK_MU};--ink-g:{CSS_INK_G};
  --si:#8B4A2B;--si-l:#C47B52;--si-p:{"rgba(139,74,43,0.18)" if dm else "#F0DDD1"};
  --sage:#5C7A5E;--sage-p:{"rgba(92,122,94,0.18)" if dm else "#D6E5D7"};
  --amb:#B87333;--amb-p:{"rgba(184,115,51,0.18)" if dm else "#F2E8D0"};
  --bdr:{CSS_BORDER};--bdr-w:{CSS_BORDER_W};
  --glass:{CSS_GLASS};
  --sh:0 2px 20px rgba(44,36,22,{"0.28" if dm else "0.08"});
  --sh-l:0 8px 36px rgba(44,36,22,{"0.4" if dm else "0.13"});
  --r-sm:8px;--r-md:14px;--r-lg:20px;
  --fd:'Playfair Display',Georgia,serif;
  --fb:'DM Sans',system-ui,sans-serif;
  --fm:'DM Mono',monospace;
}}

html,body,[class*="css"]{{font-family:var(--fb)!important;background:var(--cr)!important;color:var(--ink)!important;}}
.stApp{{background:var(--cr)!important;}}
.main .block-container{{padding-top:0.4rem!important;max-width:1380px;padding-left:1.5rem!important;padding-right:1.5rem!important;}}

/* INPUTS */
.stSelectbox>div>div{{background:var(--cr-d)!important;border:1px solid var(--bdr)!important;color:var(--ink)!important;border-radius:var(--r-sm)!important;}}
.stTextInput input,.stNumberInput input{{background:var(--cr-d)!important;border:1px solid var(--bdr)!important;color:var(--ink)!important;border-radius:var(--r-sm)!important;}}
.stSlider>div>div>div{{background:var(--si)!important;}}
[data-testid="stSlider"] .rc-slider-track{{background:var(--si)!important;}}
.stRadio label,.stCheckbox label{{color:var(--ink-m)!important;}}
.stFileUploader{{background:var(--cr-d)!important;border:1.5px dashed var(--parch)!important;border-radius:var(--r-md)!important;}}
[data-testid="stFileUploader"]>div{{background:var(--cr-d)!important;}}

/* TABS */
.stTabs [data-baseweb="tab-list"]{{background:var(--cr-d);border-radius:var(--r-md);padding:4px;gap:3px;border:1px solid var(--bdr);}}
.stTabs [data-baseweb="tab"]{{border-radius:var(--r-sm)!important;color:var(--ink-mu)!important;font-weight:500!important;font-size:0.82rem!important;font-family:var(--fb)!important;transition:all 0.2s!important;padding:0.45rem 0.9rem!important;}}
.stTabs [aria-selected="true"]{{background:var(--si)!important;color:#fff!important;box-shadow:0 2px 12px rgba(139,74,43,0.3)!important;}}

/* BUTTONS */
.stButton>button{{background:var(--si)!important;color:#fff!important;border:none!important;border-radius:var(--r-sm)!important;padding:0.6rem 1.8rem!important;font-weight:600!important;font-size:0.87rem!important;font-family:var(--fb)!important;letter-spacing:0.02em!important;transition:all 0.25s!important;box-shadow:0 3px 14px rgba(139,74,43,0.28)!important;}}
.stButton>button:hover{{background:#7A3D22!important;transform:translateY(-2px)!important;box-shadow:0 6px 22px rgba(139,74,43,0.38)!important;}}

/* METRICS */
[data-testid="stMetric"]{{background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-md);padding:1rem 1.2rem;}}
[data-testid="stMetricValue"]{{color:var(--ink)!important;font-family:var(--fd)!important;font-size:1.55rem!important;}}
[data-testid="stMetricLabel"]{{color:var(--ink-mu)!important;font-size:0.68rem!important;text-transform:uppercase!important;letter-spacing:0.1em!important;}}
[data-testid="stMetricDelta"]{{font-size:0.72rem!important;}}

/* EXPANDER */
.streamlit-expanderHeader{{background:var(--cr-d)!important;border-radius:var(--r-sm)!important;color:var(--si)!important;font-size:0.88rem!important;}}
.streamlit-expanderContent{{background:var(--cr)!important;border:1px solid var(--bdr)!important;border-radius:0 0 var(--r-sm) var(--r-sm)!important;}}

/* DATAFRAME */
[data-testid="stDataFrame"]{{border-radius:var(--r-md)!important;overflow:hidden;border:1px solid var(--bdr)!important;}}
.dvn-scroller{{background:var(--cr-d)!important;}}

/* SCROLLBAR */
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:var(--cr-d);}}
::-webkit-scrollbar-thumb{{background:var(--parch);border-radius:999px;}}
::-webkit-scrollbar-thumb:hover{{background:var(--si-l);}}

/* GLASS CARD */
.gc{{background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-lg);padding:1.4rem 1.7rem;box-shadow:var(--sh);transition:box-shadow 0.25s,transform 0.25s;}}
.gc:hover{{box-shadow:var(--sh-l);transform:translateY(-2px);}}
.gc-sm{{background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-md);padding:0.9rem 1.1rem;box-shadow:var(--sh);}}

/* TAGS */
.tag{{display:inline-flex;align-items:center;gap:0.3rem;background:var(--cr-d);border:1px solid var(--bdr);border-radius:999px;padding:0.2rem 0.7rem;font-size:0.71rem;color:var(--ink-mu);font-family:var(--fb);}}
.tag-s{{border-color:var(--bdr-w);color:var(--si);background:var(--si-p);}}
.tag-g{{border-color:rgba(92,122,94,0.35);color:var(--sage);background:var(--sage-p);}}
.tag-a{{border-color:rgba(184,115,51,0.35);color:var(--amb);background:var(--amb-p);}}

/* SECTION HEADER */
.sh{{display:flex;align-items:center;gap:0.7rem;font-family:var(--fd);font-size:1.1rem;font-weight:600;color:var(--si);border-bottom:1px solid var(--bdr);padding-bottom:0.5rem;margin:1.8rem 0 1rem 0;}}
.sn{{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;background:var(--si);color:#fff;font-family:var(--fb);font-weight:700;font-size:0.76rem;flex-shrink:0;}}

/* INSIGHT BADGE */
.ib{{display:flex;align-items:flex-start;gap:0.8rem;background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-md);padding:0.9rem 1.1rem;margin-bottom:0.6rem;transition:border-color 0.2s,box-shadow 0.2s;}}
.ib:hover{{border-color:var(--si-l);box-shadow:var(--sh);}}
.ib-icon{{font-size:1.15rem;flex-shrink:0;margin-top:0.05rem;}}
.ib-title{{font-weight:600;color:var(--ink);margin-bottom:0.18rem;font-size:0.86rem;}}
.ib-text{{font-size:0.81rem;color:var(--ink-m);line-height:1.55;}}

.sr{{height:1px;background:linear-gradient(90deg,transparent,var(--parch),transparent);margin:1.8rem 0;}}

/* PREDICTION ANIM */
@keyframes pR{{from{{opacity:0;transform:scale(0.94) translateY(14px);}}to{{opacity:1;transform:scale(1) translateY(0);}}}}
.pr{{animation:pR 0.5s cubic-bezier(0.22,1,0.36,1) forwards;}}

/* HERO FADE */
@keyframes hfu{{from{{opacity:0;transform:translateY(14px);}}to{{opacity:1;transform:translateY(0);}}}}
.hc{{animation:hfu 0.65s cubic-bezier(0.22,1,0.36,1) both;}}
.hc:nth-child(2){{animation-delay:0.08s;}}.hc:nth-child(3){{animation-delay:0.16s;}}.hc:nth-child(4){{animation-delay:0.24s;}}

/* EMPTY STATE */
.es{{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:2.5rem 2rem;background:var(--glass);border:1.5px dashed var(--parch);border-radius:var(--r-lg);gap:0.65rem;}}
.es-icon{{font-size:2.3rem;margin-bottom:0.3rem;opacity:0.55;}}
.es-title{{font-family:var(--fd);font-size:1.1rem;color:var(--ink-m);font-weight:600;}}
.es-sub{{font-size:0.81rem;color:var(--ink-mu);max-width:340px;line-height:1.6;}}
.es-tip{{background:var(--si-p);border:1px solid var(--bdr-w);border-radius:var(--r-sm);padding:0.45rem 0.85rem;font-size:0.76rem;color:var(--si);}}

/* PULSE DOT */
@keyframes pd{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:0.5;transform:scale(1.5);}}}}
.pdt{{animation:pd 2.5s ease-in-out infinite;}}

/* ══════════════════════════════════
   HAMBURGER NAVBAR
══════════════════════════════════ */
.nav-wrap{{
  position:sticky;top:0;z-index:9999;
  display:flex;align-items:center;justify-content:space-between;
  background:var(--glass);
  border-bottom:1px solid var(--bdr);
  padding:0.55rem 1.2rem;
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  margin-bottom:0.8rem;
  border-radius:0 0 var(--r-md) var(--r-md);
}}
.nav-brand{{display:flex;align-items:center;gap:0.55rem;}}
.nav-brand-text{{font-family:var(--fd);font-size:1.08rem;font-weight:700;color:var(--si);letter-spacing:-0.01em;}}
.nav-brand-dot{{width:8px;height:8px;border-radius:50%;background:var(--si);flex-shrink:0;}}
.nav-right{{display:flex;align-items:center;gap:0.6rem;}}
.nav-model-pill{{display:inline-flex;align-items:center;gap:0.4rem;background:var(--si-p);border:1px solid var(--bdr-w);border-radius:999px;padding:0.28rem 0.85rem;font-size:0.76rem;color:var(--si);font-weight:600;cursor:pointer;transition:all 0.2s;}}
.nav-model-pill:hover{{background:var(--si);color:#fff;}}
.nav-theme-btn{{width:32px;height:32px;border-radius:50%;background:var(--cr-dd);border:1px solid var(--bdr);display:flex;align-items:center;justify-content:center;font-size:0.9rem;cursor:pointer;transition:all 0.2s;}}
.nav-theme-btn:hover{{background:var(--si-p);border-color:var(--bdr-w);}}

/* HAMBURGER DRAWER */
.hb-btn{{
  display:flex;flex-direction:column;justify-content:center;align-items:center;
  width:36px;height:36px;border-radius:var(--r-sm);
  background:var(--cr-dd);border:1px solid var(--bdr);
  cursor:pointer;gap:4.5px;transition:all 0.2s;flex-shrink:0;
}}
.hb-btn:hover{{background:var(--si-p);border-color:var(--bdr-w);}}
.hb-line{{width:16px;height:1.5px;background:var(--ink-m);border-radius:999px;transition:all 0.22s;}}
.hb-open .hb-line:nth-child(1){{transform:translateY(6px) rotate(45deg);}}
.hb-open .hb-line:nth-child(2){{opacity:0;transform:scaleX(0);}}
.hb-open .hb-line:nth-child(3){{transform:translateY(-6px) rotate(-45deg);}}

.drawer{{
  position:fixed;top:0;right:0;width:300px;height:100vh;
  background:var(--cr-d);border-left:1px solid var(--bdr);
  z-index:99999;padding:1.5rem;
  box-shadow:-8px 0 40px rgba(44,36,22,{"0.45" if dm else "0.18"});
  overflow-y:auto;
  transform:translateX(100%);transition:transform 0.3s cubic-bezier(0.22,1,0.36,1);
}}
.drawer.open{{transform:translateX(0);}}
.drawer-overlay{{
  position:fixed;inset:0;background:rgba(0,0,0,{"0.55" if dm else "0.35"});
  z-index:99998;opacity:0;pointer-events:none;
  transition:opacity 0.3s;
}}
.drawer-overlay.open{{opacity:1;pointer-events:all;}}
.drawer-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid var(--bdr);}}
.drawer-title{{font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--si);}}
.drawer-close{{width:28px;height:28px;border-radius:50%;background:var(--cr-dd);border:1px solid var(--bdr);display:flex;align-items:center;justify-content:center;font-size:0.85rem;cursor:pointer;color:var(--ink-mu);}}
.drawer-section{{margin-bottom:1.2rem;}}
.drawer-section-label{{font-size:0.68rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.55rem;}}
.model-option{{
  display:flex;align-items:center;gap:0.65rem;
  padding:0.7rem 0.9rem;border-radius:var(--r-sm);
  border:1px solid var(--bdr);background:var(--glass);
  margin-bottom:0.4rem;cursor:pointer;transition:all 0.18s;
}}
.model-option:hover{{border-color:var(--si-l);background:var(--si-p);transform:translateX(3px);}}
.model-option.active{{border-color:var(--bdr-w);background:var(--si-p);}}
.model-icon{{font-size:1.1rem;flex-shrink:0;}}
.model-name{{font-size:0.85rem;font-weight:600;color:var(--ink);line-height:1.2;}}
.model-desc{{font-size:0.71rem;color:var(--ink-mu);}}
.model-check{{margin-left:auto;color:var(--si);font-size:0.85rem;font-weight:700;}}

/* CONFIG ITEM */
.cfg-row{{display:flex;align-items:center;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--bdr);}}
.cfg-label{{font-size:0.82rem;color:var(--ink-m);}}
.cfg-val{{font-family:var(--fm);font-size:0.78rem;color:var(--si);font-weight:500;}}

/* PROGRESS BAR */
.pbar-wrap{{background:var(--cr-dd);border-radius:999px;height:5px;overflow:hidden;}}
.pbar-fill{{height:5px;border-radius:999px;background:linear-gradient(90deg,var(--si),var(--si-l));transition:width 0.6s ease;}}

/* SCORE CARD */
.score-card{{
  display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin:1rem 0;
}}
.score-item{{
  background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-md);
  padding:1rem;text-align:center;
}}
.score-val{{font-family:var(--fd);font-size:1.5rem;font-weight:700;color:var(--si);}}
.score-label{{font-size:0.68rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.09em;margin-top:0.25rem;}}

/* GROWTH TABLE */
.growth-row{{display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid var(--bdr);}}
.growth-year{{font-family:var(--fm);font-size:0.8rem;color:var(--ink-mu);width:60px;flex-shrink:0;}}
.growth-bar{{flex:1;height:8px;background:var(--cr-dd);border-radius:999px;overflow:hidden;}}
.growth-fill{{height:8px;border-radius:999px;background:linear-gradient(90deg,var(--sage),#8BB890);}}
.growth-val{{font-weight:600;font-size:0.82rem;color:var(--ink);width:80px;text-align:right;flex-shrink:0;}}

/* HEALTH SCORE */
.health-ring{{
  width:90px;height:90px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--fd);font-size:1.35rem;font-weight:700;
  flex-shrink:0;
}}

/* NOTIFICATION TOAST */
@keyframes toastIn{{from{{opacity:0;transform:translateY(20px);}}to{{opacity:1;transform:translateY(0);}}}}
.toast{{
  position:fixed;bottom:1.5rem;right:1.5rem;
  background:var(--si);color:#fff;
  padding:0.65rem 1.2rem;border-radius:var(--r-md);
  font-size:0.83rem;font-weight:500;
  box-shadow:0 4px 20px rgba(139,74,43,0.4);
  animation:toastIn 0.35s ease forwards;
  z-index:99999;
}}
</style>

<!-- BACKGROUND CANVAS -->
<canvas id="bg-canvas" style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;opacity:{"0.2" if dm else "0.32"};z-index:0;"></canvas>
<script>
(function(){{
  const c=document.getElementById('bg-canvas');if(!c)return;
  const x=c.getContext('2d');let W,H,orbs=[];
  function resize(){{W=c.width=window.innerWidth;H=c.height=window.innerHeight;}}
  const pal=['rgba(196,123,82,0.14)','rgba(212,201,176,0.18)','rgba(92,122,94,0.08)','rgba(184,115,51,0.1)','rgba(240,221,209,0.2)','rgba(139,74,43,0.06)'];
  function mk(){{return{{x:Math.random()*W,y:Math.random()*H,r:100+Math.random()*180,vx:(Math.random()-0.5)*0.15,vy:(Math.random()-0.5)*0.12,col:pal[Math.floor(Math.random()*pal.length)],ph:Math.random()*Math.PI*2,fr:0.0003+Math.random()*0.0004}};}}
  resize();window.addEventListener('resize',resize);
  for(let i=0;i<7;i++)orbs.push(mk());
  function tick(t){{
    x.clearRect(0,0,W,H);
    orbs.forEach(o=>{{
      const r=o.r*(1+0.05*Math.sin(t*o.fr+o.ph));
      const g=x.createRadialGradient(o.x,o.y,0,o.x,o.y,r);
      g.addColorStop(0,o.col);g.addColorStop(1,'rgba(0,0,0,0)');
      x.beginPath();x.arc(o.x,o.y,r,0,Math.PI*2);x.fillStyle=g;x.fill();
      o.x+=o.vx;o.y+=o.vy;
      if(o.x<-o.r)o.x=W+o.r;if(o.x>W+o.r)o.x=-o.r;
      if(o.y<-o.r)o.y=H+o.r;if(o.y>H+o.r)o.y=-o.r;
    }});
    requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
</script>
""")


# ── MODEL REGISTRY ────────────────────────────
MODELS = [
    ("Random Forest",     "🌲", "Ensemble · Best accuracy"),
    ("Gradient Boosting", "⚡", "Boosted · Nonlinear"),
    ("Extra Trees",       "🌳", "Fast · Low variance"),
    ("Ridge Regression",  "📐", "Linear · L2 regularisation"),
    ("Lasso Regression",  "🔗", "Linear · L1 · Sparse"),
]
MODEL_NAMES = [m[0] for m in MODELS]
MODEL_ICONS = {m[0]: m[1] for m in MODELS}
MODEL_DESC  = {m[0]: m[2] for m in MODELS}

# ── HAMBURGER NAV ────────────────────────────
am = st.session_state.active_model

# Build drawer HTML (pure display — interaction via Streamlit buttons below)
drawer_items_html = ""
for name, icon, desc in MODELS:
    active_cls = "active" if name == am else ""
    check = '<span class="model-check">✓</span>' if name == am else ""
    drawer_items_html += f"""
    <div class="model-option {active_cls}" id="opt-{name.replace(' ','-')}">
      <span class="model-icon">{icon}</span>
      <div><div class="model-name">{name}</div><div class="model-desc">{desc}</div></div>
      {check}
    </div>"""

dm_icon = "☀️" if dm else "🌙"
nav_open = st.session_state.nav_open
drawer_cls = "open" if nav_open else ""

st.html(f"""
<div class="nav-wrap">
  <div class="nav-brand">
    <div class="nav-brand-dot pdt"></div>
    <span class="nav-brand-text">SalaryIQ</span>
    <span style="font-size:0.7rem;color:var(--ink-mu);margin-left:0.4rem;border-left:1px solid var(--bdr);padding-left:0.6rem;">ML Salary Intelligence</span>
  </div>
  <div class="nav-right">
    <span class="nav-model-pill">
      {MODEL_ICONS[am]} {am}
    </span>
    <span class="tag tag-s" style="font-size:0.68rem;">{"🌙 Dark" if dm else "☀️ Light"}</span>
  </div>
</div>

<div class="drawer-overlay {drawer_cls}" id="drawer-overlay" onclick="closeDrawer()"></div>
<div class="drawer {drawer_cls}" id="main-drawer">
  <div class="drawer-header">
    <span class="drawer-title">⚙️ Configuration</span>
    <span class="drawer-close" onclick="closeDrawer()">✕</span>
  </div>

  <div class="drawer-section">
    <div class="drawer-section-label">Select Algorithm</div>
    {drawer_items_html}
  </div>

  <div class="drawer-section" style="margin-top:1.5rem;">
    <div class="drawer-section-label">Dataset Info</div>
    <div class="gc-sm">
      <div style="font-size:0.78rem;color:var(--ink-mu);line-height:1.9;">
        📁 adult.csv · UCI Census<br>
        👥 ~48,000 records<br>
        📊 14 features
      </div>
    </div>
  </div>
</div>

<script>
function closeDrawer(){{
  document.getElementById('main-drawer').classList.remove('open');
  document.getElementById('drawer-overlay').classList.remove('open');
}}
</script>
""")

# Hamburger + theme buttons as real Streamlit widgets
top_cols = st.columns([0.04, 0.04, 0.92])
with top_cols[0]:
    if st.button("☰", help="Open model selector", key="hb_btn"):
        st.session_state.nav_open = not st.session_state.nav_open
        st.rerun()
with top_cols[1]:
    dm_lbl = "☀️" if dm else "🌙"
    if st.button(dm_lbl, help="Toggle dark/light mode", key="theme_btn"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Model selector buttons inside an expander (opens when hamburger clicked)
if nav_open:
    with st.expander("🎛️ Select Model & Settings", expanded=True):
        st.markdown("**Choose Algorithm**")
        mcols = st.columns(len(MODELS))
        for mc, (name, icon, desc) in zip(mcols, MODELS):
            with mc:
                is_active = name == am
                lbl = f"{'✓ ' if is_active else ''}{icon} {name.split()[0]}"
                if st.button(lbl, key=f"nav_m_{name}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    st.session_state.active_model = name
                    st.session_state.nav_open = False
                    st.rerun()

        st.html('<div class="sr" style="margin:0.8rem 0;"></div>')
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            test_size_nav = st.slider("Test split (%)", 10, 40, 20, step=5, key="ts_nav") / 100
            random_seed_nav = st.number_input("Random seed", value=42, step=1, key="rs_nav")
        with col_s2:
            show_confidence   = st.checkbox("Confidence intervals", value=True, key="ci_nav")
            enable_comparison = st.checkbox("Model comparison",      value=False, key="mc_nav")
            show_feat_imp     = st.checkbox("Feature importance",    value=True,  key="fi_nav")
            show_cv           = st.checkbox("5-fold cross-val",      value=False, key="cv_nav")
            show_salary_sim   = st.checkbox("Salary simulator",      value=True,  key="ss_nav")

        # Store in session
        st.session_state['cfg_test_size']        = test_size_nav
        st.session_state['cfg_seed']             = random_seed_nav
        st.session_state['cfg_confidence']       = show_confidence
        st.session_state['cfg_comparison']       = enable_comparison
        st.session_state['cfg_feat_imp']         = show_feat_imp
        st.session_state['cfg_cv']               = show_cv
        st.session_state['cfg_salary_sim']       = show_salary_sim
else:
    # Defaults if never opened
    if 'cfg_test_size' not in st.session_state:
        st.session_state['cfg_test_size']  = 0.20
        st.session_state['cfg_seed']       = 42
        st.session_state['cfg_confidence'] = True
        st.session_state['cfg_comparison'] = False
        st.session_state['cfg_feat_imp']   = True
        st.session_state['cfg_cv']         = False
        st.session_state['cfg_salary_sim'] = True

# Pull config
model_choice      = st.session_state.active_model
test_size         = st.session_state['cfg_test_size']
random_seed       = st.session_state['cfg_seed']
show_confidence   = st.session_state['cfg_confidence']
enable_comparison = st.session_state['cfg_comparison']
show_feat_imp     = st.session_state['cfg_feat_imp']
show_cv           = st.session_state['cfg_cv']
show_salary_sim   = st.session_state['cfg_salary_sim']


# ── MATPLOTLIB THEME ────────────────────────
def dark_fig(w=10, h=4, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG_PY)
    ax_list = [axes] if nrows * ncols == 1 else axes.flatten()
    for ax in ax_list:
        ax.set_facecolor(SURFACE_PY)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_PY)
        ax.tick_params(colors=MUTED_PY, labelsize=9)
        ax.xaxis.label.set_color(TEXT_PY)
        ax.yaxis.label.set_color(TEXT_PY)
        ax.title.set_color(ACC)
        ax.title.set_fontsize(11)
    return fig, axes


# ── DATA HELPERS ──────────────────────────────
@st.cache_data
def load_data(uploaded_file):
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
    for v in ['Without-pay', 'Never-worked']:
        if 'workclass' in data.columns:
            data = data[data['workclass'] != v]
    for v in ['1st-4th', '5th-6th', 'Preschool']:
        if 'education' in data.columns:
            data = data[data['education'] != v]

    def cap(s):
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        return pd.Series(np.clip(s, Q1 - 1.5 * IQR, Q3 + 1.5 * IQR), index=s.index)

    for col in ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']:
        if col in data.columns:
            data[col] = cap(data[col])
    if 'education' in data.columns:
        data.drop(columns=['education'], inplace=True)
    if 'income' in data.columns:
        data['income_clean'] = data['income'].str.strip().str.replace('.', '', regex=False)
        data['income_numeric'] = data['income_clean'].apply(lambda x: 25000 if x == '<=50K' else 75000)
        data.drop(columns=['income_clean'], inplace=True)
    return data


@st.cache_resource
def train_model(X_train, y_train, _prep, model_type, seed=42):
    mp = {
        "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=seed),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=150, random_state=seed, n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=1.0, max_iter=5000),
    }
    reg = mp.get(model_type, mp["Random Forest"])
    m = Pipeline(steps=[('preprocessor', _prep), ('regressor', reg)])
    m.fit(X_train, y_train)
    return m


def get_feat_imp(model, cat_f, num_f):
    try:
        reg = model.named_steps['regressor']
        if not hasattr(reg, 'feature_importances_'):
            return None
        ohe = model.named_steps['preprocessor'].transformers_[0][1]
        names = ohe.get_feature_names_out(cat_f).tolist() + list(num_f)
        df2 = pd.DataFrame({'Feature': names, 'Importance': reg.feature_importances_})
        return df2.sort_values('Importance', ascending=False).head(15)
    except:
        return None


def salary_tier(v):
    if v < 30000:   return "Entry Level",   "#C07A5A"
    elif v < 55000: return "Mid Level",     ACC3
    elif v < 75000: return "Senior Level",  ACC2
    else:           return "Executive",     ACC


def data_health_score(df):
    """Compute a 0-100 data health score."""
    score = 100
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    score -= min(30, missing_pct * 3)
    if 'income_numeric' in df.columns:
        vc = df['income_numeric'].value_counts(normalize=True)
        imbalance = abs(vc.iloc[0] - 0.5) * 100
        score -= min(20, imbalance * 0.4)
    n = df.shape[0]
    if n < 1000:   score -= 20
    elif n < 5000: score -= 10
    elif n > 40000: score += 5
    score = max(0, min(100, score))
    label = "Excellent" if score >= 85 else ("Good" if score >= 65 else ("Fair" if score >= 45 else "Poor"))
    color = ACC2 if score >= 85 else (ACC3 if score >= 65 else (ACC4 if score >= 45 else ACC))
    return int(score), label, color


def salary_growth_projection(base_salary, years=10, growth_rate=0.04):
    """Project salary growth over N years."""
    return [base_salary * ((1 + growth_rate) ** y) for y in range(years + 1)]


# ── HERO ──────────────────────────────────────
st.html(f"""
<div class="gc hc" style="margin-bottom:1.2rem;border-radius:var(--r-lg);position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--si-l),transparent);opacity:0.55;"></div>
  <div style="display:flex;align-items:baseline;gap:0.65rem;margin-bottom:0.45rem;" class="hc">
    <span style="font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:700;color:var(--si);letter-spacing:-0.01em;line-height:1;">SalaryIQ</span>
    <span style="font-size:1.2rem;opacity:0.6;">💡</span>
    <span class="tag tag-s" style="margin-left:0.3rem;">v2.0</span>
  </div>
  <p class="hc" style="color:var(--ink-m);font-size:0.95rem;max-width:540px;line-height:1.65;margin:0 0 1rem;font-weight:300;">
    Machine-learning salary intelligence built on census data. Upload, train in one click, and uncover pay insights instantly.
  </p>
  <div class="hc" style="display:flex;gap:0.45rem;flex-wrap:wrap;">
    <span class="tag tag-s">5 ML Models</span>
    <span class="tag tag-g">EDA + Visualisations</span>
    <span class="tag tag-a">Batch Export</span>
    <span class="tag">Growth Projection</span>
    <span class="tag">Data Health Score</span>
    <span class="tag">Salary Simulator</span>
    <span class="tag">PDF Report</span>
  </div>
</div>
""")

# ── FILE UPLOAD ───────────────────────────────
st.html('<div class="sh"><span class="sn">1</span> Upload Dataset</div>')
uploaded_file = st.file_uploader("Upload adult.csv", type="csv", label_visibility="collapsed")

if uploaded_file is None:
    st.html("""
    <div class="es" style="margin-bottom:1.5rem;">
      <div class="es-icon">📂</div>
      <div class="es-title">No dataset loaded yet</div>
      <div class="es-sub">Drop your <code style="background:var(--cr-dd);padding:2px 6px;border-radius:4px;font-family:'DM Mono',monospace;font-size:0.82em;">adult.csv</code> above to begin. The app cleans, preprocesses and prepares data automatically.</div>
      <div class="es-tip">UCI Census Income dataset · 14 features · ~48,000 records</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;">
      <div class="gc-sm" style="text-align:center;padding:1.3rem 1rem;">
        <div style="font-size:1.7rem;margin-bottom:0.45rem;">🔬</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.9rem;font-weight:600;color:var(--ink);margin-bottom:0.3rem;">Explore & Visualise</div>
        <div style="font-size:0.76rem;color:var(--ink-mu);line-height:1.55;">Histograms, box plots, correlation matrices, and income breakdowns.</div>
      </div>
      <div class="gc-sm" style="text-align:center;padding:1.3rem 1rem;">
        <div style="font-size:1.7rem;margin-bottom:0.45rem;">🤖</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.9rem;font-weight:600;color:var(--ink);margin-bottom:0.3rem;">Train Any Model</div>
        <div style="font-size:0.76rem;color:var(--ink-mu);line-height:1.55;">5 ML algorithms, one-click training, cross-validation & evaluation.</div>
      </div>
      <div class="gc-sm" style="text-align:center;padding:1.3rem 1rem;">
        <div style="font-size:1.7rem;margin-bottom:0.45rem;">📦</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.9rem;font-weight:600;color:var(--ink);margin-bottom:0.3rem;">Predict & Export</div>
        <div style="font-size:0.76rem;color:var(--ink-mu);line-height:1.55;">Single predictions, batch CSV, growth projections and PDF report.</div>
      </div>
    </div>
    """)
    st.stop()

with st.spinner("Loading and preprocessing data…"):
    df = load_data(uploaded_file)

if df is None:
    st.error("Could not process the file.")
    st.stop()

if 'income_numeric' not in df.columns:
    st.error("⚠️ Column `income_numeric` could not be created. Ensure your CSV has an `income` column with `<=50K` / `>50K` values.")
    st.stop()

n_rows, n_cols = df.shape
high_earners = (df['income_numeric'] == 75000).sum()
pct_high = high_earners / n_rows * 100
avg_salary = df['income_numeric'].mean()
health_score, health_label, health_color = data_health_score(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Records",        f"{n_rows:,}",          "after cleaning")
c2.metric("Features",       f"{n_cols - 2}",         "input columns")
c3.metric("High Earners",   f"{pct_high:.1f}%",      f"{high_earners:,} records")
c4.metric("Algorithm",      MODEL_ICONS[model_choice] + " " + model_choice.split()[0])
c5.metric("Data Health",    f"{health_score}/100",   health_label)


# ── TABS ──────────────────────────────────────
tabs = st.tabs([
    "📊 Data Explorer",
    "🤖 Train Model",
    "🔍 Single Predict",
    "📦 Batch Predict",
    "💡 Salary Insights",
    "🎯 Simulator",
    "📈 What-If",
    "🌱 Growth Projector",  # NEW
    "🏥 Data Health",       # NEW
    "⚖️ Compare Models",
])
(tab_eda, tab_train, tab_predict, tab_batch,
 tab_insights, tab_sim, tab_whatif,
 tab_growth, tab_health, tab_compare) = tabs


# ═══════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════
with tab_eda:
    st.html('<div class="sh"><span class="sn">2</span> Explore Your Data</div>')
    eda1, eda2, eda3, eda4, eda5 = st.tabs(["Preview & Stats", "Distributions", "Correlation", "Income Breakdown", "Data Quality"])

    with eda1:
        ca, cb = st.columns([3, 1])
        with ca:
            st.markdown("**Dataset Preview** (first 50 rows)")
            st.dataframe(df.drop(columns=['income_numeric'], errors='ignore').head(50), use_container_width=True)
        with cb:
            st.markdown("**Quick Stats**")
            nd = df.select_dtypes(include=np.number)
            for col in nd.columns[:6]:
                st.html(f"""<div class="ib" style="padding:0.5rem 0.8rem;margin-bottom:0.3rem;">
                  <div><div class="ib-title" style="font-size:0.75rem;">{col}</div>
                  <div class="ib-text" style="font-size:0.69rem;">μ {nd[col].mean():,.1f} · σ {nd[col].std():,.1f}</div></div></div>""")
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with eda2:
        cs, cp = st.columns([1, 3])
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        with cs:
            chart_type = st.radio("Chart type", ["Histogram", "Categorical bar", "Box plot", "Violin plot"])
            if chart_type in ["Histogram", "Box plot", "Violin plot"]:
                col_to_plot = st.selectbox("Column", num_cols)
                split_income = st.checkbox("Split by income group", value=True)
            else:
                col_to_plot = st.selectbox("Column", cat_cols)
                split_income = False
        with cp:
            if chart_type == "Histogram":
                fig, ax = dark_fig(9, 4)
                if split_income and 'income' in df.columns:
                    for grp, col in zip(df['income'].unique(), [ACC, ACC2]):
                        vals = df.loc[df['income'] == grp, col_to_plot].dropna()
                        ax.hist(vals, bins=40, alpha=0.6, color=col, label=grp, edgecolor="none")
                    ax.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=9)
                else:
                    ax.hist(df[col_to_plot].dropna(), bins=40, color=ACC, edgecolor="none", alpha=0.85)
                ax.set_xlabel(col_to_plot); ax.set_ylabel("Count"); ax.set_title(f"Distribution of {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_type == "Box plot":
                fig, ax = dark_fig(9, 4)
                if split_income and 'income' in df.columns:
                    groups = [df.loc[df['income'] == g, col_to_plot].dropna() for g in df['income'].unique()]
                    labels = df['income'].unique().tolist()
                    bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=True,
                                    medianprops=dict(color=ACC2, linewidth=2))
                    for patch, c in zip(bp['boxes'], [ACC, ACC2]):
                        patch.set_facecolor(c); patch.set_alpha(0.45)
                    for el in ['whiskers', 'caps', 'fliers']:
                        for it in bp[el]: it.set_color(MUTED_PY)
                else:
                    bp = ax.boxplot(df[col_to_plot].dropna(), patch_artist=True, notch=True,
                                    medianprops=dict(color=ACC2, linewidth=2))
                    bp['boxes'][0].set_facecolor(ACC); bp['boxes'][0].set_alpha(0.45)
                ax.set_ylabel(col_to_plot); ax.set_title(f"Box Plot — {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_type == "Violin plot":
                fig, ax = dark_fig(9, 4)
                if split_income and 'income' in df.columns:
                    groups = [df.loc[df['income'] == g, col_to_plot].dropna().values for g in df['income'].unique()]
                    vp = ax.violinplot(groups, showmedians=True)
                    for body, col in zip(vp['bodies'], [ACC, ACC2]):
                        body.set_facecolor(col); body.set_alpha(0.5)
                    vp['cmedians'].set_color(MUTED_PY)
                    ax.set_xticks([1, 2]); ax.set_xticklabels(df['income'].unique().tolist())
                else:
                    vp = ax.violinplot([df[col_to_plot].dropna().values], showmedians=True)
                    vp['bodies'][0].set_facecolor(ACC); vp['bodies'][0].set_alpha(0.5)
                    vp['cmedians'].set_color(MUTED_PY)
                ax.set_ylabel(col_to_plot); ax.set_title(f"Violin Plot — {col_to_plot}")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            else:
                counts = df[col_to_plot].value_counts()
                fig, ax = dark_fig(9, max(3, len(counts) * 0.42))
                colors = plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.85, len(counts)))
                bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none")
                ax.set_xlabel("Count"); ax.set_title(f"Counts by {col_to_plot}"); ax.invert_yaxis()
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_width() + counts.values.max() * 0.01,
                            bar.get_y() + bar.get_height() / 2, f"{val:,}",
                            va='center', color=MUTED_PY, fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with eda3:
        nd = df.select_dtypes(include=np.number)
        corr = nd.corr(); n = len(corr)
        fig, ax = dark_fig(8, 6)
        im = ax.imshow(corr.values, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center',
                        color=TEXT_PY if abs(corr.values[i, j]) < 0.5 else '#fff', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors=MUTED_PY)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with eda4:
        if 'income' in df.columns and 'occupation' in df.columns:
            fig, axes = dark_fig(12, 9, nrows=2, ncols=2)
            ax1, ax2, ax3, ax4 = axes.flatten()
            oi = df.groupby('occupation')['income_numeric'].mean().sort_values()
            ax1.barh(oi.index, oi.values, color=[ACC2 if v >= 50000 else ACC for v in oi.values], edgecolor="none")
            ax1.axvline(oi.mean(), color=ACC3, linestyle='--', linewidth=1, label='Mean')
            ax1.set_xlabel("Avg Income ($)"); ax1.set_title("Avg Income by Occupation")
            ax1.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=8)
            sample = df.sample(min(2000, len(df)), random_state=42)
            ax2.scatter(sample['age'], sample['hours-per-week'], alpha=0.22, s=10,
                        c=[ACC if v == 75000 else ACC2 for v in sample['income_numeric']], edgecolors='none')
            ax2.set_xlabel("Age"); ax2.set_ylabel("Hours / Week"); ax2.set_title("Age vs Hours")
            ax2.legend(handles=[mpatches.Patch(color=ACC, label='>50K'), mpatches.Patch(color=ACC2, label='≤50K')],
                       facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=8)
            if 'educational-num' in df.columns:
                ei = df.groupby('educational-num')['income_numeric'].mean()
                ax3.bar(ei.index, ei.values,
                        color=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.85, len(ei))), edgecolor="none")
                ax3.set_xlabel("Education Level (1–16)"); ax3.set_ylabel("Avg Income ($)"); ax3.set_title("Education vs Avg Income")
            if 'gender' in df.columns:
                gi = df.groupby('gender')['income_numeric'].mean()
                bars = ax4.bar(gi.index, gi.values, color=[ACC, ACC2], edgecolor="none", width=0.4)
                ax4.set_ylabel("Avg Income ($)"); ax4.set_title("Income by Gender")
                for bar in bars:
                    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 400,
                             f"${bar.get_height():,.0f}", ha='center', va='bottom', color=TEXT_PY, fontsize=9)
            plt.tight_layout(pad=2.0); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Income breakdown requires income and occupation columns.")

    with eda5:
        st.markdown("**Data Quality Report**")
        missing = df.isnull().sum()
        quality_df = pd.DataFrame({
            'Missing': missing,
            'Missing %': (missing / len(df) * 100).round(2),
            'Dtype': df.dtypes,
            'Unique Values': df.nunique()
        })
        st.dataframe(quality_df, use_container_width=True)
        if missing.sum() > 0:
            fig, ax = dark_fig(9, 3)
            ax.bar(missing.index, missing.values, color=ACC, edgecolor="none", alpha=0.8)
            ax.set_xlabel("Column"); ax.set_ylabel("Missing Count"); ax.set_title("Missing Values per Column")
            plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.success("✅ No missing values detected.")
        if 'income' in df.columns:
            st.markdown("**Class Balance**")
            vc = df['income'].value_counts()
            fig, ax = dark_fig(6, 3)
            ax.pie(vc.values, labels=vc.index, colors=[ACC, ACC2], autopct='%1.1f%%',
                   startangle=90, wedgeprops=dict(edgecolor=SURFACE_PY, linewidth=2))
            ax.set_title("Income Class Distribution")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ═══════════════════════════════════════════════
with tab_train:
    st.html('<div class="sh"><span class="sn">3</span> Train the ML Model</div>')
    if 'income_numeric' not in df.columns:
        st.error("⚠️ `income_numeric` column missing. Ensure CSV has `income` column.")
        st.stop()

    X = df.drop(columns=[c for c in ['income', 'income_numeric'] if c in df.columns])
    y = df['income_numeric']
    cat_f = X.select_dtypes(include='object').columns.tolist()
    num_f = X.select_dtypes(include=np.number).columns.tolist()
    prep = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_f),
        ('num', 'passthrough', num_f)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_seed))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training samples", f"{len(X_train):,}")
    c2.metric("Testing samples",  f"{len(X_test):,}")
    c3.metric("Cat. features",    len(cat_f))
    c4.metric("Num. features",    len(num_f))

    st.html(f"""<div class="gc-sm" style="margin:0.8rem 0;">
      <div style="font-size:0.7rem;color:var(--ink-mu);margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.09em;">Ready to train</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:var(--si);">{MODEL_ICONS[model_choice]} {model_choice}</div>
      <div style="font-size:0.76rem;color:var(--ink-mu);margin-top:0.2rem;">{len(cat_f)} categorical · {len(num_f)} numerical · {test_size * 100:.0f}% test · seed {int(random_seed)}</div>
    </div>""")

    if st.button("🚀 Train Model Now", use_container_width=True):
        with st.spinner(f"Training {model_choice}…"):
            model = train_model(X_train, y_train, prep, model_type=model_choice, seed=int(random_seed))
        st.session_state.update({'model': model, 'X_columns': X.columns.tolist(), 'df': df,
                                  'X_test': X_test, 'y_test': y_test, 'cat_f': cat_f, 'num_f': num_f})
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        st.session_state['metrics'] = dict(mae=mae, rmse=rmse, r2=r2)

        if show_cv:
            with st.spinner("Running 5-fold cross-validation…"):
                full_model = train_model(X, y, prep, model_type=model_choice, seed=int(random_seed))
                cv_scores  = cross_val_score(full_model, X, y, cv=5, scoring='r2', n_jobs=-1)
                st.session_state['cv_scores'] = cv_scores
        st.success(f"✅ {model_choice} trained successfully!")

    if 'metrics' in st.session_state:
        m = st.session_state['metrics']
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",      f"${m['mae']:,.0f}",   "avg absolute error")
        c2.metric("RMSE",     f"${m['rmse']:,.0f}",  "root mean squared")
        c3.metric("R² Score", f"{m['r2']:.3f}",      "1.0 = perfect fit")

        r2_pct = int(m['r2'] * 100)
        ring_color = ACC2 if r2_pct >= 80 else (ACC3 if r2_pct >= 60 else ACC)
        quality_label = ("Excellent — model explains most variance." if r2_pct >= 80 else
                         "Good — reasonable predictive power." if r2_pct >= 60 else
                         "Moderate — consider a different algorithm.")

        # R² ring + progress bar
        st.html(f"""<div class="gc-sm" style="display:flex;align-items:center;gap:1.4rem;margin:1rem 0;">
          <div style="width:72px;height:72px;border-radius:50%;border:3px solid {ring_color};
            display:flex;align-items:center;justify-content:center;
            font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{ring_color};flex-shrink:0;">
            {r2_pct}%</div>
          <div style="flex:1;">
            <div style="font-family:'Playfair Display',serif;font-weight:600;color:var(--ink);margin-bottom:0.3rem;">Model Fit Score</div>
            <div style="font-size:0.81rem;color:var(--ink-m);margin-bottom:0.5rem;">{quality_label}</div>
            <div class="pbar-wrap"><div class="pbar-fill" style="width:{r2_pct}%;"></div></div>
          </div>
        </div>""")

        if show_cv and 'cv_scores' in st.session_state:
            cv = st.session_state['cv_scores']
            st.html(f"""<div class="gc-sm" style="margin-bottom:1rem;">
              <div style="font-size:0.7rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">5-Fold Cross Validation R²</div>
              <div style="font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700;color:var(--si);">{cv.mean():.3f}
                <span style="font-size:0.82rem;color:var(--ink-mu);font-family:'DM Sans',sans-serif;">± {cv.std():.3f}</span></div>
              <div style="font-size:0.76rem;color:var(--ink-mu);margin-top:0.25rem;">Folds: {', '.join([f'{s:.3f}' for s in cv])}</div>
            </div>""")

        fi_df = get_feat_imp(st.session_state['model'], cat_f, num_f)
        if fi_df is not None and show_feat_imp:
            st.markdown("**Top 15 Feature Importances**")
            fig, ax = dark_fig(10, 5)
            bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                           color=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.9, len(fi_df)))[::-1],
                           edgecolor="none")
            ax.invert_yaxis(); ax.set_xlabel("Importance"); ax.set_title("Feature Importances (Top 15)")
            for bar, val in zip(bars, fi_df['Importance']):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va='center', color=MUTED_PY, fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        X_te = st.session_state['X_test']; y_te = st.session_state['y_test']
        y_pr = st.session_state['model'].predict(X_te); res = y_te - y_pr

        cl, cr = st.columns(2)
        with cl:
            fig, ax = dark_fig(6, 4)
            jit = np.random.RandomState(0).uniform(-1500, 1500, len(y_te))
            ax.scatter(y_te + jit, y_pr, alpha=0.2, s=10, color=ACC, edgecolors="none")
            mn, mx = min(y_te.min(), y_pr.min()), max(y_te.max(), y_pr.max())
            ax.plot([mn, mx], [mn, mx], color=ACC3, linewidth=1.5, linestyle="--", label="Perfect fit")
            ax.set_xlabel("Actual ($)"); ax.set_ylabel("Predicted ($)"); ax.set_title("Actual vs. Predicted")
            ax.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with cr:
            fig, ax = dark_fig(6, 4)
            ax.hist(res, bins=40, color=ACC2, edgecolor="none", alpha=0.8)
            ax.axvline(0, color=ACC, linestyle="--", linewidth=1.5, label="Zero error")
            ax.set_xlabel("Residual ($)"); ax.set_ylabel("Count"); ax.set_title("Residual Distribution")
            ax.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        sum_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²', 'Model', 'Test Size', 'Train Samples', 'Test Samples'],
            'Value':  [f"${m['mae']:,.0f}", f"${m['rmse']:,.0f}", f"{m['r2']:.4f}",
                       model_choice, f"{test_size * 100:.0f}%", f"{len(X_train):,}", f"{len(X_test):,}"]
        })
        st.download_button("⬇️ Download Model Summary", data=sum_df.to_csv(index=False).encode(),
                           file_name="model_summary.csv", mime="text/csv")
    else:
        st.html("""<div class="es"><div class="es-icon">🤖</div><div class="es-title">No model trained yet</div>
        <div class="es-sub">Click <b>Train Model Now</b> above. Training typically takes 10–30 seconds.</div>
        <div class="es-tip">Tip: Random Forest gives best accuracy on this dataset</div></div>""")


# ═══════════════════════════════════════════════
# TAB 3 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
with tab_predict:
    st.html('<div class="sh"><span class="sn">4</span> Single Employee Prediction</div>')
    if 'model' not in st.session_state:
        st.html("""<div class="es"><div class="es-icon">🔍</div><div class="es-title">Train a model first</div>
        <div class="es-sub">Go to <b>Train Model</b>, click <b>Train Model Now</b>, then return here.</div>
        <div class="es-tip">Predictions include confidence interval and percentile ranking</div></div>""")
    else:
        model  = st.session_state['model']
        X_cols = st.session_state['X_columns']
        df_r   = st.session_state.get('df', df)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.html('<div style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-mu);margin-bottom:0.7rem;">👤 Personal Info</div>')
            age     = st.slider("Age", int(df_r['age'].min()), int(df_r['age'].max()), 30)
            gender  = st.radio("Gender", df_r['gender'].unique().tolist())
            race    = st.selectbox("Race", df_r['race'].unique().tolist())
            native_country = st.selectbox("Native Country", sorted(df_r['native-country'].unique().tolist()))
        with c2:
            st.html('<div style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-mu);margin-bottom:0.7rem;">🎓 Education & Work</div>')
            edu_num    = st.slider("Education Level (1–16)", int(df_r['educational-num'].min()), int(df_r['educational-num'].max()), 10)
            workclass  = st.selectbox("Work Class", df_r['workclass'].unique().tolist())
            occupation = st.selectbox("Occupation", df_r['occupation'].unique().tolist())
            hrs        = st.slider("Hours / Week", int(df_r['hours-per-week'].min()), int(df_r['hours-per-week'].max()), 40)
        with c3:
            st.html('<div style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-mu);margin-bottom:0.7rem;">🏠 Household & Capital</div>')
            marital      = st.selectbox("Marital Status", df_r['marital-status'].unique().tolist())
            relationship = st.selectbox("Relationship", df_r['relationship'].unique().tolist())
            cap_gain     = st.number_input("Capital Gain ($)", min_value=0, max_value=int(df_r['capital-gain'].max()), value=0)
            cap_loss     = st.number_input("Capital Loss ($)", min_value=0, max_value=int(df_r['capital-loss'].max()), value=0)
            fnlwgt       = st.number_input("Final Weight", min_value=int(df_r['fnlwgt'].min()), max_value=int(df_r['fnlwgt'].max()), value=200000)

        if st.button("💡 Predict Salary", use_container_width=True):
            nd = pd.DataFrame([{
                'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt,
                'educational-num': edu_num, 'marital-status': marital,
                'occupation': occupation, 'relationship': relationship,
                'race': race, 'gender': gender, 'capital-gain': cap_gain,
                'capital-loss': cap_loss, 'hours-per-week': hrs,
                'native-country': native_country
            }])[X_cols]
            try:
                pred  = model.predict(nd)[0]
                low, high = pred * 0.85, pred * 1.15
                nat_avg   = df_r['income_numeric'].mean()
                pct       = (df_r['income_numeric'] < pred).mean() * 100
                tier_label, tier_color = salary_tier(pred)

                st.html(f"""<div class="pr gc" style="text-align:center;margin-top:1.4rem;border-color:rgba(139,74,43,0.3);padding:1.8rem;">
                  <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.14em;color:var(--ink-mu);margin-bottom:0.45rem;">Estimated Annual Salary</div>
                  <div style="font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;color:var(--si);line-height:1.1;">${pred:,.0f}</div>
                  {"<div style='font-size:0.82rem;color:var(--ink-m);margin-top:0.4rem;'>Confidence range: $" + f"{low:,.0f} – ${high:,.0f}</div>" if show_confidence else ""}
                  <div style="margin-top:0.75rem;"><span style="background:var(--si-p);border:1px solid var(--bdr-w);border-radius:999px;padding:3px 14px;font-size:0.74rem;color:var(--si);">{tier_label}</span></div>
                </div>""")

                ca, cb, cc = st.columns(3)
                ca.metric("Dataset Average", f"${nat_avg:,.0f}")
                cb.metric("Percentile", f"{pct:.0f}th")
                cc.metric("vs. Average", f"${pred - nat_avg:+,.0f}")

                fig, ax = dark_fig(7, 0.85)
                ax.barh([0], [100], color=CARD_PY, height=0.5, edgecolor="none")
                ax.barh([0], [pct], color=(ACC2 if pct >= 70 else (ACC3 if pct >= 40 else ACC)), height=0.5, edgecolor="none")
                ax.set_xlim(0, 100); ax.set_yticks([])
                ax.set_xlabel("Salary Percentile in Dataset")
                ax.set_title(f"This profile is in the {pct:.0f}th percentile", fontsize=10)
                for sp in ax.spines.values(): sp.set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                if 'pred_history' not in st.session_state:
                    st.session_state['pred_history'] = []
                st.session_state['pred_history'].append({
                    'Age': age, 'Occupation': occupation, 'Hrs/Wk': hrs,
                    'Edu Level': edu_num, 'Predicted ($)': int(pred),
                    'Percentile': f"{pct:.0f}th", 'Tier': tier_label
                })
            except Exception as e:
                st.error(f"Prediction error: {e}")

        if 'pred_history' in st.session_state and st.session_state['pred_history']:
            with st.expander(f"📋 Prediction History ({len(st.session_state['pred_history'])} runs)"):
                hdf = pd.DataFrame(st.session_state['pred_history'])
                st.dataframe(hdf, use_container_width=True)
                if len(hdf) > 1:
                    fig, ax = dark_fig(8, 3)
                    ax.bar(range(len(hdf)), hdf['Predicted ($)'], color=ACC, edgecolor="none", alpha=0.8)
                    ax.axhline(avg_salary, color=ACC2, linestyle='--', linewidth=1.2, label='Dataset avg')
                    ax.set_xticks(range(len(hdf)))
                    ax.set_xticklabels([f"#{i + 1}" for i in range(len(hdf))], fontsize=8)
                    ax.set_ylabel("Predicted Salary ($)"); ax.set_title("Prediction History")
                    ax.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                st.download_button("⬇️ Export History", data=hdf.to_csv(index=False).encode(),
                                   file_name="prediction_history.csv", mime="text/csv")
                if st.button("🗑️ Clear History"):
                    st.session_state['pred_history'] = []; st.rerun()


# ═══════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ═══════════════════════════════════════════════
with tab_batch:
    st.html('<div class="sh"><span class="sn">5</span> Batch Prediction</div>')
    if 'model' not in st.session_state:
        st.html("""<div class="es"><div class="es-icon">📦</div><div class="es-title">No model available</div>
        <div class="es-sub">Train a model in <b>Train Model</b> first, then upload a batch CSV here.</div>
        <div class="es-tip">Batch results include salary tier labels and percentile rankings</div></div>""")
    else:
        model  = st.session_state['model']
        X_cols = st.session_state['X_columns']
        with st.expander("📄 View expected CSV format"):
            sc = "age,workclass,fnlwgt,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country\n35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States\n28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States"
            st.code(sc, language="csv")
            st.download_button("⬇️ Download Sample CSV", data=sc, file_name="sample_batch.csv", mime="text/csv")

        bf = st.file_uploader("Upload batch CSV", type="csv", key="batch_uploader")
        if bf:
            with st.spinner("Running batch predictions…"):
                try:
                    bd = pd.read_csv(bf)
                    for col in bd.select_dtypes(include='object').columns:
                        bd[col] = bd[col].str.strip()
                    preds = model.predict(bd[X_cols])
                    bd['Predicted_Salary'] = preds.round(0).astype(int)
                    bd['Salary_Tier']      = [salary_tier(p)[0] for p in preds]
                    bd['Percentile']       = [f"{(df['income_numeric'] < p).mean() * 100:.0f}th" for p in preds]
                    bd['vs_Avg']           = [f"${p - avg_salary:+,.0f}" for p in preds]
                    st.success(f"✅ Predicted salaries for {len(bd):,} records.")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Average",  f"${preds.mean():,.0f}")
                    c2.metric("Median",   f"${np.median(preds):,.0f}")
                    c3.metric("Min",      f"${preds.min():,.0f}")
                    c4.metric("Max",      f"${preds.max():,.0f}")

                    cl, cr = st.columns([2, 1])
                    with cl:
                        fig, ax = dark_fig(7, 3.5)
                        ax.hist(preds, bins=30, color=ACC, edgecolor="none", alpha=0.85)
                        ax.axvline(preds.mean(),     color=ACC2, linewidth=1.5, linestyle="--", label=f"Mean ${preds.mean():,.0f}")
                        ax.axvline(np.median(preds), color=ACC3, linewidth=1.5, linestyle=":",  label=f"Median ${np.median(preds):,.0f}")
                        ax.set_xlabel("Predicted Salary ($)"); ax.set_ylabel("Count"); ax.set_title("Batch Salary Distribution")
                        ax.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=9)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                    with cr:
                        st.markdown("**Tier Breakdown**")
                        for tier, count in bd['Salary_Tier'].value_counts().items():
                            pct_t = count / len(bd) * 100
                            st.html(f"""<div class="gc-sm" style="padding:0.6rem 0.85rem;margin-bottom:0.35rem;">
                              <div style="display:flex;justify-content:space-between;margin-bottom:0.28rem;">
                                <span style="font-size:0.8rem;">{tier}</span>
                                <span style="font-size:0.74rem;color:var(--si);">{count} ({pct_t:.0f}%)</span>
                              </div>
                              <div class="pbar-wrap"><div class="pbar-fill" style="width:{pct_t}%;"></div></div>
                            </div>""")

                    st.dataframe(bd, use_container_width=True)
                    st.download_button("⬇️ Download Predictions CSV",
                                       data=bd.to_csv(index=False).encode('utf-8'),
                                       file_name='predicted_salaries.csv', mime='text/csv')
                except KeyError as ke:
                    st.error(f"Missing column: {ke}")
                except Exception as e:
                    st.error(f"Batch error: {e}")
        else:
            st.html("""<div class="es" style="margin-top:1rem;"><div class="es-icon">🗂️</div><div class="es-title">Upload a batch file</div>
            <div class="es-sub">Drop a CSV above. Click <b>View expected CSV format</b> for a sample template.</div></div>""")


# ═══════════════════════════════════════════════
# TAB 5 — SALARY INSIGHTS
# ═══════════════════════════════════════════════
with tab_insights:
    st.html('<div class="sh"><span class="sn">6</span> Salary Insights & Intelligence</div>')
    if 'income_numeric' not in df.columns:
        st.info("Load a dataset with income data to see insights.")
    else:
        high_df = df[df['income_numeric'] == 75000]
        low_df  = df[df['income_numeric'] == 25000]
        top_occ   = high_df['occupation'].mode()[0]    if 'occupation'      in df.columns else "N/A"
        top_wc    = high_df['workclass'].mode()[0]     if 'workclass'       in df.columns else "N/A"
        avg_edu_h = high_df['educational-num'].mean()  if 'educational-num' in df.columns else 0
        avg_edu_l = low_df['educational-num'].mean()   if 'educational-num' in df.columns else 0
        avg_hrs_h = high_df['hours-per-week'].mean()   if 'hours-per-week'  in df.columns else 0
        avg_hrs_l = low_df['hours-per-week'].mean()    if 'hours-per-week'  in df.columns else 0

        for icon, title, body in [
            ("💼", "Top High-Earning Occupation", f"<b style='color:var(--ink)'>{top_occ}</b> is the most common occupation among high earners."),
            ("🏢", "Dominant Work Class", f"High earners most frequently work in <b style='color:var(--ink)'>{top_wc}</b>."),
            ("🎓", "Education Premium", f"High earners avg edu level <b style='color:var(--si)'>{avg_edu_h:.1f}/16</b> vs <b style='color:var(--ink-mu)'>{avg_edu_l:.1f}/16</b> — a <b style='color:var(--sage)'>+{avg_edu_h - avg_edu_l:.1f} level gap</b>."),
            ("⏱️", "Hours Worked", f"High earners average <b style='color:var(--si)'>{avg_hrs_h:.0f} hrs/week</b> vs <b style='color:var(--ink-mu)'>{avg_hrs_l:.0f} hrs/week</b> for low earners."),
            ("👥", "Dataset Composition", f"<b style='color:var(--sage)'>{pct_high:.1f}%</b> high earners ({high_earners:,} records) and <b style='color:var(--ink-mu)'>{100 - pct_high:.1f}%</b> low earners."),
        ]:
            st.html(f"""<div class="ib"><div class="ib-icon">{icon}</div>
              <div><div class="ib-title">{title}</div><div class="ib-text">{body}</div></div></div>""")

        st.html('<div class="sr"></div>')
        st.markdown("#### 📊 Salary Ranking by Category")
        rank_col = st.selectbox("Rank by", [c for c in ['occupation', 'workclass', 'marital-status', 'race', 'gender'] if c in df.columns])
        grp = df.groupby(rank_col)['income_numeric'].agg(['mean', 'std', 'count']).sort_values('mean')
        fig, ax = dark_fig(10, max(4, df[rank_col].nunique() * 0.4))
        bars = ax.barh(grp.index, grp['mean'],
                       color=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3, 0.88, len(grp))), edgecolor="none")
        ax.axvline(df['income_numeric'].mean(), color=ACC2, linestyle='--', linewidth=1.2, label="Dataset mean")
        ax.set_xlabel("Average Salary ($)"); ax.set_title(f"Average Salary by {rank_col.title()}")
        ax.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=9)
        for bar, (idx, row) in zip(bars, grp.iterrows()):
            ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                    f"${row['mean']:,.0f} (n={int(row['count'])})", va='center', color=MUTED_PY, fontsize=7.5)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.html('<div class="sr"></div>')
        st.markdown("#### 🏆 High vs Low Earner Profile Comparison")
        profile_cols = [c for c in ['age', 'educational-num', 'hours-per-week', 'capital-gain'] if c in df.columns]
        comp_data = [{'Feature': col,
                      'High Earners (>50K)': f"{high_df[col].mean():.1f}",
                      'Low Earners (≤50K)':  f"{low_df[col].mean():.1f}"} for col in profile_cols]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 6 — SALARY SIMULATOR
# ═══════════════════════════════════════════════
with tab_sim:
    st.html('<div class="sh"><span class="sn">7</span> Salary Simulator</div>')
    if not show_salary_sim:
        st.info("Enable **Salary Simulator** in the ☰ menu to use this tool.")
    elif 'model' not in st.session_state:
        st.html("""<div class="es"><div class="es-icon">🎯</div><div class="es-title">Train a model first</div>
        <div class="es-sub">The salary simulator needs a trained model. Go to <b>Train Model</b> tab first.</div></div>""")
    else:
        st.html("""<div class="gc-sm" style="margin-bottom:1.1rem;">
          <div class="ib-title" style="margin-bottom:0.25rem;">Interactive Career Planner</div>
          <div class="ib-text">Adjust sliders to simulate how changes in education, hours worked, or age affect predicted salary.</div>
        </div>""")
        model  = st.session_state['model']
        X_cols = st.session_state['X_columns']
        df_r   = st.session_state.get('df', df)

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("**Base Profile**")
            s_occ     = st.selectbox("Occupation",     df_r['occupation'].unique().tolist(),    key="sim_occ")
            s_wc      = st.selectbox("Work Class",     df_r['workclass'].unique().tolist(),     key="sim_wc")
            s_mar     = st.selectbox("Marital Status", df_r['marital-status'].unique().tolist(),key="sim_mar")
            s_rel     = st.selectbox("Relationship",   df_r['relationship'].unique().tolist(),  key="sim_rel")
            s_gen     = st.radio("Gender",             df_r['gender'].unique().tolist(),        key="sim_gen")
            s_race    = st.selectbox("Race",           df_r['race'].unique().tolist(),          key="sim_race")
            s_country = st.selectbox("Country",        sorted(df_r['native-country'].unique().tolist()), key="sim_country")

        with col_r:
            st.markdown("**Simulate Changes**")
            s_age = st.slider("Age", int(df_r['age'].min()), int(df_r['age'].max()), 35, key="sim_age")
            s_edu = st.slider("Education Level (1–16)", 1, 16, 10, key="sim_edu")
            s_hrs = st.slider("Hours per Week", 10, 80, 40, key="sim_hrs")
            s_cap = st.slider("Capital Gain ($)", 0, int(df_r['capital-gain'].max()), 0, key="sim_cap")

            def sim_predict(age, edu, hrs, cap):
                nd = pd.DataFrame([{
                    'age': age, 'workclass': s_wc, 'fnlwgt': 200000,
                    'educational-num': edu, 'marital-status': s_mar,
                    'occupation': s_occ, 'relationship': s_rel,
                    'race': s_race, 'gender': s_gen, 'capital-gain': cap,
                    'capital-loss': 0, 'hours-per-week': hrs,
                    'native-country': s_country
                }])[X_cols]
                return model.predict(nd)[0]

            try:
                base_pred = sim_predict(s_age, s_edu, s_hrs, s_cap)
                tier_label, _ = salary_tier(base_pred)
                pct_above = (df_r['income_numeric'] < base_pred).mean() * 100

                st.html(f"""<div class="gc-sm" style="text-align:center;margin-top:1rem;border-color:rgba(139,74,43,0.25);">
                  <div style="font-size:0.68rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.28rem;">Predicted Salary</div>
                  <div style="font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:700;color:var(--si);">${base_pred:,.0f}</div>
                  <div style="font-size:0.76rem;color:var(--ink-mu);margin-top:0.25rem;">{tier_label} · {pct_above:.0f}th percentile</div>
                </div>""")

                st.markdown("**Sensitivity Analysis**")
                edu_range  = range(1, 17)
                hrs_range  = range(20, 81, 5)
                age_range  = range(20, 65, 5)
                edu_preds  = [sim_predict(s_age, e, s_hrs, s_cap) for e in edu_range]
                hrs_preds  = [sim_predict(s_age, s_edu, h, s_cap) for h in hrs_range]
                age_preds  = [sim_predict(a, s_edu, s_hrs, s_cap) for a in age_range]

                fig, axes = dark_fig(12, 3.5, nrows=1, ncols=3)
                ax1, ax2, ax3 = axes.flatten()
                ax1.plot(list(edu_range), edu_preds, color=ACC,  linewidth=2, marker='o', markersize=4)
                ax1.axvline(s_edu, color=ACC3, linestyle='--', linewidth=1)
                ax1.set_xlabel("Education Level"); ax1.set_ylabel("Salary ($)"); ax1.set_title("Education Impact")
                ax2.plot(list(hrs_range), hrs_preds, color=ACC2, linewidth=2, marker='o', markersize=4)
                ax2.axvline(s_hrs, color=ACC3, linestyle='--', linewidth=1)
                ax2.set_xlabel("Hours / Week"); ax2.set_ylabel("Salary ($)"); ax2.set_title("Hours Impact")
                ax3.plot(list(age_range), age_preds, color=ACC4, linewidth=2, marker='o', markersize=4)
                ax3.axvline(s_age, color=ACC3, linestyle='--', linewidth=1)
                ax3.set_xlabel("Age"); ax3.set_ylabel("Salary ($)"); ax3.set_title("Age Impact")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.error(f"Simulation error: {e}")


# ═══════════════════════════════════════════════
# TAB 7 — WHAT-IF
# ═══════════════════════════════════════════════
with tab_whatif:
    st.html('<div class="sh"><span class="sn">8</span> What-If Analysis</div>')
    if 'model' not in st.session_state:
        st.html("""<div class="es"><div class="es-icon">📈</div><div class="es-title">Train a model first</div>
        <div class="es-sub">What-If analysis needs a trained model. Go to <b>Train Model</b> tab first.</div></div>""")
    else:
        st.html("""<div class="gc-sm" style="margin-bottom:1.1rem;">
          <div class="ib-title" style="margin-bottom:0.25rem;">Side-by-Side Scenario Comparison</div>
          <div class="ib-text">Define two employee profiles and compare their predicted salaries instantly.</div>
        </div>""")
        model  = st.session_state['model']
        X_cols = st.session_state['X_columns']
        df_r   = st.session_state.get('df', df)

        col_a, col_b = st.columns(2)
        profiles = {}
        for col, label, key_prefix, default_age, default_edu in [
            (col_a, "👤 Person A", "pa", 35, 10),
            (col_b, "👤 Person B", "pb", 45, 14)
        ]:
            with col:
                st.markdown(f"**{label}**")
                p = {
                    'age':            st.slider("Age", int(df_r['age'].min()), int(df_r['age'].max()), default_age, key=f"{key_prefix}_age"),
                    'workclass':      st.selectbox("Work Class",     df_r['workclass'].unique().tolist(),      key=f"{key_prefix}_wc"),
                    'fnlwgt':         200000,
                    'educational-num':st.slider("Education Level", 1, 16, default_edu,                        key=f"{key_prefix}_edu"),
                    'marital-status': st.selectbox("Marital Status", df_r['marital-status'].unique().tolist(),key=f"{key_prefix}_mar"),
                    'occupation':     st.selectbox("Occupation",     df_r['occupation'].unique().tolist(),     key=f"{key_prefix}_occ"),
                    'relationship':   st.selectbox("Relationship",   df_r['relationship'].unique().tolist(),   key=f"{key_prefix}_rel"),
                    'race':           st.selectbox("Race",           df_r['race'].unique().tolist(),           key=f"{key_prefix}_race"),
                    'gender':         st.radio("Gender",             df_r['gender'].unique().tolist(),         key=f"{key_prefix}_gen"),
                    'capital-gain':   st.slider("Capital Gain ($)", 0, int(df_r['capital-gain'].max()), 0,     key=f"{key_prefix}_cap"),
                    'capital-loss':   0,
                    'hours-per-week': st.slider("Hours/Week", 10, 80, 40,                                      key=f"{key_prefix}_hrs"),
                    'native-country': st.selectbox("Country",        sorted(df_r['native-country'].unique().tolist()), key=f"{key_prefix}_ctry"),
                }
                profiles[label] = p

        if st.button("⚖️ Compare Profiles", use_container_width=True):
            try:
                results = {}
                for label, p in profiles.items():
                    nd   = pd.DataFrame([p])[X_cols]
                    pred = model.predict(nd)[0]
                    tier_label, _ = salary_tier(pred)
                    pct  = (df_r['income_numeric'] < pred).mean() * 100
                    results[label] = {'pred': pred, 'tier': tier_label, 'pct': pct}

                ca, cb = st.columns(2)
                cols_map = {list(profiles.keys())[0]: ca, list(profiles.keys())[1]: cb}
                for label, res in results.items():
                    with cols_map[label]:
                        st.html(f"""<div class="gc-sm" style="text-align:center;border-color:rgba(139,74,43,0.25);margin-top:1rem;">
                          <div style="font-size:0.68rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.28rem;">{label}</div>
                          <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:var(--si);">${res['pred']:,.0f}</div>
                          <div style="font-size:0.76rem;color:var(--ink-mu);margin-top:0.25rem;">{res['tier']} · {res['pct']:.0f}th percentile</div>
                        </div>""")

                preds  = [r['pred'] for r in results.values()]
                diff   = abs(preds[0] - preds[1])
                higher = list(results.keys())[0] if preds[0] >= preds[1] else list(results.keys())[1]
                st.html(f"""<div class="ib" style="margin-top:1.1rem;">
                  <div class="ib-icon">💡</div>
                  <div><div class="ib-title">Comparison Result</div>
                  <div class="ib-text"><b style="color:var(--si)">{higher}</b> earns <b style="color:var(--si)">${diff:,.0f} more</b> per year.</div></div>
                </div>""")

                fig, ax = dark_fig(6, 3)
                labels_list = list(results.keys())
                pred_list   = [results[l]['pred'] for l in labels_list]
                bars = ax.bar(labels_list, pred_list, color=[ACC, ACC2], edgecolor="none", width=0.4)
                ax.axhline(df_r['income_numeric'].mean(), color=ACC3, linestyle='--', linewidth=1.2, label='Dataset avg')
                ax.set_ylabel("Predicted Salary ($)"); ax.set_title("Profile Salary Comparison")
                for bar, val in zip(bars, pred_list):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                            f"${val:,.0f}", ha='center', va='bottom', color=TEXT_PY, fontsize=10)
                ax.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.error(f"Comparison error: {e}")


# ═══════════════════════════════════════════════
# TAB 8 — SALARY GROWTH PROJECTOR (NEW)
# ═══════════════════════════════════════════════
with tab_growth:
    st.html('<div class="sh"><span class="sn">9</span> Salary Growth Projector</div>')
    st.html("""<div class="gc-sm" style="margin-bottom:1.2rem;">
      <div class="ib-title" style="margin-bottom:0.25rem;">📈 Career Growth Forecast</div>
      <div class="ib-text">Model your salary trajectory over time using adjustable growth rates. Compare multiple scenarios side-by-side.</div>
    </div>""")

    c1, c2, c3 = st.columns(3)
    with c1:
        base_sal    = st.number_input("Starting Salary ($)", min_value=10000, max_value=500000, value=int(avg_salary), step=1000)
        proj_years  = st.slider("Projection Years", 5, 30, 10)
    with c2:
        growth_rate1 = st.slider("Conservative Growth (%/yr)", 1, 10, 3) / 100
        growth_rate2 = st.slider("Moderate Growth (%/yr)",     1, 15, 6) / 100
    with c3:
        growth_rate3 = st.slider("Aggressive Growth (%/yr)", 1, 20, 10) / 100
        inflation    = st.slider("Inflation Rate (%/yr)",    1,  8,  3) / 100

    years   = list(range(proj_years + 1))
    proj1   = salary_growth_projection(base_sal, proj_years, growth_rate1)
    proj2   = salary_growth_projection(base_sal, proj_years, growth_rate2)
    proj3   = salary_growth_projection(base_sal, proj_years, growth_rate3)
    real1   = [p / ((1 + inflation) ** y) for y, p in zip(years, proj1)]
    real3   = [p / ((1 + inflation) ** y) for y, p in zip(years, proj3)]

    fig, axes = dark_fig(12, 4.5, nrows=1, ncols=2)
    ax1, ax2  = axes.flatten()

    ax1.plot(years, proj1, color=MUTED_PY, linewidth=2, linestyle="--", label=f"Conservative ({growth_rate1*100:.0f}%)")
    ax1.plot(years, proj2, color=ACC3,     linewidth=2.5, label=f"Moderate ({growth_rate2*100:.0f}%)")
    ax1.plot(years, proj3, color=ACC2,     linewidth=2.5, label=f"Aggressive ({growth_rate3*100:.0f}%)")
    ax1.fill_between(years, proj1, proj3, alpha=0.1, color=ACC3)
    ax1.axhline(base_sal, color=ACC, linestyle=":", linewidth=1, alpha=0.6)
    ax1.set_xlabel("Years"); ax1.set_ylabel("Salary ($)"); ax1.set_title("Salary Projection Scenarios")
    ax1.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2.plot(years, proj2, color=ACC3, linewidth=2.5,   label="Nominal (Moderate)")
    ax2.plot(years, real1, color=MUTED_PY, linewidth=2, linestyle="--", label=f"Real (Conservative, {inflation*100:.0f}% inflation)")
    ax2.plot(years, real3, color=ACC2,     linewidth=2, linestyle="-.", label=f"Real (Aggressive, {inflation*100:.0f}% inflation)")
    ax2.set_xlabel("Years"); ax2.set_ylabel("Salary ($)"); ax2.set_title("Nominal vs. Real Salary")
    ax2.legend(facecolor=SURFACE_PY, edgecolor=BORDER_PY, labelcolor=TEXT_PY, fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Summary table
    st.markdown("**Projection Summary**")
    summary_rows = []
    for yr in [1, 3, 5, 10, proj_years]:
        if yr <= proj_years:
            summary_rows.append({
                'Year': f"Year {yr}",
                f"Conservative ({growth_rate1*100:.0f}%)": f"${proj1[yr]:,.0f}",
                f"Moderate ({growth_rate2*100:.0f}%)":     f"${proj2[yr]:,.0f}",
                f"Aggressive ({growth_rate3*100:.0f}%)":   f"${proj3[yr]:,.0f}",
                f"Real Value (Mod., {inflation*100:.0f}% inf.)": f"${proj2[yr]/((1+inflation)**yr):,.0f}",
            })
    proj_df = pd.DataFrame(summary_rows)
    st.dataframe(proj_df, use_container_width=True, hide_index=True)

    # Milestones
    st.markdown("**💰 Salary Milestones** (Moderate scenario)")
    milestones = [50000, 75000, 100000, 150000, 200000]
    for ms in milestones:
        if proj2[-1] >= ms:
            yr_reach = next((y for y, p in enumerate(proj2) if p >= ms), None)
            if yr_reach is not None:
                st.html(f"""<div class="ib" style="padding:0.55rem 0.9rem;margin-bottom:0.3rem;">
                  <div class="ib-icon">🎯</div>
                  <div><div class="ib-title" style="font-size:0.82rem;">${ms:,} milestone</div>
                  <div class="ib-text" style="font-size:0.76rem;">Reached in Year {yr_reach} (${proj2[min(yr_reach,proj_years)]:,.0f})</div></div>
                </div>""")

    export_data = proj_df.to_csv(index=False).encode()
    st.download_button("⬇️ Export Growth Projection", data=export_data, file_name="salary_growth.csv", mime="text/csv")


# ═══════════════════════════════════════════════
# TAB 9 — DATA HEALTH (NEW)
# ═══════════════════════════════════════════════
with tab_health:
    st.html('<div class="sh"><span class="sn">10</span> Data Health Dashboard</div>')

    hs, hl, hc = data_health_score(df)

    col_ring, col_detail = st.columns([1, 3])
    with col_ring:
        ring_style = f"border: 4px solid {hc};"
        st.html(f"""<div style="display:flex;flex-direction:column;align-items:center;gap:0.6rem;padding:1.5rem;">
          <div class="health-ring" style="{ring_style}color:{hc};">{hs}</div>
          <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:600;color:var(--ink);">{hl}</div>
          <div style="font-size:0.75rem;color:var(--ink-mu);text-align:center;">Overall Data<br>Health Score</div>
        </div>""")

    with col_detail:
        # Individual checks
        checks = []

        # Missing values
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        checks.append(("Missing Values", missing_pct == 0, f"{missing_pct:.2f}% missing overall"))

        # Class balance
        if 'income_numeric' in df.columns:
            vc = df['income_numeric'].value_counts(normalize=True)
            ratio = vc.iloc[0]
            checks.append(("Class Balance", 0.3 <= ratio <= 0.7, f"Majority class: {ratio*100:.1f}%"))

        # Row count
        checks.append(("Sufficient Data", n_rows >= 5000, f"{n_rows:,} records ({'>=' if n_rows >= 5000 else '<'} 5,000 recommended)"))

        # Duplicate rows
        dup_count = df.duplicated().sum()
        checks.append(("No Duplicates", dup_count == 0, f"{dup_count:,} duplicate rows found"))

        # Numeric range sanity
        age_ok = df['age'].between(15, 90).all() if 'age' in df.columns else True
        checks.append(("Age Range Sane", age_ok, "All ages between 15–90" if age_ok else "Some ages outside expected range"))

        hrs_ok = df['hours-per-week'].between(1, 100).all() if 'hours-per-week' in df.columns else True
        checks.append(("Hours/Week Sane", hrs_ok, "All hours-per-week between 1–100" if hrs_ok else "Some hours outside expected range"))

        for name, passed, detail in checks:
            icon  = "✅" if passed else "⚠️"
            color = ACC2 if passed else ACC3
            st.html(f"""<div class="ib" style="padding:0.6rem 0.9rem;margin-bottom:0.35rem;border-color:{'rgba(92,122,94,0.3)' if passed else 'rgba(184,115,51,0.3)'};">
              <div class="ib-icon">{icon}</div>
              <div>
                <div class="ib-title" style="font-size:0.82rem;color:{color};">{name}</div>
                <div class="ib-text" style="font-size:0.75rem;">{detail}</div>
              </div>
            </div>""")

    st.html('<div class="sr"></div>')

    # Column-level stats
    st.markdown("**Column Health Report**")
    col_health = []
    for col in df.columns:
        miss = df[col].isnull().sum()
        uniq = df[col].nunique()
        dtype = str(df[col].dtype)
        issues = []
        if miss > 0:              issues.append(f"{miss} missing")
        if uniq == 1:             issues.append("constant")
        if uniq == len(df):       issues.append("all unique (ID-like)")
        col_health.append({
            'Column':         col,
            'Type':           dtype,
            'Unique Values':  uniq,
            'Missing':        miss,
            'Missing %':      f"{miss/len(df)*100:.1f}%",
            'Status':         "⚠️ " + "; ".join(issues) if issues else "✅ OK"
        })
    st.dataframe(pd.DataFrame(col_health), use_container_width=True, hide_index=True)

    # Distribution skewness
    st.markdown("**Numeric Column Skewness**")
    num_df = df.select_dtypes(include=np.number)
    skew_data = []
    for col in num_df.columns:
        sk = num_df[col].skew()
        skew_data.append({'Column': col, 'Skewness': round(sk, 3),
                          'Verdict': "Symmetric" if abs(sk) < 0.5 else ("Moderate" if abs(sk) < 1 else "Highly Skewed")})
    skew_df = pd.DataFrame(skew_data).sort_values('Skewness', key=abs, ascending=False)
    st.dataframe(skew_df, use_container_width=True, hide_index=True)

    fig, ax = dark_fig(9, 3)
    colors = [ACC2 if abs(s) < 0.5 else (ACC3 if abs(s) < 1 else ACC) for s in skew_df['Skewness']]
    ax.barh(skew_df['Column'], skew_df['Skewness'], color=colors, edgecolor="none")
    ax.axvline(0, color=MUTED_PY, linewidth=1, linestyle="--")
    ax.set_xlabel("Skewness"); ax.set_title("Feature Skewness (green = symmetric)")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Export
    health_export = pd.DataFrame(col_health)
    st.download_button("⬇️ Export Health Report", data=health_export.to_csv(index=False).encode(),
                       file_name="data_health_report.csv", mime="text/csv")


# ═══════════════════════════════════════════════
# TAB 10 — COMPARE MODELS
# ═══════════════════════════════════════════════
with tab_compare:
    st.html('<div class="sh"><span class="sn">11</span> Compare Models</div>')
    if not enable_comparison:
        st.info("Enable **Model comparison** in the ☰ menu to use this tab.")
    elif 'model' not in st.session_state:
        st.html("""<div class="es"><div class="es-icon">⚖️</div><div class="es-title">Train a model first</div>
        <div class="es-sub">Train at least one model, then enable model comparison in the ☰ menu.</div></div>""")
    else:
        st.html("""<div class="gc-sm" style="margin-bottom:1.1rem;">
          <div class="ib-title" style="margin-bottom:0.25rem;">Multi-Algorithm Benchmark</div>
          <div class="ib-text">Train all 5 models on identical data splits and compare MAE, RMSE, and R² side-by-side.</div>
        </div>""")

        if st.button("🏆 Run Full Model Comparison", use_container_width=True):
            X_c   = df.drop(columns=[c for c in ['income', 'income_numeric'] if c in df.columns])
            y_c   = df['income_numeric']
            cat_fc = X_c.select_dtypes(include='object').columns.tolist()
            num_fc = X_c.select_dtypes(include=np.number).columns.tolist()
            prep_c = ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_fc),
                ('num', 'passthrough', num_fc)
            ])
            Xtr, Xte, ytr, yte = train_test_split(X_c, y_c, test_size=test_size, random_state=int(random_seed))

            comp_results = []
            prog = st.progress(0, text="Training models…")
            for i, (name, icon, desc) in enumerate(MODELS):
                with st.spinner(f"Training {icon} {name}…"):
                    m  = train_model(Xtr, ytr, prep_c, model_type=name, seed=int(random_seed))
                    yp = m.predict(Xte)
                    comp_results.append({
                        'Model':    f"{icon} {name}",
                        'MAE ($)':  int(mean_absolute_error(yte, yp)),
                        'RMSE ($)': int(np.sqrt(mean_squared_error(yte, yp))),
                        'R²':       round(r2_score(yte, yp), 4),
                    })
                prog.progress((i + 1) / len(MODELS), text=f"Done: {name}")
            prog.empty()

            comp_df = pd.DataFrame(comp_results).sort_values('R²', ascending=False)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            fig, axes = dark_fig(12, 4, nrows=1, ncols=3)
            ax1, ax2, ax3 = axes.flatten()
            colors = plt.cm.get_cmap('YlOrBr')(np.linspace(0.35, 0.85, len(comp_df)))

            for ax, metric, title in [
                (ax1, 'MAE ($)',  'MAE — lower is better'),
                (ax2, 'RMSE ($)', 'RMSE — lower is better'),
                (ax3, 'R²',       'R² — higher is better')
            ]:
                vals = comp_df[metric].values
                bars = ax.barh(comp_df['Model'], vals, color=colors, edgecolor="none")
                ax.invert_yaxis(); ax.set_title(title)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                            f"${val:,.0f}" if metric != 'R²' else f"{val:.3f}",
                            va='center', color=MUTED_PY, fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            best = comp_df.iloc[0]
            st.html(f"""<div class="ib" style="margin-top:0.5rem;">
              <div class="ib-icon">🏆</div>
              <div><div class="ib-title">Best Model: {best['Model']}</div>
              <div class="ib-text">R² = {best['R²']:.4f} · MAE = ${best['MAE ($)']:,} · RMSE = ${best['RMSE ($)']:,}</div></div>
            </div>""")
            st.download_button("⬇️ Download Comparison CSV",
                               data=comp_df.to_csv(index=False).encode(),
                               file_name="model_comparison.csv", mime="text/csv")
