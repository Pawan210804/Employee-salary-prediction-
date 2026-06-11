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
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>

#MainMenu{visibility:hidden;}header{visibility:hidden;}footer{visibility:hidden;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stDecoration"]{display:none!important;}
[data-testid="stStatusWidget"]{display:none!important;}


:root{
  --cr:#F5F0E8;--cr-d:#EDE6D6;--cr-dd:#E3D9C6;
  --parch:#D4C9B0;--ink:#2C2416;--ink-m:#5A5040;
  --ink-mu:#8A7D6A;--ink-g:#B8AA96;
  --si:#8B4A2B;--si-l:#C47B52;--si-p:#F0DDD1;
  --sage:#5C7A5E;--sage-p:#D6E5D7;
  --amb:#B87333;--amb-p:#F2E8D0;
  --bdr:rgba(90,70,50,0.14);--bdr-w:rgba(139,74,43,0.22);
  --glass:rgba(245,240,232,0.82);
  --sh:0 2px 20px rgba(44,36,22,0.08);
  --sh-l:0 8px 36px rgba(44,36,22,0.13);
  --r-sm:8px;--r-md:14px;--r-lg:20px;
  --fd:'Playfair Display',Georgia,serif;
  --fb:'DM Sans',system-ui,sans-serif;
  --fm:'DM Mono',monospace;
}
html,body,[class*="css"]{font-family:var(--fb)!important;background:var(--cr)!important;color:var(--ink)!important;}
.stApp{background:var(--cr)!important;}
.main .block-container{padding-top:0.5rem!important;max-width:1340px;}

/* SIDEBAR */
[data-testid="stSidebar"]{background:var(--cr-d)!important;border-right:1px solid var(--bdr)!important;}
[data-testid="stSidebar"] *{color:var(--ink)!important;}

/* INPUTS */
.stSelectbox select,.stTextInput input,.stNumberInput input{
  background:var(--cr)!important;border:1px solid var(--bdr)!important;
  color:var(--ink)!important;border-radius:var(--r-sm)!important;font-family:var(--fb)!important;}
.stSlider>div>div{color:var(--si)!important;}
.stRadio label,.stCheckbox label{color:var(--ink-m)!important;}
.stFileUploader{background:var(--cr-d)!important;border:1.5px dashed var(--parch)!important;border-radius:var(--r-md)!important;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--cr-d);border-radius:var(--r-md);padding:5px;gap:4px;border:1px solid var(--bdr);}
.stTabs [data-baseweb="tab"]{border-radius:var(--r-sm)!important;color:var(--ink-mu)!important;font-weight:500!important;font-size:0.875rem!important;font-family:var(--fb)!important;transition:all 0.2s!important;}
.stTabs [aria-selected="true"]{background:var(--si)!important;color:#fff!important;box-shadow:0 2px 12px rgba(139,74,43,0.3)!important;}

/* BUTTONS */
.stButton>button{background:var(--si)!important;color:#fff!important;border:none!important;border-radius:var(--r-sm)!important;padding:0.6rem 2rem!important;font-weight:600!important;font-size:0.88rem!important;font-family:var(--fb)!important;letter-spacing:0.02em!important;transition:all 0.25s!important;box-shadow:0 3px 14px rgba(139,74,43,0.28)!important;}
.stButton>button:hover{background:#7A3D22!important;transform:translateY(-2px)!important;box-shadow:0 6px 22px rgba(139,74,43,0.38)!important;}

/* METRICS */
[data-testid="stMetric"]{background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-md);padding:1.1rem 1.3rem;}
[data-testid="stMetricValue"]{color:var(--ink)!important;font-family:var(--fd)!important;font-size:1.65rem!important;}
[data-testid="stMetricLabel"]{color:var(--ink-mu)!important;font-size:0.7rem!important;text-transform:uppercase!important;letter-spacing:0.1em!important;}

/* EXPANDER */
.streamlit-expanderHeader{background:var(--cr-d)!important;border-radius:var(--r-sm)!important;color:var(--si)!important;}
.streamlit-expanderContent{background:var(--cr)!important;border:1px solid var(--bdr)!important;border-radius:0 0 var(--r-sm) var(--r-sm)!important;}

/* SCROLLBAR */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--cr-d);}
::-webkit-scrollbar-thumb{background:var(--parch);border-radius:999px;}
::-webkit-scrollbar-thumb:hover{background:var(--si-l);}

/* GLASS CARD */
.gc{background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-lg);padding:1.5rem 1.8rem;box-shadow:var(--sh);transition:box-shadow 0.25s,transform 0.25s;}
.gc:hover{box-shadow:var(--sh-l);transform:translateY(-2px);}
.gc-sm{background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-md);padding:1rem 1.2rem;box-shadow:var(--sh);}

/* TAGS */
.tag{display:inline-flex;align-items:center;gap:0.3rem;background:var(--cr-d);border:1px solid var(--bdr);border-radius:999px;padding:0.22rem 0.75rem;font-size:0.72rem;color:var(--ink-mu);font-family:var(--fb);}
.tag-s{border-color:var(--bdr-w);color:var(--si);background:var(--si-p);}
.tag-g{border-color:rgba(92,122,94,0.3);color:var(--sage);background:var(--sage-p);}
.tag-a{border-color:rgba(184,115,51,0.3);color:var(--amb);background:var(--amb-p);}

/* SECTION HEADER */
.sh{display:flex;align-items:center;gap:0.7rem;font-family:var(--fd);font-size:1.12rem;font-weight:600;color:var(--si);border-bottom:1px solid var(--bdr);padding-bottom:0.55rem;margin:2rem 0 1rem 0;}
.sn{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:var(--si);color:#fff;font-family:var(--fb);font-weight:700;font-size:0.78rem;flex-shrink:0;}

/* INSIGHT BADGE */
.ib{display:flex;align-items:flex-start;gap:0.8rem;background:var(--glass);border:1px solid var(--bdr);border-radius:var(--r-md);padding:1rem 1.2rem;margin-bottom:0.65rem;transition:border-color 0.2s,box-shadow 0.2s;}
.ib:hover{border-color:var(--si-l);box-shadow:var(--sh);}
.ib-icon{font-size:1.2rem;flex-shrink:0;margin-top:0.1rem;}
.ib-title{font-weight:600;color:var(--ink);margin-bottom:0.2rem;font-size:0.875rem;}
.ib-text{font-size:0.83rem;color:var(--ink-m);line-height:1.55;}

/* SECTION RULE */
.sr{height:1px;background:linear-gradient(90deg,transparent,var(--parch),transparent);margin:2rem 0;}

/* PREDICTION */
@keyframes pR{from{opacity:0;transform:scale(0.93) translateY(12px);}to{opacity:1;transform:scale(1) translateY(0);}}
.pr{animation:pR 0.5s cubic-bezier(0.22,1,0.36,1) forwards;}

/* COMPARE TABLE */
.ct th{color:var(--si);font-size:0.76rem;letter-spacing:0.07em;text-transform:uppercase;padding:0.7rem 1rem;border-bottom:1px solid var(--bdr);}
.ct td{padding:0.6rem 1rem;border-bottom:1px solid rgba(90,70,50,0.07);font-size:0.84rem;color:var(--ink-m);}
.ct tr:last-child td{border-bottom:none;}
.ct tr:hover td{background:var(--si-p);}

/* PULSE */
@keyframes pd{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.5;transform:scale(1.4);}}
.pdt{animation:pd 2.5s ease-in-out infinite;}

/* HERO FADE */
@keyframes hfu{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
.hc{animation:hfu 0.7s cubic-bezier(0.22,1,0.36,1) both;}
.hc:nth-child(2){animation-delay:0.1s;}.hc:nth-child(3){animation-delay:0.2s;}.hc:nth-child(4){animation-delay:0.3s;}

/* EMPTY STATE */
.es{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:3rem 2rem;background:var(--glass);border:1.5px dashed var(--parch);border-radius:var(--r-lg);gap:0.75rem;}
.es-icon{font-size:2.5rem;margin-bottom:0.4rem;opacity:0.6;}
.es-title{font-family:var(--fd);font-size:1.15rem;color:var(--ink-m);font-weight:600;}
.es-sub{font-size:0.83rem;color:var(--ink-mu);max-width:360px;line-height:1.6;}
.es-tip{background:var(--si-p);border:1px solid var(--bdr-w);border-radius:var(--r-sm);padding:0.5rem 0.9rem;font-size:0.78rem;color:var(--si);}

/* ── NAVBAR ── */
.navbar{
  display:flex;align-items:center;gap:0;
  background:var(--glass);
  border:1px solid var(--bdr);
  border-radius:var(--r-lg);
  padding:0.5rem 0.75rem;
  margin-bottom:1.2rem;
  box-shadow:var(--sh);
  backdrop-filter:blur(12px);
  flex-wrap:wrap;
  gap:0.4rem;
  position:relative;
}
.navbar-brand{
  font-family:var(--fd);font-size:1.05rem;font-weight:700;color:var(--si);
  display:flex;align-items:center;gap:0.45rem;
  padding-right:0.9rem;
  border-right:1px solid var(--bdr);
  margin-right:0.5rem;
  white-space:nowrap;
}
.nm-btn{
  display:inline-flex;align-items:center;gap:0.45rem;
  padding:0.38rem 0.85rem;
  border-radius:999px;
  font-size:0.78rem;font-weight:500;
  border:1px solid transparent;
  background:transparent;
  color:var(--ink-mu);
  cursor:pointer;
  transition:all 0.18s;
  font-family:var(--fb);
  white-space:nowrap;
}
.nm-btn:hover{background:var(--cr-dd);border-color:var(--bdr);color:var(--ink);}
.nm-btn.active{
  background:var(--si)!important;
  color:#fff!important;
  border-color:var(--si)!important;
  box-shadow:0 2px 10px rgba(139,74,43,0.3);
}
.nm-divider{width:1px;height:22px;background:var(--bdr);margin:0 0.25rem;flex-shrink:0;}
.nm-right{margin-left:auto;display:flex;align-items:center;gap:0.5rem;}
.nm-status{
  display:flex;align-items:center;gap:0.4rem;
  font-size:0.72rem;color:var(--ink-mu);
  padding:0.3rem 0.65rem;
  background:var(--cr-d);
  border:1px solid var(--bdr);
  border-radius:999px;
}
</style>

<!-- BACKGROUND CANVAS -->
<canvas id="bg-canvas" style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;opacity:0.35;z-index:0;"></canvas>
<script>
(function(){
  const c=document.getElementById('bg-canvas');
  if(!c)return;
  const x=c.getContext('2d');
  let W,H,orbs=[];
  function resize(){W=c.width=c.parentElement.offsetWidth+80;H=c.height=window.innerHeight;}
  const pal=['rgba(196,123,82,0.15)','rgba(212,201,176,0.2)','rgba(92,122,94,0.09)','rgba(184,115,51,0.11)','rgba(240,221,209,0.22)','rgba(139,74,43,0.07)'];
  function mk(){return{x:Math.random()*W,y:Math.random()*H,r:120+Math.random()*200,vx:(Math.random()-0.5)*0.18,vy:(Math.random()-0.5)*0.14,col:pal[Math.floor(Math.random()*pal.length)],ph:Math.random()*Math.PI*2,fr:0.0003+Math.random()*0.0004};}
  resize();window.addEventListener('resize',resize);
  for(let i=0;i<8;i++)orbs.push(mk());
  function tick(t){
    x.clearRect(0,0,W,H);
    orbs.forEach(o=>{
      const r=o.r*(1+0.06*Math.sin(t*o.fr+o.ph));
      const g=x.createRadialGradient(o.x,o.y,0,o.x,o.y,r);
      g.addColorStop(0,o.col);g.addColorStop(1,'rgba(245,240,232,0)');
      x.beginPath();x.arc(o.x,o.y,r,0,Math.PI*2);x.fillStyle=g;x.fill();
      o.x+=o.vx;o.y+=o.vy;
      if(o.x<-o.r)o.x=W+o.r;if(o.x>W+o.r)o.x=-o.r;
      if(o.y<-o.r)o.y=H+o.r;if(o.y>H+o.r)o.y=-o.r;
    });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>
""", unsafe_allow_html=True)

# ── MATPLOTLIB CREAM THEME ────────────────────
BG="F5F0E8";SURFACE="#EDE6D6";CARD="#E3D9C6"
ACC="#8B4A2B";ACC2="#5C7A5E";ACC3="#B87333";ACC4="#C47B52"
TEXT="#2C2416";MUTED="#8A7D6A";BORDER="#D4C9B0"

def dark_fig(w=10,h=4,nrows=1,ncols=1):
    fig,axes=plt.subplots(nrows,ncols,figsize=(w,h))
    fig.patch.set_facecolor("#F5F0E8")
    ax_list=[axes] if nrows*ncols==1 else axes.flatten()
    for ax in ax_list:
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():sp.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED,labelsize=9)
        ax.xaxis.label.set_color(TEXT);ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(ACC);ax.title.set_fontsize(11)
    return fig,axes

# ── MODEL REGISTRY ────────────────────────────
MODELS=[
    ("Random Forest",    "🌲","Ensemble · Best accuracy"),
    ("Gradient Boosting","⚡","Boosted · Nonlinear"),
    ("Extra Trees",      "🌳","Fast · Low variance"),
    ("Ridge Regression", "📐","Linear · L2 regularisation"),
    ("Lasso Regression", "🔗","Linear · L1 · Sparse"),
]
MODEL_NAMES=[m[0] for m in MODELS]
MODEL_ICONS={m[0]:m[1] for m in MODELS}
MODEL_DESC ={m[0]:m[2] for m in MODELS}

# ── SESSION DEFAULTS ──────────────────────────
if "active_model" not in st.session_state:
    st.session_state.active_model="Random Forest"
if "show_nav" not in st.session_state:
    st.session_state.show_nav=True

# ── NAVBAR ────────────────────────────────────
am=st.session_state.active_model

nav_html=f"""
<div class="navbar hc">
  <div class="navbar-brand">
    <span class="pdt" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--si);flex-shrink:0;"></span>
    SalaryIQ
  </div>
"""
for name,icon,desc in MODELS:
    active_cls="active" if name==am else ""
    # No onclick — Streamlit's iframe blocks fixed/sessionStorage JS.
    # Model selection is handled solely by the sidebar selectbox (source of truth).
    nav_html+=f'<div class="nm-btn {active_cls}" title="{desc} · Change via sidebar">{icon} {name}</div>'

nav_html+=f"""
  <div class="nm-right">
    <div class="nm-status">
      <span class="pdt" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--sage);"></span>
      Active: {MODEL_ICONS[am]} {am}
    </div>
  </div>
</div>
"""
st.markdown(nav_html,unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 0.4rem;display:flex;align-items:center;gap:0.55rem;">
      <div style="width:7px;height:7px;border-radius:50%;background:var(--si);" class="pdt"></div>
      <span style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:600;color:var(--ink);">SalaryIQ</span>
    </div>
    """,unsafe_allow_html=True)
    st.divider()
    st.markdown("### ⚙️ Model Config")
    model_choice=st.selectbox("Algorithm",MODEL_NAMES,
        index=MODEL_NAMES.index(st.session_state.active_model),
        help="Also selectable via the top navbar.")
    st.session_state.active_model=model_choice
    test_size=st.slider("Test split (%)",10,40,20,step=5)/100
    random_seed=st.number_input("Random seed",value=42,step=1)

    st.markdown("### 🎛️ Advanced")
    show_confidence =st.checkbox("Show confidence intervals",value=True)
    enable_comparison=st.checkbox("Enable model comparison",value=False)
    show_feat_imp   =st.checkbox("Feature importance chart",value=True)
    show_cv         =st.checkbox("5-fold cross validation",value=False)
    show_salary_sim =st.checkbox("Salary simulator tool",value=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.78rem;color:var(--ink-mu);line-height:1.75;">
      <b style="color:var(--si);">Quick guide</b><br>
      1. Upload <code>adult.csv</code><br>
      2. Select model via navbar or sidebar<br>
      3. Explore data in EDA<br>
      4. Train &amp; evaluate<br>
      5. Predict single or batch<br>
      6. Export results
    </div>
    """,unsafe_allow_html=True)
    st.divider()
    st.markdown('<span class="tag tag-s">adult.csv · UCI Census Income</span>',unsafe_allow_html=True)

# ── DATA HELPERS ──────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    try:
        data=pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}");return None
    data=data.copy()
    for col in data.select_dtypes(include='object').columns:
        data[col]=data[col].str.strip()
    data.replace('?','Others',inplace=True)
    if 'native-country' in data.columns:
        data['native-country'].replace('?','United-States',inplace=True)
    for v in ['Without-pay','Never-worked']:
        if 'workclass' in data.columns:data=data[data['workclass']!=v]
    for v in ['1st-4th','5th-6th','Preschool']:
        if 'education' in data.columns:data=data[data['education']!=v]
    def cap(s):
        Q1,Q3=s.quantile(0.25),s.quantile(0.75);IQR=Q3-Q1
        return pd.Series(np.clip(s,Q1-1.5*IQR,Q3+1.5*IQR),index=s.index)
    for col in ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']:
        if col in data.columns:data[col]=cap(data[col])
    if 'education' in data.columns:data.drop(columns=['education'],inplace=True)
    if 'income' in data.columns:
        data['income_clean']=data['income'].str.strip().str.replace('.','',regex=False)
        data['income_numeric']=data['income_clean'].apply(lambda x:25000 if x=='<=50K' else 75000)
        data.drop(columns=['income_clean'],inplace=True)
    return data

@st.cache_resource
def train_model(X_train,y_train,_prep,model_type,seed=42):
    mp={
        "Random Forest":     RandomForestRegressor(n_estimators=150,random_state=seed,n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150,random_state=seed),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=150,random_state=seed,n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=1.0,max_iter=5000),
    }
    reg=mp.get(model_type,mp["Random Forest"])
    m=Pipeline(steps=[('preprocessor',_prep),('regressor',reg)])
    m.fit(X_train,y_train);return m

def get_feat_imp(model,cat_f,num_f):
    try:
        reg=model.named_steps['regressor']
        if not hasattr(reg,'feature_importances_'):return None
        ohe=model.named_steps['preprocessor'].transformers_[0][1]
        names=ohe.get_feature_names_out(cat_f).tolist()+list(num_f)
        df=pd.DataFrame({'Feature':names,'Importance':reg.feature_importances_})
        return df.sort_values('Importance',ascending=False).head(15)
    except:return None

def salary_tier(v):
    if v<30000:return "Entry Level","#C07A5A"
    elif v<55000:return "Mid Level",ACC3
    elif v<75000:return "Senior Level",ACC2
    else:return "Executive",ACC

# ── HERO ──────────────────────────────────────
st.markdown("""
<div class="gc hc" style="margin-bottom:1.5rem;border-radius:var(--r-lg);position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,var(--si-l),transparent);opacity:0.5;"></div>
  <div class="hc" style="display:flex;align-items:baseline;gap:0.7rem;margin-bottom:0.5rem;">
    <span style="font-family:'Playfair Display',serif;font-size:2.3rem;font-weight:700;color:var(--si);letter-spacing:-0.01em;line-height:1;">SalaryIQ</span>
    <span style="font-size:1.3rem;opacity:0.65;">💡</span>
  </div>
  <p class="hc" style="color:var(--ink-m);font-size:0.98rem;max-width:560px;line-height:1.65;margin:0 0 1.1rem;font-weight:300;">
    Machine-learning salary intelligence built on census data. Upload, train in one click, and uncover pay insights instantly.
  </p>
  <div class="hc" style="display:flex;gap:0.5rem;flex-wrap:wrap;">
    <span class="tag tag-s">5 ML Models</span>
    <span class="tag tag-g">EDA + Visualisations</span>
    <span class="tag tag-a">Batch Export</span>
    <span class="tag">Salary Insights</span>
    <span class="tag">Salary Simulator</span>
    <span class="tag">Model Comparison</span>
  </div>
</div>
""",unsafe_allow_html=True)

# ── FILE UPLOAD ───────────────────────────────
st.markdown('<div class="sh"><span class="sn">1</span> Upload Dataset</div>',unsafe_allow_html=True)
uploaded_file=st.file_uploader("Upload adult.csv",type="csv",label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="es" style="margin-bottom:1.5rem;">
      <div class="es-icon">📂</div>
      <div class="es-title">No dataset loaded yet</div>
      <div class="es-sub">Drop your <code style="background:var(--cr-dd);padding:2px 6px;border-radius:4px;font-family:'DM Mono',monospace;font-size:0.82em;">adult.csv</code> file in the uploader above to begin. The app will automatically clean, preprocess, and prepare the data for training.</div>
      <div class="es-tip">UCI Census Income dataset · 14 features · ~48,000 records</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
      <div class="gc-sm" style="text-align:center;padding:1.4rem 1rem;">
        <div style="font-size:1.8rem;margin-bottom:0.5rem;">🔬</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.93rem;font-weight:600;color:var(--ink);margin-bottom:0.35rem;">Explore &amp; Visualise</div>
        <div style="font-size:0.78rem;color:var(--ink-mu);line-height:1.55;">Histograms, box plots, correlation matrices, and income breakdowns.</div>
      </div>
      <div class="gc-sm" style="text-align:center;padding:1.4rem 1rem;">
        <div style="font-size:1.8rem;margin-bottom:0.5rem;">🤖</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.93rem;font-weight:600;color:var(--ink);margin-bottom:0.35rem;">Train Any Model</div>
        <div style="font-size:0.78rem;color:var(--ink-mu);line-height:1.55;">5 ML algorithms, one-click training, full evaluation with cross-validation.</div>
      </div>
      <div class="gc-sm" style="text-align:center;padding:1.4rem 1rem;">
        <div style="font-size:1.8rem;margin-bottom:0.5rem;">📦</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.93rem;font-weight:600;color:var(--ink);margin-bottom:0.35rem;">Predict &amp; Export</div>
        <div style="font-size:0.78rem;color:var(--ink-mu);line-height:1.55;">Single-record predictions, batch CSV uploads, and salary simulator.</div>
      </div>
    </div>
    """,unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading and preprocessing data…"):
    df=load_data(uploaded_file)

if df is None:
    st.error("Could not process the file.");st.stop()

n_rows,n_cols=df.shape
high_earners=(df['income_numeric']==75000).sum() if 'income_numeric' in df.columns else 0
pct_high=high_earners/n_rows*100
avg_salary=df['income_numeric'].mean() if 'income_numeric' in df.columns else 0

c1,c2,c3,c4=st.columns(4)
c1.metric("Records",f"{n_rows:,}","after cleaning")
c2.metric("Features",f"{n_cols-2}","input columns")
c3.metric("High Earners >50K",f"{pct_high:.1f}%",f"{high_earners:,} records")
c4.metric("Algorithm",model_choice,"selected")

# ── TABS ──────────────────────────────────────
tabs=st.tabs([
    "📊 Data Explorer",
    "🤖 Train Model",
    "🔍 Single Prediction",
    "📦 Batch Prediction",
    "💡 Salary Insights",
    "🎯 Salary Simulator",
    "📈 What-If Analysis",
    "⚖️ Compare Models",
])
tab_eda,tab_train,tab_predict,tab_batch,tab_insights,tab_sim,tab_whatif,tab_compare=tabs

# ═══════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="sh"><span class="sn">2</span> Explore Your Data</div>',unsafe_allow_html=True)
    eda1,eda2,eda3,eda4,eda5=st.tabs(["Preview & Stats","Distributions","Correlation","Income Breakdown","Data Quality"])

    with eda1:
        ca,cb=st.columns([3,1])
        with ca:
            st.markdown("**Dataset Preview** (first 50 rows)")
            st.dataframe(df.drop(columns=['income_numeric'],errors='ignore').head(50),use_container_width=True)
        with cb:
            st.markdown("**Quick Stats**")
            nd=df.select_dtypes(include=np.number)
            for col in nd.columns[:6]:
                st.markdown(f"""<div class="ib" style="padding:0.55rem 0.85rem;margin-bottom:0.35rem;">
                  <div><div class="ib-title" style="font-size:0.76rem;">{col}</div>
                  <div class="ib-text" style="font-size:0.7rem;">μ {nd[col].mean():,.1f} · σ {nd[col].std():,.1f}</div></div></div>""",unsafe_allow_html=True)
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(2),use_container_width=True)

    with eda2:
        cs,cp=st.columns([1,3])
        num_cols=df.select_dtypes(include=np.number).columns.tolist()
        cat_cols=df.select_dtypes(include='object').columns.tolist()
        with cs:
            chart_type=st.radio("Chart type",["Histogram","Categorical bar","Box plot","Violin plot"])
            if chart_type in ["Histogram","Box plot","Violin plot"]:
                col_to_plot=st.selectbox("Column",num_cols)
                split_income=st.checkbox("Split by income group",value=True)
            else:
                col_to_plot=st.selectbox("Column",cat_cols);split_income=False
        with cp:
            if chart_type=="Histogram":
                fig,ax=dark_fig(9,4)
                if split_income and 'income' in df.columns:
                    for grp,col in zip(df['income'].unique(),[ACC,ACC2]):
                        vals=df.loc[df['income']==grp,col_to_plot].dropna()
                        ax.hist(vals,bins=40,alpha=0.6,color=col,label=grp,edgecolor="none")
                    ax.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
                else:
                    ax.hist(df[col_to_plot].dropna(),bins=40,color=ACC,edgecolor="none",alpha=0.85)
                ax.set_xlabel(col_to_plot);ax.set_ylabel("Count");ax.set_title(f"Distribution of {col_to_plot}")
                plt.tight_layout();st.pyplot(fig);plt.close(fig)

            elif chart_type=="Box plot":
                fig,ax=dark_fig(9,4)
                if split_income and 'income' in df.columns:
                    groups=[df.loc[df['income']==g,col_to_plot].dropna() for g in df['income'].unique()]
                    labels=df['income'].unique().tolist()
                    bp=ax.boxplot(groups,labels=labels,patch_artist=True,notch=True,medianprops=dict(color=ACC2,linewidth=2))
                    for patch,c in zip(bp['boxes'],[ACC,ACC2]):patch.set_facecolor(c);patch.set_alpha(0.45)
                    for el in ['whiskers','caps','fliers']:
                        for it in bp[el]:it.set_color(MUTED)
                else:
                    bp=ax.boxplot(df[col_to_plot].dropna(),patch_artist=True,notch=True,medianprops=dict(color=ACC2,linewidth=2))
                    bp['boxes'][0].set_facecolor(ACC);bp['boxes'][0].set_alpha(0.45)
                ax.set_ylabel(col_to_plot);ax.set_title(f"Box Plot — {col_to_plot}")
                plt.tight_layout();st.pyplot(fig);plt.close(fig)

            elif chart_type=="Violin plot":
                fig,ax=dark_fig(9,4)
                if split_income and 'income' in df.columns:
                    groups=[df.loc[df['income']==g,col_to_plot].dropna().values for g in df['income'].unique()]
                    vp=ax.violinplot(groups,showmedians=True)
                    for i,(body,col) in enumerate(zip(vp['bodies'],[ACC,ACC2])):
                        body.set_facecolor(col);body.set_alpha(0.5)
                    vp['cmedians'].set_color(MUTED)
                    ax.set_xticks([1,2]);ax.set_xticklabels(df['income'].unique().tolist())
                else:
                    vp=ax.violinplot([df[col_to_plot].dropna().values],showmedians=True)
                    vp['bodies'][0].set_facecolor(ACC);vp['bodies'][0].set_alpha(0.5)
                    vp['cmedians'].set_color(MUTED)
                ax.set_ylabel(col_to_plot);ax.set_title(f"Violin Plot — {col_to_plot}")
                plt.tight_layout();st.pyplot(fig);plt.close(fig)

            else:
                counts=df[col_to_plot].value_counts()
                fig,ax=dark_fig(9,max(3,len(counts)*0.45))
                colors=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3,0.85,len(counts)))
                bars=ax.barh(counts.index,counts.values,color=colors,edgecolor="none")
                ax.set_xlabel("Count");ax.set_title(f"Counts by {col_to_plot}");ax.invert_yaxis()
                for bar,val in zip(bars,counts.values):
                    ax.text(bar.get_width()+counts.values.max()*0.01,bar.get_y()+bar.get_height()/2,f"{val:,}",va='center',color=MUTED,fontsize=8)
                plt.tight_layout();st.pyplot(fig);plt.close(fig)

    with eda3:
        nd=df.select_dtypes(include=np.number)
        corr=nd.corr();n=len(corr)
        fig,ax=dark_fig(8,6)
        im=ax.imshow(corr.values,cmap='RdYlBu',vmin=-1,vmax=1,aspect='auto')
        ax.set_xticks(range(n));ax.set_xticklabels(corr.columns,rotation=45,ha='right',fontsize=8)
        ax.set_yticks(range(n));ax.set_yticklabels(corr.columns,fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j,i,f"{corr.values[i,j]:.2f}",ha='center',va='center',
                        color=TEXT if abs(corr.values[i,j])<0.5 else '#fff',fontsize=7)
        fig.colorbar(im,ax=ax,fraction=0.03).ax.tick_params(colors=MUTED)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout();st.pyplot(fig);plt.close(fig)

    with eda4:
        if 'income' in df.columns and 'occupation' in df.columns:
            fig,axes=dark_fig(12,9,nrows=2,ncols=2)
            ax1,ax2,ax3,ax4=axes.flatten()
            oi=df.groupby('occupation')['income_numeric'].mean().sort_values()
            ax1.barh(oi.index,oi.values,color=[ACC2 if v>=50000 else ACC for v in oi.values],edgecolor="none")
            ax1.axvline(oi.mean(),color=ACC3,linestyle='--',linewidth=1,label='Mean')
            ax1.set_xlabel("Avg Income ($)");ax1.set_title("Avg Income by Occupation")
            ax1.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=8)
            sample=df.sample(min(2000,len(df)),random_state=42)
            ax2.scatter(sample['age'],sample['hours-per-week'],alpha=0.22,s=10,
                        c=[ACC if v==75000 else ACC2 for v in sample['income_numeric']],edgecolors='none')
            ax2.set_xlabel("Age");ax2.set_ylabel("Hours / Week");ax2.set_title("Age vs Hours (colour = income tier)")
            ax2.legend(handles=[mpatches.Patch(color=ACC,label='>50K'),mpatches.Patch(color=ACC2,label='≤50K')],
                       facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=8)
            if 'educational-num' in df.columns:
                ei=df.groupby('educational-num')['income_numeric'].mean()
                ax3.bar(ei.index,ei.values,color=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3,0.85,len(ei))),edgecolor="none")
                ax3.set_xlabel("Education Level (1–16)");ax3.set_ylabel("Avg Income ($)");ax3.set_title("Education Level vs Avg Income")
            if 'gender' in df.columns:
                gi=df.groupby('gender')['income_numeric'].mean()
                bars=ax4.bar(gi.index,gi.values,color=[ACC,ACC2],edgecolor="none",width=0.4)
                ax4.set_ylabel("Avg Income ($)");ax4.set_title("Income by Gender")
                for bar in bars:
                    ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+400,f"${bar.get_height():,.0f}",ha='center',va='bottom',color=TEXT,fontsize=9)
            plt.tight_layout(pad=2.0);st.pyplot(fig);plt.close(fig)
        else:
            st.info("Income breakdown charts require income and occupation columns.")

    with eda5:
        st.markdown("**Data Quality Report**")
        missing=df.isnull().sum()
        dtypes=df.dtypes
        nuniq=df.nunique()
        quality_df=pd.DataFrame({'Missing':missing,'Missing %':(missing/len(df)*100).round(2),'Dtype':dtypes,'Unique Values':nuniq})
        st.dataframe(quality_df,use_container_width=True)

        # Missing value heatmap
        if missing.sum()>0:
            fig,ax=dark_fig(9,3)
            ax.bar(missing.index,missing.values,color=ACC,edgecolor="none",alpha=0.8)
            ax.set_xlabel("Column");ax.set_ylabel("Missing Count");ax.set_title("Missing Values per Column")
            plt.xticks(rotation=45,ha='right');plt.tight_layout();st.pyplot(fig);plt.close(fig)
        else:
            st.success("✅ No missing values detected in this dataset.")

        # Class balance
        if 'income' in df.columns:
            st.markdown("**Class Balance**")
            vc=df['income'].value_counts()
            fig,ax=dark_fig(6,3)
            ax.pie(vc.values,labels=vc.index,colors=[ACC,ACC2],autopct='%1.1f%%',startangle=90,
                   wedgeprops=dict(edgecolor=SURFACE,linewidth=2))
            ax.set_title("Income Class Distribution")
            plt.tight_layout();st.pyplot(fig);plt.close(fig)

# ═══════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ═══════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="sh"><span class="sn">3</span> Train the ML Model</div>',unsafe_allow_html=True)
    X=df.drop(columns=[c for c in ['income','income_numeric'] if c in df.columns])
    y=df['income_numeric']
    cat_f=X.select_dtypes(include='object').columns.tolist()
    num_f=X.select_dtypes(include=np.number).columns.tolist()
    prep=ColumnTransformer(transformers=[('cat',OneHotEncoder(handle_unknown='ignore'),cat_f),('num','passthrough',num_f)])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=int(random_seed))

    c1,c2=st.columns(2)
    c1.metric("Training samples",f"{len(X_train):,}");c2.metric("Testing samples",f"{len(X_test):,}")

    st.markdown(f"""<div class="gc-sm" style="margin-bottom:1rem;">
      <div style="font-size:0.72rem;color:var(--ink-mu);margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.09em;">Selected model</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:var(--si);">{MODEL_ICONS[model_choice]} {model_choice}</div>
      <div style="font-size:0.78rem;color:var(--ink-mu);margin-top:0.25rem;">{len(cat_f)} categorical · {len(num_f)} numerical · {test_size*100:.0f}% test · seed {int(random_seed)}</div>
    </div>""",unsafe_allow_html=True)

    if st.button("🚀 Train Model Now",use_container_width=True):
        with st.spinner(f"Training {model_choice}…"):
            model=train_model(X_train,y_train,prep,model_type=model_choice,seed=int(random_seed))
        st.session_state.update({'model':model,'X_columns':X.columns.tolist(),'df':df,
                                  'X_test':X_test,'y_test':y_test,'cat_f':cat_f,'num_f':num_f})
        y_pred=model.predict(X_test)
        mae=mean_absolute_error(y_test,y_pred)
        rmse=np.sqrt(mean_squared_error(y_test,y_pred))
        r2=r2_score(y_test,y_pred)
        st.session_state['metrics']=dict(mae=mae,rmse=rmse,r2=r2)

        if show_cv:
            with st.spinner("Running 5-fold cross-validation…"):
                full_model=train_model(X,y,prep,model_type=model_choice,seed=int(random_seed))
                cv_scores=cross_val_score(full_model,X,y,cv=5,scoring='r2',n_jobs=-1)
                st.session_state['cv_scores']=cv_scores
        st.success(f"✅ {model_choice} trained successfully!")

    if 'metrics' in st.session_state:
        m=st.session_state['metrics']
        c1,c2,c3=st.columns(3)
        c1.metric("MAE",f"${m['mae']:,.0f}","avg absolute error")
        c2.metric("RMSE",f"${m['rmse']:,.0f}","root mean squared")
        c3.metric("R² Score",f"{m['r2']:.3f}","1.0 = perfect fit")

        r2_pct=int(m['r2']*100)
        ring_color=ACC2 if r2_pct>=80 else (ACC3 if r2_pct>=60 else ACC)
        quality_label=("Excellent fit — model explains most variance." if r2_pct>=80 else
                       "Good fit — reasonable predictive power." if r2_pct>=60 else
                       "Moderate fit — consider a different algorithm.")
        st.markdown(f"""<div class="gc-sm" style="display:flex;align-items:center;gap:1.5rem;margin:1.2rem 0;">
          <div style="width:76px;height:76px;border-radius:50%;border:2.5px solid {ring_color};display:flex;align-items:center;justify-content:center;font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{ring_color};flex-shrink:0;">
            {r2_pct}%</div>
          <div><div style="font-family:'Playfair Display',serif;font-weight:600;color:var(--ink);margin-bottom:0.2rem;">Model Fit Score</div>
          <div style="font-size:0.83rem;color:var(--ink-m);">{quality_label}</div></div>
        </div>""",unsafe_allow_html=True)

        if show_cv and 'cv_scores' in st.session_state:
            cv=st.session_state['cv_scores']
            st.markdown(f"""<div class="gc-sm" style="margin-bottom:1rem;">
              <div style="font-size:0.72rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">5-Fold Cross Validation R²</div>
              <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;color:var(--si);">{cv.mean():.3f} <span style="font-size:0.85rem;color:var(--ink-mu);font-family:'DM Sans',sans-serif;">± {cv.std():.3f}</span></div>
              <div style="font-size:0.78rem;color:var(--ink-mu);margin-top:0.3rem;">Fold scores: {', '.join([f'{s:.3f}' for s in cv])}</div>
            </div>""",unsafe_allow_html=True)

        fi_df=get_feat_imp(st.session_state['model'],cat_f,num_f)
        if fi_df is not None and show_feat_imp:
            st.markdown("**Top 15 Feature Importances**")
            fig,ax=dark_fig(10,5)
            bars=ax.barh(fi_df['Feature'],fi_df['Importance'],
                         color=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3,0.9,len(fi_df)))[::-1],edgecolor="none")
            ax.invert_yaxis();ax.set_xlabel("Importance");ax.set_title("Feature Importances (Top 15)")
            for bar,val in zip(bars,fi_df['Importance']):
                ax.text(bar.get_width()+0.001,bar.get_y()+bar.get_height()/2,f"{val:.3f}",va='center',color=MUTED,fontsize=8)
            plt.tight_layout();st.pyplot(fig);plt.close(fig)

        X_te=st.session_state['X_test'];y_te=st.session_state['y_test']
        y_pr=st.session_state['model'].predict(X_te);res=y_te-y_pr

        cl,cr=st.columns(2)
        with cl:
            fig,ax=dark_fig(6,4)
            jit=np.random.RandomState(0).uniform(-1500,1500,len(y_te))
            ax.scatter(y_te+jit,y_pr,alpha=0.2,s=10,color=ACC,edgecolors="none")
            mn,mx=min(y_te.min(),y_pr.min()),max(y_te.max(),y_pr.max())
            ax.plot([mn,mx],[mn,mx],color=ACC3,linewidth=1.5,linestyle="--",label="Perfect fit")
            ax.set_xlabel("Actual Salary ($)");ax.set_ylabel("Predicted Salary ($)");ax.set_title("Actual vs. Predicted")
            ax.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
            plt.tight_layout();st.pyplot(fig);plt.close(fig)
        with cr:
            fig,ax=dark_fig(6,4)
            ax.hist(res,bins=40,color=ACC2,edgecolor="none",alpha=0.8)
            ax.axvline(0,color=ACC,linestyle="--",linewidth=1.5,label="Zero error")
            ax.set_xlabel("Residual ($)");ax.set_ylabel("Count");ax.set_title("Residual Distribution")
            ax.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
            plt.tight_layout();st.pyplot(fig);plt.close(fig)

        sum_df=pd.DataFrame({'Metric':['MAE','RMSE','R²','Model','Test Size','Train Samples','Test Samples'],
                             'Value':[f"${m['mae']:,.0f}",f"${m['rmse']:,.0f}",f"{m['r2']:.4f}",
                                      model_choice,f"{test_size*100:.0f}%",f"{len(X_train):,}",f"{len(X_test):,}"]})
        st.download_button("⬇️ Download Model Summary",data=sum_df.to_csv(index=False).encode(),file_name="model_summary.csv",mime="text/csv")
    else:
        st.markdown("""<div class="es"><div class="es-icon">🤖</div><div class="es-title">No model trained yet</div>
        <div class="es-sub">Click <b>Train Model Now</b> above. Training takes 10–30 s depending on algorithm.</div>
        <div class="es-tip">Tip: Random Forest typically gives the best accuracy on this dataset</div></div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 3 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="sh"><span class="sn">4</span> Single Employee Prediction</div>',unsafe_allow_html=True)
    if 'model' not in st.session_state:
        st.markdown("""<div class="es"><div class="es-icon">🔍</div><div class="es-title">Train a model first</div>
        <div class="es-sub">Head to <b>Train Model</b>, click <b>Train Model Now</b>, then return here.</div>
        <div class="es-tip">Predictions include confidence interval and percentile ranking</div></div>""",unsafe_allow_html=True)
    else:
        model=st.session_state['model'];X_cols=st.session_state['X_columns'];df_r=st.session_state.get('df',df)
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown('<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-mu);margin-bottom:0.75rem;">👤 Personal Info</div>',unsafe_allow_html=True)
            age=st.slider("Age",int(df_r['age'].min()),int(df_r['age'].max()),30)
            gender=st.radio("Gender",df_r['gender'].unique().tolist())
            race=st.selectbox("Race",df_r['race'].unique().tolist())
            native_country=st.selectbox("Native Country",sorted(df_r['native-country'].unique().tolist()))
        with c2:
            st.markdown('<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-mu);margin-bottom:0.75rem;">🎓 Education & Work</div>',unsafe_allow_html=True)
            edu_num=st.slider("Education Level (1–16)",int(df_r['educational-num'].min()),int(df_r['educational-num'].max()),10,help="1=low, 16=Doctorate")
            workclass=st.selectbox("Work Class",df_r['workclass'].unique().tolist())
            occupation=st.selectbox("Occupation",df_r['occupation'].unique().tolist())
            hrs=st.slider("Hours / Week",int(df_r['hours-per-week'].min()),int(df_r['hours-per-week'].max()),40)
        with c3:
            st.markdown('<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ink-mu);margin-bottom:0.75rem;">🏠 Household & Capital</div>',unsafe_allow_html=True)
            marital=st.selectbox("Marital Status",df_r['marital-status'].unique().tolist())
            relationship=st.selectbox("Relationship",df_r['relationship'].unique().tolist())
            cap_gain=st.number_input("Capital Gain ($)",min_value=0,max_value=int(df_r['capital-gain'].max()),value=0)
            cap_loss=st.number_input("Capital Loss ($)",min_value=0,max_value=int(df_r['capital-loss'].max()),value=0)
            fnlwgt=st.number_input("Final Weight",min_value=int(df_r['fnlwgt'].min()),max_value=int(df_r['fnlwgt'].max()),value=200000)

        if st.button("💡 Predict Salary",use_container_width=True):
            nd=pd.DataFrame([{'age':age,'workclass':workclass,'fnlwgt':fnlwgt,'educational-num':edu_num,
                               'marital-status':marital,'occupation':occupation,'relationship':relationship,
                               'race':race,'gender':gender,'capital-gain':cap_gain,'capital-loss':cap_loss,
                               'hours-per-week':hrs,'native-country':native_country}])[X_cols]
            try:
                pred=model.predict(nd)[0]
                low,high=pred*0.85,pred*1.15
                nat_avg=df_r['income_numeric'].mean()
                pct=( df_r['income_numeric']<pred).mean()*100
                tier_label,tier_color=salary_tier(pred)

                st.markdown(f"""<div class="pr gc" style="text-align:center;margin-top:1.5rem;border-color:rgba(139,74,43,0.3);padding:2rem;">
                  <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.14em;color:var(--ink-mu);margin-bottom:0.5rem;">Estimated Annual Salary</div>
                  <div style="font-family:'Playfair Display',serif;font-size:3rem;font-weight:700;color:var(--si);line-height:1.1;">${pred:,.0f}</div>
                  <div style="font-size:0.84rem;color:var(--ink-m);margin-top:0.45rem;">{"Confidence range: " if show_confidence else ""}${low:,.0f} – ${high:,.0f}</div>
                  <div style="margin-top:0.8rem;"><span style="background:var(--si-p);border:1px solid var(--bdr-w);border-radius:999px;padding:4px 14px;font-size:0.76rem;color:var(--si);">{tier_label}</span></div>
                </div>""",unsafe_allow_html=True)

                ca,cb,cc=st.columns(3)
                ca.metric("Dataset Average",f"${nat_avg:,.0f}")
                cb.metric("Percentile",f"{pct:.0f}th")
                cc.metric("vs. Average",f"${pred-nat_avg:+,.0f}")

                fig,ax=dark_fig(7,0.9)
                ax.barh([0],[100],color=CARD,height=0.5,edgecolor="none")
                ax.barh([0],[pct],color=(ACC2 if pct>=70 else (ACC3 if pct>=40 else ACC)),height=0.5,edgecolor="none")
                ax.set_xlim(0,100);ax.set_yticks([]);ax.set_xlabel("Salary Percentile in Dataset")
                ax.set_title(f"This profile is in the {pct:.0f}th percentile",fontsize=10)
                for sp in ax.spines.values():sp.set_visible(False)
                plt.tight_layout();st.pyplot(fig);plt.close(fig)

                if 'pred_history' not in st.session_state:st.session_state['pred_history']=[]
                st.session_state['pred_history'].append({'Age':age,'Occupation':occupation,'Hrs/Wk':hrs,'Edu Level':edu_num,'Predicted ($)':int(pred),'Percentile':f"{pct:.0f}th",'Tier':tier_label})
            except Exception as e:st.error(f"Prediction error: {e}")

        if 'pred_history' in st.session_state and st.session_state['pred_history']:
            with st.expander(f"📋 Prediction History ({len(st.session_state['pred_history'])} runs)"):
                hdf=pd.DataFrame(st.session_state['pred_history'])
                st.dataframe(hdf,use_container_width=True)
                # mini bar chart of prediction history
                if len(hdf)>1:
                    fig,ax=dark_fig(8,3)
                    ax.bar(range(len(hdf)),hdf['Predicted ($)'],color=ACC,edgecolor="none",alpha=0.8)
                    ax.axhline(avg_salary,color=ACC2,linestyle='--',linewidth=1.2,label='Dataset avg')
                    ax.set_xticks(range(len(hdf)));ax.set_xticklabels([f"#{i+1}" for i in range(len(hdf))],fontsize=8)
                    ax.set_ylabel("Predicted Salary ($)");ax.set_title("Prediction History")
                    ax.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
                    plt.tight_layout();st.pyplot(fig);plt.close(fig)
                st.download_button("⬇️ Export History",data=hdf.to_csv(index=False).encode(),file_name="prediction_history.csv",mime="text/csv")
                if st.button("🗑️ Clear History"):st.session_state['pred_history']=[];st.rerun()

# ═══════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ═══════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sh"><span class="sn">5</span> Batch Prediction</div>',unsafe_allow_html=True)
    if 'model' not in st.session_state:
        st.markdown("""<div class="es"><div class="es-icon">📦</div><div class="es-title">No model available</div>
        <div class="es-sub">Train a model in <b>Train Model</b> first, then upload a batch CSV here.</div>
        <div class="es-tip">Batch results include salary tier labels and percentile rankings</div></div>""",unsafe_allow_html=True)
    else:
        model=st.session_state['model'];X_cols=st.session_state['X_columns']
        with st.expander("📄 View expected CSV format"):
            sc="age,workclass,fnlwgt,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country\n35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States\n28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States"
            st.code(sc,language="csv")
            st.download_button("⬇️ Download Sample CSV",data=sc,file_name="sample_batch.csv",mime="text/csv")
        bf=st.file_uploader("Upload batch CSV",type="csv",key="batch_uploader")
        if bf:
            with st.spinner("Running batch predictions…"):
                try:
                    bd=pd.read_csv(bf)
                    for col in bd.select_dtypes(include='object').columns:bd[col]=bd[col].str.strip()
                    preds=model.predict(bd[X_cols])
                    bd['Predicted_Salary']=preds.round(0).astype(int)
                    bd['Salary_Tier']=[salary_tier(p)[0] for p in preds]
                    bd['Percentile']=[f"{(df['income_numeric']<p).mean()*100:.0f}th" for p in preds]
                    bd['vs_Avg']=[f"${p-avg_salary:+,.0f}" for p in preds]
                    st.success(f"✅ Predicted salaries for {len(bd):,} records.")
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("Average",f"${preds.mean():,.0f}");c2.metric("Median",f"${np.median(preds):,.0f}")
                    c3.metric("Min",f"${preds.min():,.0f}");c4.metric("Max",f"${preds.max():,.0f}")
                    cl,cr=st.columns([2,1])
                    with cl:
                        fig,ax=dark_fig(7,3.5)
                        ax.hist(preds,bins=30,color=ACC,edgecolor="none",alpha=0.85)
                        ax.axvline(preds.mean(),color=ACC2,linewidth=1.5,linestyle="--",label=f"Mean ${preds.mean():,.0f}")
                        ax.axvline(np.median(preds),color=ACC3,linewidth=1.5,linestyle=":",label=f"Median ${np.median(preds):,.0f}")
                        ax.set_xlabel("Predicted Salary ($)");ax.set_ylabel("Count");ax.set_title("Batch Salary Distribution")
                        ax.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
                        plt.tight_layout();st.pyplot(fig);plt.close(fig)
                    with cr:
                        st.markdown("**Tier Breakdown**")
                        for tier,count in bd['Salary_Tier'].value_counts().items():
                            pct_t=count/len(bd)*100
                            st.markdown(f"""<div class="gc-sm" style="padding:0.65rem 0.9rem;margin-bottom:0.4rem;">
                              <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                                <span style="font-size:0.82rem;">{tier}</span>
                                <span style="font-size:0.76rem;color:var(--si);">{count} ({pct_t:.0f}%)</span>
                              </div>
                              <div style="height:3px;background:var(--cr-dd);border-radius:999px;">
                                <div style="height:3px;background:var(--si);border-radius:999px;width:{pct_t}%;"></div>
                              </div></div>""",unsafe_allow_html=True)
                    st.dataframe(bd,use_container_width=True)
                    st.download_button("⬇️ Download Predictions CSV",data=bd.to_csv(index=False).encode('utf-8'),file_name='predicted_salaries.csv',mime='text/csv')
                except KeyError as ke:st.error(f"Missing column: {ke}")
                except Exception as e:st.error(f"Batch error: {e}")
        else:
            st.markdown("""<div class="es" style="margin-top:1rem;"><div class="es-icon">🗂️</div><div class="es-title">Upload a batch file</div>
            <div class="es-sub">Drop a CSV above. Click <b>View expected CSV format</b> for a sample template.</div></div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 5 — SALARY INSIGHTS
# ═══════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="sh"><span class="sn">6</span> Salary Insights & Intelligence</div>',unsafe_allow_html=True)
    if 'income_numeric' not in df.columns:
        st.info("Load a dataset with income data to see insights.")
    else:
        high_df=df[df['income_numeric']==75000];low_df=df[df['income_numeric']==25000]
        top_occ=high_df['occupation'].mode()[0] if 'occupation' in df.columns else "N/A"
        top_wc=high_df['workclass'].mode()[0] if 'workclass' in df.columns else "N/A"
        avg_edu_h=high_df['educational-num'].mean() if 'educational-num' in df.columns else 0
        avg_edu_l=low_df['educational-num'].mean() if 'educational-num' in df.columns else 0
        avg_hrs_h=high_df['hours-per-week'].mean() if 'hours-per-week' in df.columns else 0
        avg_hrs_l=low_df['hours-per-week'].mean() if 'hours-per-week' in df.columns else 0

        for icon,title,body in [
            ("💼","Top High-Earning Occupation",f"<b style='color:var(--ink)'>{top_occ}</b> is the most common occupation among high earners."),
            ("🏢","Dominant Work Class",f"High earners most frequently work in <b style='color:var(--ink)'>{top_wc}</b>."),
            ("🎓","Education Premium",f"High earners average edu level <b style='color:var(--si)'>{avg_edu_h:.1f}/16</b> vs <b style='color:var(--ink-mu)'>{avg_edu_l:.1f}/16</b> — a <b style='color:var(--sage)'>+{avg_edu_h-avg_edu_l:.1f} level gap</b>."),
            ("⏱️","Hours Worked",f"High earners average <b style='color:var(--si)'>{avg_hrs_h:.0f} hrs/week</b> vs <b style='color:var(--ink-mu)'>{avg_hrs_l:.0f} hrs/week</b> for low earners."),
            ("👥","Dataset Composition",f"<b style='color:var(--sage)'>{pct_high:.1f}%</b> high earners ({high_earners:,} records) and <b style='color:var(--ink-mu)'>{100-pct_high:.1f}%</b> low earners."),
        ]:
            st.markdown(f"""<div class="ib"><div class="ib-icon">{icon}</div>
              <div><div class="ib-title">{title}</div><div class="ib-text">{body}</div></div></div>""",unsafe_allow_html=True)

        st.markdown('<div class="sr"></div>',unsafe_allow_html=True)
        st.markdown("#### 📊 Salary Ranking by Category")
        rank_col=st.selectbox("Rank by",[c for c in ['occupation','workclass','marital-status','race','gender'] if c in df.columns])
        grp=df.groupby(rank_col)['income_numeric'].agg(['mean','std','count']).sort_values('mean')
        fig,ax=dark_fig(10,max(4,df[rank_col].nunique()*0.4))
        bars=ax.barh(grp.index,grp['mean'],color=plt.cm.get_cmap('YlOrBr')(np.linspace(0.3,0.88,len(grp))),edgecolor="none")
        ax.axvline(df['income_numeric'].mean(),color=ACC2,linestyle='--',linewidth=1.2,label="Dataset mean")
        ax.set_xlabel("Average Salary ($)");ax.set_title(f"Average Salary by {rank_col.title()}")
        ax.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        for bar,(idx,row) in zip(bars,grp.iterrows()):
            ax.text(bar.get_width()+200,bar.get_y()+bar.get_height()/2,f"${row['mean']:,.0f} (n={int(row['count'])})",va='center',color=MUTED,fontsize=7.5)
        plt.tight_layout();st.pyplot(fig);plt.close(fig)

        # ── NEW: top-earner profile ──
        st.markdown('<div class="sr"></div>',unsafe_allow_html=True)
        st.markdown("#### 🏆 High Earner Profile vs Low Earner Profile")
        profile_cols=[c for c in ['age','educational-num','hours-per-week','capital-gain'] if c in df.columns]
        comp_data=[]
        for col in profile_cols:
            comp_data.append({'Feature':col,'High Earners (>50K)':f"{high_df[col].mean():.1f}",'Low Earners (≤50K)':f"{low_df[col].mean():.1f}"})
        st.dataframe(pd.DataFrame(comp_data),use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════
# TAB 6 — SALARY SIMULATOR (NEW)
# ═══════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="sh"><span class="sn">7</span> Salary Simulator</div>',unsafe_allow_html=True)
    if not show_salary_sim:
        st.info("Enable **Salary Simulator** in the sidebar to use this tool.")
    elif 'model' not in st.session_state:
        st.markdown("""<div class="es"><div class="es-icon">🎯</div><div class="es-title">Train a model first</div>
        <div class="es-sub">The salary simulator needs a trained model. Go to <b>Train Model</b> tab first.</div></div>""",unsafe_allow_html=True)
    else:
        st.markdown("""<div class="gc-sm" style="margin-bottom:1.2rem;">
          <div class="ib-title" style="margin-bottom:0.3rem;">Interactive Career Planner</div>
          <div class="ib-text">Adjust sliders to simulate how changes in education, hours worked, or age affect predicted salary. Lock a base profile and compare scenarios.</div>
        </div>""",unsafe_allow_html=True)

        model=st.session_state['model'];X_cols=st.session_state['X_columns'];df_r=st.session_state.get('df',df)

        col_l,col_r=st.columns([1,1])
        with col_l:
            st.markdown("**Base Profile**")
            s_occ=st.selectbox("Occupation",df_r['occupation'].unique().tolist(),key="sim_occ")
            s_wc=st.selectbox("Work Class",df_r['workclass'].unique().tolist(),key="sim_wc")
            s_mar=st.selectbox("Marital Status",df_r['marital-status'].unique().tolist(),key="sim_mar")
            s_rel=st.selectbox("Relationship",df_r['relationship'].unique().tolist(),key="sim_rel")
            s_gen=st.radio("Gender",df_r['gender'].unique().tolist(),key="sim_gen")
            s_race=st.selectbox("Race",df_r['race'].unique().tolist(),key="sim_race")
            s_country=st.selectbox("Country",sorted(df_r['native-country'].unique().tolist()),key="sim_country")

        with col_r:
            st.markdown("**Simulate Changes**")
            s_age=st.slider("Age",int(df_r['age'].min()),int(df_r['age'].max()),35,key="sim_age")
            s_edu=st.slider("Education Level (1–16)",1,16,10,key="sim_edu",help="Drag to see salary impact")
            s_hrs=st.slider("Hours per Week",10,80,40,key="sim_hrs")
            s_cap=st.slider("Capital Gain ($)",0,int(df_r['capital-gain'].max()),0,key="sim_cap")

            def sim_predict(age,edu,hrs,cap):
                nd=pd.DataFrame([{'age':age,'workclass':s_wc,'fnlwgt':200000,'educational-num':edu,
                                   'marital-status':s_mar,'occupation':s_occ,'relationship':s_rel,
                                   'race':s_race,'gender':s_gen,'capital-gain':cap,'capital-loss':0,
                                   'hours-per-week':hrs,'native-country':s_country}])[X_cols]
                return model.predict(nd)[0]

            try:
                base_pred=sim_predict(s_age,s_edu,s_hrs,s_cap)
                tier_label,_=salary_tier(base_pred)
                pct_above=(df_r['income_numeric']<base_pred).mean()*100

                st.markdown(f"""<div class="gc-sm" style="text-align:center;margin-top:1rem;border-color:rgba(139,74,43,0.25);">
                  <div style="font-size:0.7rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">Predicted Salary</div>
                  <div style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:700;color:var(--si);">${base_pred:,.0f}</div>
                  <div style="font-size:0.78rem;color:var(--ink-mu);margin-top:0.3rem;">{tier_label} · {pct_above:.0f}th percentile</div>
                </div>""",unsafe_allow_html=True)

                # Sensitivity analysis — vary each factor
                st.markdown("**Sensitivity: What changes the most?**")
                edu_range=range(1,17)
                hrs_range=range(20,81,5)
                age_range=range(20,65,5)

                edu_preds=[sim_predict(s_age,e,s_hrs,s_cap) for e in edu_range]
                hrs_preds=[sim_predict(s_age,s_edu,h,s_cap) for h in hrs_range]
                age_preds=[sim_predict(a,s_edu,s_hrs,s_cap) for a in age_range]

                fig,axes=dark_fig(12,3.5,nrows=1,ncols=3)
                ax1,ax2,ax3=axes.flatten()
                ax1.plot(list(edu_range),edu_preds,color=ACC,linewidth=2,marker='o',markersize=4)
                ax1.axvline(s_edu,color=ACC3,linestyle='--',linewidth=1);ax1.set_xlabel("Education Level");ax1.set_ylabel("Salary ($)");ax1.set_title("Education Impact")
                ax2.plot(list(hrs_range),hrs_preds,color=ACC2,linewidth=2,marker='o',markersize=4)
                ax2.axvline(s_hrs,color=ACC3,linestyle='--',linewidth=1);ax2.set_xlabel("Hours / Week");ax2.set_ylabel("Salary ($)");ax2.set_title("Hours Impact")
                ax3.plot(list(age_range),age_preds,color=ACC4,linewidth=2,marker='o',markersize=4)
                ax3.axvline(s_age,color=ACC3,linestyle='--',linewidth=1);ax3.set_xlabel("Age");ax3.set_ylabel("Salary ($)");ax3.set_title("Age Impact")
                plt.tight_layout();st.pyplot(fig);plt.close(fig)

            except Exception as e:st.error(f"Simulation error: {e}")


# ═══════════════════════════════════════════════
# TAB 7 — WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════
with tab_whatif:
    st.markdown('<div class="sh"><span class="sn">8</span> What-If Analysis</div>',unsafe_allow_html=True)
    if 'model' not in st.session_state:
        st.markdown("""<div class="es"><div class="es-icon">📈</div><div class="es-title">Train a model first</div>
        <div class="es-sub">What-If analysis needs a trained model. Go to <b>Train Model</b> tab first.</div></div>""",unsafe_allow_html=True)
    else:
        st.markdown("""<div class="gc-sm" style="margin-bottom:1.2rem;">
          <div class="ib-title" style="margin-bottom:0.3rem;">Side-by-Side Scenario Comparison</div>
          <div class="ib-text">Define two employee profiles and compare their predicted salaries instantly.</div>
        </div>""",unsafe_allow_html=True)

        model=st.session_state['model'];X_cols=st.session_state['X_columns'];df_r=st.session_state.get('df',df)

        col_a,col_b=st.columns(2)
        profiles={}
        for col,label,key_prefix in [(col_a,"👤 Person A","pa"),(col_b,"👤 Person B","pb")]:
            with col:
                st.markdown(f"**{label}**")
                p={
                    'age':st.slider("Age",int(df_r['age'].min()),int(df_r['age'].max()),35 if key_prefix=="pa" else 45,key=f"{key_prefix}_age"),
                    'workclass':st.selectbox("Work Class",df_r['workclass'].unique().tolist(),key=f"{key_prefix}_wc"),
                    'fnlwgt':200000,
                    'educational-num':st.slider("Education Level",1,16,10 if key_prefix=="pa" else 14,key=f"{key_prefix}_edu"),
                    'marital-status':st.selectbox("Marital Status",df_r['marital-status'].unique().tolist(),key=f"{key_prefix}_mar"),
                    'occupation':st.selectbox("Occupation",df_r['occupation'].unique().tolist(),key=f"{key_prefix}_occ"),
                    'relationship':st.selectbox("Relationship",df_r['relationship'].unique().tolist(),key=f"{key_prefix}_rel"),
                    'race':st.selectbox("Race",df_r['race'].unique().tolist(),key=f"{key_prefix}_race"),
                    'gender':st.radio("Gender",df_r['gender'].unique().tolist(),key=f"{key_prefix}_gen"),
                    'capital-gain':st.slider("Capital Gain ($)",0,int(df_r['capital-gain'].max()),0,key=f"{key_prefix}_cap"),
                    'capital-loss':0,
                    'hours-per-week':st.slider("Hours/Week",10,80,40,key=f"{key_prefix}_hrs"),
                    'native-country':st.selectbox("Country",sorted(df_r['native-country'].unique().tolist()),key=f"{key_prefix}_ctry"),
                }
                profiles[label]=p

        if st.button("⚖️ Compare Profiles",use_container_width=True):
            try:
                results={}
                for label,p in profiles.items():
                    nd=pd.DataFrame([p])[X_cols]
                    pred=model.predict(nd)[0]
                    tier_label,_=salary_tier(pred)
                    pct=(df_r['income_numeric']<pred).mean()*100
                    results[label]={'pred':pred,'tier':tier_label,'pct':pct}

                ca,cb=st.columns(2)
                cols_map={list(profiles.keys())[0]:ca,list(profiles.keys())[1]:cb}
                for label,res in results.items():
                    with cols_map[label]:
                        st.markdown(f"""<div class="gc-sm" style="text-align:center;border-color:rgba(139,74,43,0.25);margin-top:1rem;">
                          <div style="font-size:0.7rem;color:var(--ink-mu);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;">{label}</div>
                          <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:var(--si);">${res['pred']:,.0f}</div>
                          <div style="font-size:0.78rem;color:var(--ink-mu);margin-top:0.3rem;">{res['tier']} · {res['pct']:.0f}th percentile</div>
                        </div>""",unsafe_allow_html=True)

                preds=[r['pred'] for r in results.values()]
                diff=abs(preds[0]-preds[1])
                higher=list(results.keys())[0] if preds[0]>=preds[1] else list(results.keys())[1]
                st.markdown(f"""<div class="ib" style="margin-top:1.2rem;">
                  <div class="ib-icon">💡</div>
                  <div><div class="ib-title">Comparison Result</div>
                  <div class="ib-text"><b style="color:var(--si)">{higher}</b> earns <b style="color:var(--si)">${diff:,.0f} more</b> per year based on their profile differences.</div></div>
                </div>""",unsafe_allow_html=True)

                fig,ax=dark_fig(6,3)
                labels_list=list(results.keys())
                pred_list=[results[l]['pred'] for l in labels_list]
                colors=[ACC,ACC2]
                bars=ax.bar(labels_list,pred_list,color=colors,edgecolor="none",width=0.4)
                ax.axhline(df_r['income_numeric'].mean(),color=ACC3,linestyle='--',linewidth=1.2,label='Dataset avg')
                ax.set_ylabel("Predicted Salary ($)");ax.set_title("Profile Salary Comparison")
                for bar,val in zip(bars,pred_list):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+300,f"${val:,.0f}",ha='center',va='bottom',color=TEXT,fontsize=10)
                ax.legend(facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
                plt.tight_layout();st.pyplot(fig);plt.close(fig)
            except Exception as e:st.error(f"Comparison error: {e}")

# ═══════════════════════════════════════════════
# TAB 8 — COMPARE MODELS
# ═══════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="sh"><span class="sn">9</span> Compare Models</div>',unsafe_allow_html=True)
    if not enable_comparison:
        st.info("Enable **Model comparison** in the sidebar to use this tab.")
    elif 'model' not in st.session_state:
        st.markdown("""<div class="es"><div class="es-icon">⚖️</div><div class="es-title">Train a model first</div>
        <div class="es-sub">Train at least one model, then enable model comparison in the sidebar.</div></div>""",unsafe_allow_html=True)
    else:
        st.markdown("""<div class="gc-sm" style="margin-bottom:1.2rem;">
          <div class="ib-title" style="margin-bottom:0.3rem;">Multi-Algorithm Benchmark</div>
          <div class="ib-text">Train all 5 models on identical data splits and compare MAE, RMSE, and R² side-by-side.</div>
        </div>""",unsafe_allow_html=True)

        if st.button("🏆 Run Full Model Comparison",use_container_width=True):
            X_c=df.drop(columns=[c for c in ['income','income_numeric'] if c in df.columns])
            y_c=df['income_numeric']
            cat_fc=X_c.select_dtypes(include='object').columns.tolist()
            num_fc=X_c.select_dtypes(include=np.number).columns.tolist()
            prep_c=ColumnTransformer(transformers=[('cat',OneHotEncoder(handle_unknown='ignore'),cat_fc),('num','passthrough',num_fc)])
            Xtr,Xte,ytr,yte=train_test_split(X_c,y_c,test_size=test_size,random_state=int(random_seed))

            comp_results=[]
            prog=st.progress(0,text="Training models…")
            for i,(name,icon,desc) in enumerate(MODELS):
                with st.spinner(f"Training {icon} {name}…"):
                    m=train_model(Xtr,ytr,prep_c,model_type=name,seed=int(random_seed))
                    yp=m.predict(Xte)
                    comp_results.append({
                        'Model':f"{icon} {name}",
                        'MAE ($)':int(mean_absolute_error(yte,yp)),
                        'RMSE ($)':int(np.sqrt(mean_squared_error(yte,yp))),
                        'R²':round(r2_score(yte,yp),4),
                    })
                prog.progress((i+1)/len(MODELS),text=f"Done: {name}")
            prog.empty()

            comp_df=pd.DataFrame(comp_results).sort_values('R²',ascending=False)
            st.dataframe(comp_df,use_container_width=True,hide_index=True)

            fig,axes=dark_fig(12,4,nrows=1,ncols=3)
            ax1,ax2,ax3=axes.flatten()
            colors=plt.cm.get_cmap('YlOrBr')(np.linspace(0.35,0.85,len(comp_df)))

            for ax,metric,title in [(ax1,'MAE ($)','MAE — lower is better'),(ax2,'RMSE ($)','RMSE — lower is better'),(ax3,'R²','R² — higher is better')]:
                vals=comp_df[metric].values
                bars=ax.barh(comp_df['Model'],vals,color=colors,edgecolor="none")
                ax.invert_yaxis();ax.set_title(title)
                for bar,val in zip(bars,vals):
                    ax.text(bar.get_width()*1.01,bar.get_y()+bar.get_height()/2,
                            f"${val:,.0f}" if metric!='R²' else f"{val:.3f}",
                            va='center',color=MUTED,fontsize=8)
            plt.tight_layout();st.pyplot(fig);plt.close(fig)

            best=comp_df.iloc[0]
            st.markdown(f"""<div class="ib" style="margin-top:0.5rem;">
              <div class="ib-icon">🏆</div>
              <div><div class="ib-title">Best Model: {best['Model']}</div>
              <div class="ib-text">R² = {best['R²']:.4f} · MAE = ${best['MAE ($)']:,} · RMSE = ${best['RMSE ($)']:,}</div></div>
            </div>""",unsafe_allow_html=True)

            st.download_button("⬇️ Download Comparison CSV",data=comp_df.to_csv(index=False).encode(),file_name="model_comparison.csv",mime="text/csv")
