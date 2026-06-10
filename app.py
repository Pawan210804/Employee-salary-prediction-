import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="SalaryIQ — ML Salary Predictor",
    page_icon="💡",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e2e8f0; }

[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #1e2535; }

.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    border: 1px solid #4338ca; border-radius: 16px;
    padding: 2.5rem 2rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(ellipse at top right, rgba(124,58,237,0.25) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700; color: #ffffff; margin: 0 0 0.4rem 0;
}
.hero-sub { font-size: 1rem; color: #a5b4fc; max-width: 540px; line-height: 1.6; }

.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem; font-weight: 700; color: #a78bfa;
    border-bottom: 1px solid #1e2535; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0;
}
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: #7c3aed; color: #fff; font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1; min-width: 160px; background: #161b27;
    border: 1px solid #1e2535; border-radius: 12px; padding: 1.2rem 1.4rem;
}
.metric-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 0.3rem; }
.metric-card .value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
.metric-card .delta { font-size: 0.78rem; color: #a78bfa; margin-top: 0.15rem; }

.pred-result {
    background: linear-gradient(135deg, #1e1b4b, #2d1b69);
    border: 1px solid #7c3aed; border-radius: 16px;
    padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.pred-result .pred-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.14em; color: #a78bfa; margin-bottom: 0.5rem; }
.pred-result .pred-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 700; color: #fff; }
.pred-result .pred-range { font-size: 0.85rem; color: #a5b4fc; margin-top: 0.4rem; }

.pill {
    display: inline-block; background: #1e2535; border: 1px solid #334155;
    border-radius: 999px; padding: 0.25rem 0.8rem; font-size: 0.75rem; color: #94a3b8;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #94a3b8 !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: #7c3aed !important; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MATPLOTLIB DARK THEME HELPER
# ──────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#161b27"
ACCENT  = "#7c3aed"
ACCENT2 = "#06b6d4"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2535")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    return fig, ax

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
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
    if model_type == "Random Forest":
        reg = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    elif model_type == "Gradient Boosting":
        reg = GradientBoostingRegressor(n_estimators=150, random_state=42)
    else:
        reg = Ridge(alpha=1.0)

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


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Ridge Regression"],
        help="Random Forest and Gradient Boosting give feature importances."
    )
    test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100
    st.markdown("---")
    st.markdown("## 📋 How to Use")
    st.markdown("""
    1. **Upload** your `adult.csv` file
    2. **Explore** data in the EDA tab
    3. **Train** the ML model
    4. **Predict** individual or batch salaries
    """)
    st.markdown("---")
    st.markdown('<span class="pill">adult.csv · UCI Census Income</span>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">💡 SalaryIQ</div>
  <div class="hero-sub">
    Machine-learning salary predictor built on census data.
    Upload your dataset, train in one click, and get instant estimates.
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="step-badge">1</span> Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload adult.csv", type="csv", label_visibility="collapsed")

if uploaded_file is None:
    st.info("👆 Upload your `adult.csv` file above to get started.")
    st.stop()

with st.spinner("Loading and preprocessing data…"):
    df = load_and_preprocess_data(uploaded_file)

if df is None:
    st.error("Could not process the file. Please check its format.")
    st.stop()

n_rows, n_cols = df.shape
high_earners = (df['income_numeric'] == 75000).sum() if 'income_numeric' in df.columns else 0
pct_high = high_earners / n_rows * 100

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="label">Records</div><div class="value">{n_rows:,}</div><div class="delta">after cleaning</div>
  </div>
  <div class="metric-card">
    <div class="label">Features</div><div class="value">{n_cols - 2}</div><div class="delta">input columns</div>
  </div>
  <div class="metric-card">
    <div class="label">High Earners (&gt;50K)</div><div class="value">{pct_high:.1f}%</div><div class="delta">{high_earners:,} records</div>
  </div>
  <div class="metric-card">
    <div class="label">Model</div><div class="value" style="font-size:1rem;padding-top:0.3rem">{model_choice}</div><div class="delta">selected algorithm</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_eda, tab_train, tab_predict, tab_batch = st.tabs([
    "📊 Data Explorer", "🤖 Train Model", "🔍 Single Prediction", "📦 Batch Prediction"
])


# ═══════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header"><span class="step-badge">2</span> Explore Your Data</div>', unsafe_allow_html=True)

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Preview & Stats", "Distributions", "Correlation"])

    with eda_tab1:
        st.dataframe(df.drop(columns=['income_numeric'], errors='ignore').head(50), use_container_width=True)
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe(), use_container_width=True)

    with eda_tab2:
        col_sel, col_plot = st.columns([1, 3])
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        with col_sel:
            chart_type = st.radio("Chart type", ["Numerical histogram", "Categorical bar"])
            if chart_type == "Numerical histogram":
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
                        ax.hist(vals, bins=40, alpha=0.65, color=col, label=grp, edgecolor="none")
                    ax.legend(facecolor=SURFACE, edgecolor="#1e2535", labelcolor=TEXT, fontsize=9)
                else:
                    ax.hist(df[col_to_plot].dropna(), bins=40, color=ACCENT, edgecolor="none", alpha=0.85)
                ax.set_xlabel(col_to_plot)
                ax.set_ylabel("Count")
                ax.set_title(f"Distribution of {col_to_plot}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                counts = df[col_to_plot].value_counts()
                fig, ax = dark_fig(9, max(3, len(counts) * 0.45))
                colors = plt.cm.get_cmap('Purples')(np.linspace(0.4, 0.9, len(counts)))
                bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none")
                ax.set_xlabel("Count")
                ax.set_title(f"Counts by {col_to_plot}")
                ax.invert_yaxis()
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_width() + counts.values.max() * 0.01, bar.get_y() + bar.get_height()/2,
                            f"{val:,}", va='center', color=MUTED, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

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
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ═══════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="section-header"><span class="step-badge">3</span> Train the ML Model</div>', unsafe_allow_html=True)

    X = df.drop(columns=[c for c in ['income', 'income_numeric'] if c in df.columns])
    y = df['income_numeric']

    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor_transformer = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    c1, c2 = st.columns(2)
    c1.metric("Training samples", f"{len(X_train):,}")
    c2.metric("Testing samples", f"{len(X_test):,}")

    if st.button("🚀 Train Model Now"):
        with st.spinner(f"Training {model_choice}… this may take up to 30 s"):
            model = train_model(X_train, y_train, preprocessor_transformer, model_type=model_choice)

        st.session_state['model'] = model
        st.session_state['X_columns'] = X.columns.tolist()
        st.session_state['df'] = df
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        st.session_state['metrics'] = dict(mae=mae, rmse=rmse, r2=r2)
        st.success("✅ Model trained successfully!")

    if 'metrics' in st.session_state:
        m = st.session_state['metrics']
        st.markdown(f"""
        <div class="metric-grid" style="margin-top:1rem">
          <div class="metric-card"><div class="label">MAE</div><div class="value">${m['mae']:,.0f}</div><div class="delta">avg absolute error</div></div>
          <div class="metric-card"><div class="label">RMSE</div><div class="value">${m['rmse']:,.0f}</div><div class="delta">root mean squared</div></div>
          <div class="metric-card"><div class="label">R² Score</div><div class="value">{m['r2']:.3f}</div><div class="delta">1.0 = perfect fit</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance
        fi_df = get_feature_importance(st.session_state['model'], X)
        if fi_df is not None:
            st.markdown("**Top 15 Feature Importances**")
            fig, ax = dark_fig(10, 5)
            cmap_colors = plt.cm.get_cmap('Purples')(np.linspace(0.35, 0.95, len(fi_df)))
            bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=cmap_colors[::-1], edgecolor="none")
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importances (Top 15)")
            for bar, val in zip(bars, fi_df['Importance']):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va='center', color=MUTED, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Actual vs predicted
        model_eval = st.session_state['model']
        X_test_s   = st.session_state['X_test']
        y_test_s   = st.session_state['y_test']
        y_pred_s   = model_eval.predict(X_test_s)

        fig, ax = dark_fig(8, 4)
        jitter = np.random.RandomState(0).uniform(-1500, 1500, len(y_test_s))
        ax.scatter(y_test_s + jitter, y_pred_s, alpha=0.25, s=12, color=ACCENT, edgecolors="none")
        mn, mx = min(y_test_s.min(), y_pred_s.min()), max(y_test_s.max(), y_pred_s.max())
        ax.plot([mn, mx], [mn, mx], color="#a78bfa", linewidth=1.5, linestyle="--", label="Perfect fit")
        ax.set_xlabel("Actual Salary ($)")
        ax.set_ylabel("Predicted Salary ($)")
        ax.set_title("Actual vs. Predicted Salary")
        ax.legend(facecolor=SURFACE, edgecolor="#1e2535", labelcolor=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Click **Train Model Now** to begin.")


# ═══════════════════════════════════════════════
# TAB 3 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-header"><span class="step-badge">4</span> Single Employee Prediction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("⚠️ Train the model first (Tab 2) before making predictions.")
    else:
        model     = st.session_state['model']
        X_columns = st.session_state['X_columns']
        df_ref    = st.session_state.get('df', df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Info**")
            age            = st.slider("Age", int(df_ref['age'].min()), int(df_ref['age'].max()), 30)
            gender         = st.radio("Gender", df_ref['gender'].unique().tolist())
            race           = st.selectbox("Race", df_ref['race'].unique().tolist())
            native_country = st.selectbox("Native Country", sorted(df_ref['native-country'].unique().tolist()))

        with col2:
            st.markdown("**Education & Work**")
            educational_num = st.slider("Education Level (1–16)",
                                        int(df_ref['educational-num'].min()),
                                        int(df_ref['educational-num'].max()), 10,
                                        help="1=low, 16=Doctorate")
            workclass       = st.selectbox("Work Class", df_ref['workclass'].unique().tolist())
            occupation      = st.selectbox("Occupation", df_ref['occupation'].unique().tolist())
            hours_per_week  = st.slider("Hours / Week",
                                        int(df_ref['hours-per-week'].min()),
                                        int(df_ref['hours-per-week'].max()), 40)

        with col3:
            st.markdown("**Household & Capital**")
            marital_status = st.selectbox("Marital Status", df_ref['marital-status'].unique().tolist())
            relationship   = st.selectbox("Relationship", df_ref['relationship'].unique().tolist())
            capital_gain   = st.number_input("Capital Gain ($)", min_value=0,
                                             max_value=int(df_ref['capital-gain'].max()), value=0)
            capital_loss   = st.number_input("Capital Loss ($)", min_value=0,
                                             max_value=int(df_ref['capital-loss'].max()), value=0)
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
                st.markdown(f"""
                <div class="pred-result">
                  <div class="pred-label">Estimated Annual Salary</div>
                  <div class="pred-value">${predicted:,.0f}</div>
                  <div class="pred-range">Likely range: ${low:,.0f} – ${high:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

                nat_avg   = df_ref['income_numeric'].mean()
                pct_above = (df_ref['income_numeric'] < predicted).mean() * 100
                st.markdown(f"""
                <div style="margin-top:1rem;display:flex;gap:1rem;flex-wrap:wrap">
                  <div class="pill">📈 Dataset avg: ${nat_avg:,.0f}</div>
                  <div class="pill">🏆 Higher than {pct_above:.0f}% of dataset</div>
                </div>
                """, unsafe_allow_html=True)

                # Mini gauge bar
                fig, ax = dark_fig(7, 0.9)
                ax.barh([0], [100], color="#1e2535", height=0.5, edgecolor="none")
                ax.barh([0], [pct_above], color=ACCENT, height=0.5, edgecolor="none")
                ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("Percentile in dataset")
                ax.set_title(f"Salary Percentile: {pct_above:.0f}th", fontsize=10)
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ═══════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ═══════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-header"><span class="step-badge">5</span> Batch Prediction</div>', unsafe_allow_html=True)

    if 'model' not in st.session_state:
        st.warning("⚠️ Train the model first (Tab 2) before running batch predictions.")
    else:
        model     = st.session_state['model']
        X_columns = st.session_state['X_columns']

        with st.expander("📄 View expected CSV format"):
            st.code(
                "age,workclass,fnlwgt,educational-num,marital-status,occupation,"
                "relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country\n"
                "35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States\n"
                "28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States",
                language="csv"
            )
            sample_csv = (
                "age,workclass,fnlwgt,educational-num,marital-status,occupation,"
                "relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country\n"
                "35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States\n"
                "28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States\n"
                "50,Self-emp-inc,180000,14,Married-civ-spouse,Prof-specialty,Wife,Asian-Pac-Islander,Female,10000,0,50,India"
            )
            st.download_button("⬇️ Download sample CSV", data=sample_csv,
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

                    st.success(f"✅ Predicted salaries for {len(batch_data):,} records.")

                    pa, pb, pc = st.columns(3)
                    pa.metric("Average Salary", f"${preds.mean():,.0f}")
                    pb.metric("Min Salary",     f"${preds.min():,.0f}")
                    pc.metric("Max Salary",     f"${preds.max():,.0f}")

                    # Distribution chart
                    fig, ax = dark_fig(9, 3.5)
                    ax.hist(preds, bins=30, color=ACCENT, edgecolor="none", alpha=0.85)
                    ax.axvline(preds.mean(), color=ACCENT2, linewidth=1.5, linestyle="--", label=f"Mean ${preds.mean():,.0f}")
                    ax.set_xlabel("Predicted Salary ($)")
                    ax.set_ylabel("Count")
                    ax.set_title("Salary Distribution — Batch Results")
                    ax.legend(facecolor=SURFACE, edgecolor="#1e2535", labelcolor=TEXT, fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(batch_data, use_container_width=True)

                    csv_out = batch_data.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions CSV", data=csv_out,
                                       file_name='predicted_salaries.csv', mime='text/csv')

                except KeyError as ke:
                    st.error(f"Missing column in your CSV: {ke}. Make sure all required columns are present.")
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")
