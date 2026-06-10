import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import io

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
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #0f1117; color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #7c3aed;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    border: 1px solid #4338ca;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at top right, rgba(124,58,237,0.25) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: #a5b4fc;
    max-width: 540px;
    line-height: 1.6;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #a78bfa;
    border-bottom: 1px solid #1e2535;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* ── Metric cards ── */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1; min-width: 160px;
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.metric-card .label {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #64748b; margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700; color: #e2e8f0;
}
.metric-card .delta { font-size: 0.78rem; color: #a78bfa; margin-top: 0.15rem; }

/* ── Prediction result ── */
.pred-result {
    background: linear-gradient(135deg, #1e1b4b, #2d1b69);
    border: 1px solid #7c3aed;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.pred-result .pred-label {
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.14em;
    color: #a78bfa; margin-bottom: 0.5rem;
}
.pred-result .pred-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 700; color: #fff;
}
.pred-result .pred-range { font-size: 0.85rem; color: #a5b4fc; margin-top: 0.4rem; }

/* ── Info pill ── */
.pill {
    display: inline-block;
    background: #1e2535; border: 1px solid #334155;
    border-radius: 999px; padding: 0.25rem 0.8rem;
    font-size: 0.75rem; color: #94a3b8;
}

/* ── Step badge ── */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: #7c3aed; color: #fff;
    font-size: 0.8rem; font-weight: 700;
    flex-shrink: 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #94a3b8 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #7c3aed !important; color: #fff !important;
}

/* ── Dataframe / tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Inputs ── */
.stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div {
    background: #161b27 !important; border-color: #1e2535 !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


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
    # Strip whitespace from string columns
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
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return pd.Series(np.clip(df_col, lo, hi), index=df_col.index)

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
    """Extract feature importances (works for tree-based models)."""
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
        importances = reg.feature_importances_
        fi_df = pd.DataFrame({'Feature': all_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False).head(15)
        return fi_df
    except Exception:
        return None


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("**Model Selection**")
    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Ridge Regression"],
        help="Random Forest and Gradient Boosting are tree-based and give feature importances."
    )

    st.markdown("**Train/Test Split**")
    test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100

    st.markdown("---")
    st.markdown("## 📋 How to Use")
    st.markdown("""
    1. **Upload** your `adult.csv` file  
    2. **Explore** the data in the EDA tab  
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

# ── Quick dataset metrics ──
n_rows, n_cols = df.shape
high_earners = (df['income_numeric'] == 75000).sum() if 'income_numeric' in df.columns else 0
pct_high = high_earners / n_rows * 100

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="label">Records</div>
    <div class="value">{n_rows:,}</div>
    <div class="delta">after cleaning</div>
  </div>
  <div class="metric-card">
    <div class="label">Features</div>
    <div class="value">{n_cols - 2}</div>
    <div class="delta">input columns</div>
  </div>
  <div class="metric-card">
    <div class="label">High Earners (>50K)</div>
    <div class="value">{pct_high:.1f}%</div>
    <div class="delta">{high_earners:,} records</div>
  </div>
  <div class="metric-card">
    <div class="label">Model</div>
    <div class="value" style="font-size:1rem; padding-top:0.3rem">{model_choice}</div>
    <div class="delta">selected algorithm</div>
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
        with col_sel:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            chart_type = st.radio("Chart type", ["Numerical histogram", "Categorical bar"])
            if chart_type == "Numerical histogram":
                col_to_plot = st.selectbox("Column", num_cols)
                color_by = st.checkbox("Color by income group", value=True)
            else:
                col_to_plot = st.selectbox("Column", cat_cols)
                color_by = True

        with col_plot:
            if chart_type == "Numerical histogram":
                color_col = "income" if (color_by and "income" in df.columns) else None
                fig = px.histogram(
                    df, x=col_to_plot, color=color_col,
                    barmode="overlay", nbins=40,
                    template="plotly_dark",
                    color_discrete_sequence=["#7c3aed", "#06b6d4"]
                )
                fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                counts = df[col_to_plot].value_counts().reset_index()
                counts.columns = [col_to_plot, 'count']
                fig = px.bar(
                    counts, x=col_to_plot, y='count',
                    template="plotly_dark",
                    color='count', color_continuous_scale='Viridis'
                )
                fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                                  xaxis_tickangle=-30, margin=dict(t=20, b=60))
                st.plotly_chart(fig, use_container_width=True)

    with eda_tab3:
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            showscale=True
        ))
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e2e8f0"), margin=dict(t=20),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


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

    if st.button("🚀 Train Model Now", use_container_width=False):
        with st.spinner(f"Training {model_choice}… this may take up to 30 s"):
            model = train_model(X_train, y_train, preprocessor_transformer, model_type=model_choice)

        st.session_state['model'] = model
        st.session_state['X_columns'] = X.columns.tolist()
        st.session_state['df'] = df

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.session_state['metrics'] = dict(mae=mae, rmse=rmse, r2=r2)
        st.success("✅ Model trained successfully!")

    if 'metrics' in st.session_state:
        m = st.session_state['metrics']
        st.markdown(f"""
        <div class="metric-grid" style="margin-top:1rem">
          <div class="metric-card">
            <div class="label">MAE</div>
            <div class="value">${m['mae']:,.0f}</div>
            <div class="delta">avg absolute error</div>
          </div>
          <div class="metric-card">
            <div class="label">RMSE</div>
            <div class="value">${m['rmse']:,.0f}</div>
            <div class="delta">root mean squared</div>
          </div>
          <div class="metric-card">
            <div class="label">R² Score</div>
            <div class="value">{m['r2']:.3f}</div>
            <div class="delta">1.0 = perfect fit</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance chart
        if 'model' in st.session_state:
            fi_df = get_feature_importance(st.session_state['model'], X)
            if fi_df is not None:
                st.markdown("**Top Feature Importances**")
                fig = px.bar(
                    fi_df, x='Importance', y='Feature', orientation='h',
                    template='plotly_dark',
                    color='Importance', color_continuous_scale='Purples'
                )
                fig.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=10, b=10), height=420
                )
                st.plotly_chart(fig, use_container_width=True)

        # Actual vs predicted scatter
        if 'model' in st.session_state:
            model_eval = st.session_state['model']
            y_pred_plot = model_eval.predict(X_test)
            scatter_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_plot})
            fig2 = px.scatter(
                scatter_df, x='Actual', y='Predicted',
                template='plotly_dark', opacity=0.4,
                color_discrete_sequence=["#7c3aed"]
            )
            fig2.add_shape(type="line", x0=scatter_df['Actual'].min(), y0=scatter_df['Actual'].min(),
                           x1=scatter_df['Actual'].max(), y1=scatter_df['Actual'].max(),
                           line=dict(color="#a78bfa", dash="dash"))
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                               margin=dict(t=10, b=10), height=350,
                               title="Actual vs. Predicted Salary")
            st.plotly_chart(fig2, use_container_width=True)
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
        model = st.session_state['model']
        X_columns = st.session_state['X_columns']
        df_ref = st.session_state.get('df', df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Info**")
            age = st.slider("Age", int(df_ref['age'].min()), int(df_ref['age'].max()), 30)
            gender = st.radio("Gender", df_ref['gender'].unique().tolist())
            race = st.selectbox("Race", df_ref['race'].unique().tolist())
            native_country = st.selectbox("Native Country", sorted(df_ref['native-country'].unique().tolist()))

        with col2:
            st.markdown("**Education & Work**")
            educational_num = st.slider(
                "Education Level (1–16)", int(df_ref['educational-num'].min()),
                int(df_ref['educational-num'].max()), 10,
                help="1=low, 16=Doctorate"
            )
            workclass = st.selectbox("Work Class", df_ref['workclass'].unique().tolist())
            occupation = st.selectbox("Occupation", df_ref['occupation'].unique().tolist())
            hours_per_week = st.slider("Hours / Week", int(df_ref['hours-per-week'].min()),
                                       int(df_ref['hours-per-week'].max()), 40)

        with col3:
            st.markdown("**Household & Capital**")
            marital_status = st.selectbox("Marital Status", df_ref['marital-status'].unique().tolist())
            relationship = st.selectbox("Relationship", df_ref['relationship'].unique().tolist())
            capital_gain = st.number_input("Capital Gain ($)", min_value=0,
                                           max_value=int(df_ref['capital-gain'].max()), value=0)
            capital_loss = st.number_input("Capital Loss ($)", min_value=0,
                                           max_value=int(df_ref['capital-loss'].max()), value=0)
            fnlwgt = st.number_input("Final Weight (fnlwgt)",
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

                # Context bar
                nat_avg = df_ref['income_numeric'].mean()
                pct_above = (df_ref['income_numeric'] < predicted).mean() * 100
                st.markdown(f"""
                <div style="margin-top:1rem; display:flex; gap:1rem; flex-wrap:wrap">
                  <div class="pill">📈 Dataset avg: ${nat_avg:,.0f}</div>
                  <div class="pill">🏆 Higher than {pct_above:.0f}% of dataset</div>
                </div>
                """, unsafe_allow_html=True)

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
        model = st.session_state['model']
        X_columns = st.session_state['X_columns']

        with st.expander("📄 View expected CSV format"):
            st.code(
                "age,workclass,fnlwgt,educational-num,marital-status,occupation,"
                "relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country\n"
                "35,Private,200000,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,40,United-States\n"
                "28,Local-gov,150000,10,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States",
                language="csv"
            )

            # Downloadable sample
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
                    # Strip whitespace
                    for col in batch_data.select_dtypes(include='object').columns:
                        batch_data[col] = batch_data[col].str.strip()

                    batch_pred_input = batch_data[X_columns]
                    preds = model.predict(batch_pred_input)
                    batch_data['Predicted_Salary'] = preds.round(0).astype(int)

                    st.success(f"✅ Predicted salaries for {len(batch_data):,} records.")

                    # Summary stats
                    pa, pb, pc = st.columns(3)
                    pa.metric("Average Salary", f"${preds.mean():,.0f}")
                    pb.metric("Min Salary", f"${preds.min():,.0f}")
                    pc.metric("Max Salary", f"${preds.max():,.0f}")

                    # Distribution chart
                    fig = px.histogram(
                        x=preds, nbins=30, template='plotly_dark',
                        labels={'x': 'Predicted Salary'},
                        color_discrete_sequence=["#7c3aed"]
                    )
                    fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                                      margin=dict(t=10, b=10), height=280,
                                      title="Salary Distribution (Batch)")
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(batch_data, use_container_width=True)

                    csv_out = batch_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv_out,
                        file_name='predicted_salaries.csv',
                        mime='text/csv'
                    )
                except KeyError as ke:
                    st.error(f"Missing column in your CSV: {ke}. Make sure all required columns are present.")
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")
