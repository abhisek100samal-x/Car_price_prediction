import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─── Shared theme ───────────────────────────────────────────────────────────────
NAVY, BLUE, LIGHT_BLUE, RED = "#0f3460", "#1a73e8", "#a0c4ff", "#e63946"
PLOTLY_TEMPLATE = "plotly_white"
CONTINUOUS_SCALE = ["#e8ecf8", "#a0c4ff", "#1a73e8", "#0f3460"]

def style_fig(fig, height=380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Segoe UI, sans-serif", color="#1a1a2e"),
        plot_bgcolor="#f5f7ff",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Force white background everywhere */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], section.main, .block-container {
    background-color: #ffffff !important;
    color: #1a1a2e !important;
}

[data-testid="stSidebar"] {
    background-color: #f5f7ff !important;
}

/* Hide default streamlit header chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Typography */
h1, h2, h3, h4 { color: #1a1a2e !important; font-family: 'Segoe UI', sans-serif; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    color: white;
    margin-bottom: 2rem;
}
.hero h1 { color: white !important; font-size: 2.2rem; margin: 0; font-weight: 700; }
.hero p  { color: #a0c4ff; margin: 0.5rem 0 0; font-size: 1rem; }

/* Metric cards */
.metric-card {
    background: #f5f7ff;
    border-left: 4px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val  { font-size: 2rem; font-weight: 700; color: #0f3460; }
.metric-card .lbl  { font-size: 0.8rem; color: #555; text-transform: uppercase; letter-spacing: .05em; }

/* Section headings */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 2px solid #e8ecf8;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* Prediction result box */
.pred-box {
    background: linear-gradient(135deg, #0f3460, #1a73e8);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: white;
}
.pred-box .amount { font-size: 2.8rem; font-weight: 800; }
.pred-box .label  { font-size: 0.95rem; opacity: 0.85; margin-top: 0.3rem; }

/* Upload zone */
.stFileUploader > div {
    border: 2px dashed #a0c4ff !important;
    border-radius: 12px !important;
    background: #f5f7ff !important;
}

/* Buttons */
div.stButton > button {
    background: #0f3460 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600 !important; color: #1a1a2e !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #0f3460 !important;
    color: #0f3460 !important;
}

/* Selectbox / Slider label */
label { color: #333 !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚗 Ford Car Price Predictor</h1>
  <p>Upload your dataset → explore patterns → get instant price estimates</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar — Upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader("Upload ford.csv", type=["csv"])
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This app trains a **Linear Regression** model on Ford used-car listings "
        "and lets you predict price from key vehicle attributes."
    )

# ─── Load & prepare data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train(raw_bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))

    # ── basic cleaning
    df = df.dropna()
    df = df[df["price"] > 0]
    df = df[df["mileage"] >= 0]
    df = df[df["engineSize"] > 0]

    X = df.drop(columns=["price"])
    Y = df["price"]

    # ── One-hot encode categoricals
    X_enc = pd.get_dummies(X, columns=["model", "transmission", "fuelType"], drop_first=True)
    X_enc = X_enc.astype(int)

    # ── Scale numerics
    numerical_cols = ["year", "mileage", "tax", "engineSize"]
    # only scale columns that actually exist
    num_present = [c for c in numerical_cols if c in X_enc.columns]
    scaler = StandardScaler()
    X_enc[num_present] = scaler.fit_transform(X_enc[num_present])

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, Y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape
    adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

    return df, model, scaler, X_enc.columns.tolist(), r2, adj_r2, y_test, y_pred, num_present


# ─── State guard ────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload **ford.csv** from the sidebar to get started.")
    st.stop()

with st.spinner("Training model…"):
    df, model, scaler, feature_cols, r2, adj_r2, y_test, y_pred, num_present = load_and_train(
        uploaded.read()
    )

# ─── Top metric strip ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    (f"{len(df):,}", "Total Records"),
    (f"{df['model'].nunique()}", "Car Models"),
    (f"{r2*100:.1f}%", "R² Score"),
    (f"{adj_r2*100:.1f}%", "Adjusted R²"),
]
for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 EDA Charts", "🤖 Model Insights", "💰 Predict Price"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Data Overview
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Dataset Shape</div>', unsafe_allow_html=True)
        st.write(f"**Rows:** {df.shape[0]:,} &nbsp;&nbsp; **Columns:** {df.shape[1]}")
        st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
        miss = df.isnull().sum()
        st.dataframe(miss[miss > 0].rename("Missing") if miss.sum() > 0
                     else pd.DataFrame({"Status": ["No missing values ✅"]}),
                     use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().T.style.format("{:.1f}"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════
with tab2:
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-title">Price Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="price", nbins=50, marginal="box",
                            color_discrete_sequence=[NAVY])
        fig.update_layout(xaxis_title="Price (£)", yaxis_title="Count")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-title">Price vs Mileage</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="mileage", y="price", color="fuelType",
                          opacity=0.5, color_discrete_sequence=px.colors.qualitative.Bold,
                          hover_data=["model", "year"])
        fig.update_layout(xaxis_title="Mileage", yaxis_title="Price (£)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-title">Price by Transmission</div>', unsafe_allow_html=True)
        fig = px.violin(df, x="transmission", y="price", color="transmission", box=True,
                         color_discrete_sequence=[NAVY, BLUE, LIGHT_BLUE, "#ccc"])
        fig.update_layout(xaxis_title="", yaxis_title="Price (£)", showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-title">Price by Fuel Type</div>', unsafe_allow_html=True)
        fig = px.violin(df, x="fuelType", y="price", color="fuelType", box=True,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(xaxis_title="", yaxis_title="Price (£)", showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="section-title">Price vs Engine Size</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="engineSize", y="price", color="transmission",
                          opacity=0.55, color_discrete_sequence=[NAVY, BLUE, LIGHT_BLUE, "#ccc"])
        fig.update_layout(xaxis_title="Engine Size (L)", yaxis_title="Price (£)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with r3c2:
        st.markdown('<div class="section-title">Correlation with Price</div>', unsafe_allow_html=True)
        num_corr = df.corr(numeric_only=True)["price"].drop("price").sort_values()
        colors = [RED if v < 0 else NAVY for v in num_corr.values]
        fig = go.Figure(go.Bar(x=num_corr.values, y=num_corr.index, orientation="h",
                                marker_color=colors))
        fig.update_layout(xaxis_title="Correlation coefficient", yaxis_title="")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Average Price Trend by Year</div>', unsafe_allow_html=True)
    yearly = df.groupby("year")["price"].agg(["mean", "count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly["year"], y=yearly["count"], name="Listings",
                          marker_color=LIGHT_BLUE, yaxis="y2", opacity=0.6))
    fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["mean"], name="Avg. price",
                              mode="lines+markers", line=dict(color=NAVY, width=3)))
    fig.update_layout(
        yaxis=dict(title="Average Price (£)"),
        yaxis2=dict(title="Listing count", overlaying="y", side="right", showgrid=False),
        xaxis_title="Year",
    )
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    st.markdown('<div class="section-title">Correlation Heat Map</div>', unsafe_allow_html=True)
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)
    fig = px.imshow(corr_masked, text_auto=".2f", color_continuous_scale=CONTINUOUS_SCALE,
                     aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(title=""))
    st.plotly_chart(style_fig(fig, height=480), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════
with tab3:
    residuals = np.array(y_test) - y_pred

    m1, m2, m3 = st.columns(3)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    for col, (val, lbl) in zip([m1, m2, m3], [
        (f"£{mae:,.0f}", "Mean Abs. Error"),
        (f"£{rmse:,.0f}", "RMSE"),
        (f"{r2*100:.1f}%", "R² on Test Set"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
    st.write("")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predictions",
                                  marker=dict(color=BLUE, opacity=0.35, size=6)))
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect fit",
                                  line=dict(color=RED, dash="dash")))
        fig.update_layout(xaxis_title="Actual Price (£)", yaxis_title="Predicted Price (£)")
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Residuals vs Predicted</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                  marker=dict(color=NAVY, opacity=0.35, size=6)))
        fig.add_hline(y=0, line_dash="dash", line_color=RED)
        fig.update_layout(xaxis_title="Predicted Price (£)", yaxis_title="Residual (£)",
                           showlegend=False)
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    st.markdown('<div class="section-title">Residuals Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(residuals, nbins=60, color_discrete_sequence=[NAVY])
    fig.add_vline(x=0, line_dash="dash", line_color=RED)
    fig.update_layout(xaxis_title="Residual (£)", yaxis_title="Count", showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Top 15 Features by Coefficient Magnitude</div>',
                unsafe_allow_html=True)
    coef_df = (
        pd.DataFrame({"Feature": feature_cols, "Coefficient": model.coef_})
        .assign(Abs=lambda d: d["Coefficient"].abs())
        .sort_values("Abs", ascending=False)
        .head(15)
        .sort_values("Coefficient")
    )
    colors = [NAVY if v >= 0 else RED for v in coef_df["Coefficient"]]
    fig = go.Figure(go.Bar(x=coef_df["Coefficient"], y=coef_df["Feature"],
                            orientation="h", marker_color=colors))
    fig.add_vline(x=0, line_color="#333")
    fig.update_layout(xaxis_title="Coefficient value", yaxis_title="")
    st.plotly_chart(style_fig(fig, height=460), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Predict Price
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Enter Vehicle Details</div>', unsafe_allow_html=True)

    models_list   = sorted(df["model"].unique().tolist())
    trans_list    = sorted(df["transmission"].unique().tolist())
    fuel_list     = sorted(df["fuelType"].unique().tolist())
    year_min, year_max = int(df["year"].min()), int(df["year"].max())

    p1, p2, p3 = st.columns(3)
    with p1:
        sel_model   = st.selectbox("Model", models_list)
        sel_year    = st.slider("Year", year_min, year_max, 2018)
        sel_mileage = st.number_input("Mileage (miles)", min_value=0, max_value=200_000,
                                      value=20_000, step=1_000)
    with p2:
        sel_trans   = st.selectbox("Transmission", trans_list)
        sel_fuel    = st.selectbox("Fuel Type", fuel_list)
        sel_tax     = st.number_input("Road Tax (£/yr)", min_value=0, max_value=600,
                                      value=150, step=10)
    with p3:
        sel_mpg     = st.number_input("MPG", min_value=10.0, max_value=100.0,
                                      value=45.0, step=0.5)
        sel_engine  = st.selectbox("Engine Size (L)",
                                   sorted(df["engineSize"].unique().tolist()))

    if st.button("🔮 Predict Price"):
        try:
            row = pd.DataFrame([{
                "model": sel_model, "year": sel_year, "transmission": sel_trans,
                "mileage": sel_mileage, "fuelType": sel_fuel, "tax": sel_tax,
                "mpg": sel_mpg, "engineSize": float(sel_engine),
            }])

            # One-hot encode exactly as training
            row_enc = pd.get_dummies(row, columns=["model", "transmission", "fuelType"])
            row_enc = row_enc.astype(int)

            # Align columns
            for c in feature_cols:
                if c not in row_enc.columns:
                    row_enc[c] = 0
            row_enc = row_enc[feature_cols]

            # Scale numerics
            num_scale = [c for c in num_present if c in row_enc.columns]
            row_enc[num_scale] = scaler.transform(row_enc[num_scale])

            predicted = model.predict(row_enc)[0]
            predicted = max(0, predicted)

            st.markdown(f"""
            <div class="pred-box">
              <div class="amount">£{predicted:,.0f}</div>
              <div class="label">Estimated Market Price</div>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
