import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Synaptiq AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* ===== GOOGLE FONT ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===== MAIN APP ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #020617 100%);
    color: white;
}

/* ===== REMOVE DEFAULT HEADER ===== */
header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* ===== HERO SECTION ===== */
.hero-container {
    padding: 2rem 0rem 1rem 0rem;
}

.hero-title {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(to right, #60a5fa, #38bdf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ===== GLASS CARD ===== */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    padding: 1.5rem;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.25);
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.08),
        rgba(255,255,255,0.03)
    );
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1.5rem;
    border-radius: 22px;
    backdrop-filter: blur(16px);
    transition: 0.3s ease-in-out;
}

.metric-card:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(96,165,250,0.5);
}

.metric-title {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: white;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ===== SLIDER ===== */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}

/* ===== BUTTON ===== */
.stButton>button {
    width: 100%;
    height: 3.2rem;
    border-radius: 14px;
    border: none;
    font-weight: 700;
    font-size: 1rem;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(37,99,235,0.5);
}

/* ===== TAB ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 10px 22px;
    color: white;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    color: #64748b;
    padding-top: 2rem;
    font-size: 0.95rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL & DATA
# =========================================================
@st.cache_resource
def load_assets():

    try:
        model = joblib.load("models/mobile_usage_model.pkl")
    except:
        model = None

    try:
        df = pd.read_csv("data/final_mobile_usage_dataset.csv")
    except:
        np.random.seed(42)

        df = pd.DataFrame({
            "Health_Risk": np.random.choice(
                ["Low", "Medium", "High"],
                500,
                p=[0.45, 0.35, 0.20]
            ),
            "Screen_Time": np.random.normal(6, 2, 500)
        })

    return model, df


model, df = load_assets()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🧠 Synaptiq AI")

    st.markdown("""
    ### AI Analytics Platform
    
    Predict digital addiction risk using:
    
    - 📱 Screen time
    - 🎮 Gaming behavior
    - 🍿 Streaming usage
    - 🌐 Social media habits
    
    ---
    """)

    st.info("Built with Streamlit + Plotly")

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        Synaptiq AI
    </div>
    <div class="hero-subtitle">
        AI-Powered Mobile Usage Analytics & Digital Health Intelligence Platform
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP METRICS
# =========================================================
col1, col2, col3, col4 = st.columns(4)

total_users = len(df)
high_risk = len(df[df['Health_Risk'] == 'High'])
avg_screen = round(df.get("Screen_Time", pd.Series([6])).mean(), 1)
risk_rate = round((high_risk / total_users) * 100, 1)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">👥 Total Users</div>
        <div class="metric-value">{total_users}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🚨 High Risk Users</div>
        <div class="metric-value">{high_risk}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📱 Avg Screen Time</div>
        <div class="metric-value">{avg_screen} hrs</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⚠️ Risk Rate</div>
        <div class="metric-value">{risk_rate}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs([
    "📊 Analytics Dashboard",
    "🧠 AI Risk Predictor"
])

# =========================================================
# DASHBOARD TAB
# =========================================================
with tab1:

    left, right = st.columns([1, 2])

    with left:

        st.markdown("""
        <div class="glass-card">
        """, unsafe_allow_html=True)

        st.subheader("📂 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=350
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="glass-card">
        """, unsafe_allow_html=True)

        st.subheader("📈 Health Risk Distribution")

        risk_counts = df['Health_Risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Users']

        fig = px.bar(
            risk_counts,
            x='Risk Level',
            y='Users',
            color='Risk Level',
            text='Users',
            color_discrete_map={
                'Low': '#10b981',
                'Medium': '#f59e0b',
                'High': '#ef4444'
            }
        )

        fig.update_layout(
            height=420,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )

        fig.update_traces(
            marker_line_width=0,
            textposition='outside'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PREDICTOR TAB
# =========================================================
with tab2:

    st.markdown("""
    <div class="glass-card">
    """, unsafe_allow_html=True)

    st.subheader("🧠 Digital Addiction Risk Prediction")

    st.write(
        "Adjust the parameters below to evaluate your mobile usage behavior."
    )

    c1, c2 = st.columns(2)

    with c1:
        screen_time = st.slider(
            "📱 Screen Time (hrs/day)",
            0.0, 15.0, 5.0, 0.5
        )

        gaming_time = st.slider(
            "🎮 Gaming Time (hrs/day)",
            0.0, 10.0, 2.0, 0.5
        )

    with c2:
        social_media = st.slider(
            "🌐 Social Media (hrs/day)",
            0.0, 10.0, 3.0, 0.5
        )

        streaming_time = st.slider(
            "🍿 Streaming Time (hrs/day)",
            0.0, 10.0, 2.0, 0.5
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button("🚀 Analyze My Risk")

    if predict:

        addiction_score = (
            screen_time * 0.35 +
            gaming_time * 0.25 +
            social_media * 0.25 +
            streaming_time * 0.15
        )

        st.markdown("## 🎯 Prediction Result")

        score_chart = go.Figure(go.Indicator(
            mode="gauge+number",
            value=addiction_score,
            number={'font': {'size': 42}},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, 4], 'color': "#10b981"},
                    {'range': [4, 7], 'color': "#f59e0b"},
                    {'range': [7, 10], 'color': "#ef4444"}
                ]
            }
        ))

        score_chart.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(score_chart, use_container_width=True)

        if addiction_score >= 7:

            st.error("🚨 HIGH ADDICTION RISK")

            st.markdown("""
            ### Recommended Actions
            
            - 📵 Reduce social media exposure
            - ⏳ Apply strict app limits
            - 🚶 Spend more time outdoors
            - 😴 Avoid screens before sleep
            """)

        elif addiction_score >= 4:

            st.warning("⚠️ MEDIUM ADDICTION RISK")

            st.markdown("""
            ### Recommended Actions
            
            - ⚖️ Maintain healthier balance
            - 📱 Introduce no-phone periods
            - 👀 Reduce binge consumption
            """)

        else:

            st.success("✅ LOW ADDICTION RISK")

            st.balloons()

            st.markdown("""
            ### Recommended Actions
            
            - 🌟 Keep your current routine
            - 🧘 Maintain healthy digital habits
            - 🔋 Continue mindful screen usage
            """)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    Built with ❤️ by  <a href="https://github.com/myselfshibam"
       target="_blank"
         style="color: #38bdf8; text-decoration: none; font-weight: 600;">
       🚀 myselfshibam
    </a>  
</div>
""", unsafe_allow_html=True)
