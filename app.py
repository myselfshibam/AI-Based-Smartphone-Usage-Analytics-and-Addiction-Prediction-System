import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# PAGE CONFIG & STYLING
# -----------------------------
st.set_page_config(
    page_title="Synaptiq AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, modern UI
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4A90E2, #50E3C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    /* Style the metric boxes */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #4A90E2;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA & MODEL (Mocked for safety)
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/mobile_usage_model.pkl')
    except Exception:
        model = None # Fallback if file isn't found locally
        
    try:
        df = pd.read_csv('data/final_mobile_usage_dataset.csv')
    except Exception:
        # Fallback dummy data if file isn't found locally for testing
        df = pd.DataFrame({
            'Health_Risk': ['Low', 'Medium', 'Low', 'High', 'Medium', 'High', 'Low', 'Low']
        })
    return model, df

model, df = load_assets()

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<p class="main-header">📱 Synaptiq AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Mobile Usage Analytics & Health Risk Prediction</p>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# LAYOUT: TABS
# -----------------------------
tab1, tab2 = st.tabs(["📊 Dashboard & Analytics", "🧠 Risk Predictor"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Dataset Overview")
        st.write("A quick look at the mobile usage dataset fueling our AI insights.")
        with st.expander("🔍 View Raw Data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            
    with col2:
        st.subheader("Health Risk Distribution")
        
        # Enhanced Plotly Chart
        risk_counts = df['Health_Risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        risk_chart = px.bar(
            risk_counts, 
            x='Risk Level', 
            y='Count',
            color='Risk Level',
            color_discrete_map={
                'Low': '#50E3C2',
                'Medium': '#F5A623',
                'High': '#D0021B'
            },
            title="Current User Health Risk Levels",
            text='Count'
        )
        
        # Clean up the chart layout
        risk_chart.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, l=0, r=0, b=0),
            xaxis_title=None,
            yaxis_title=None,
            showlegend=False
        )
        risk_chart.update_traces(textposition='outside')
        
        st.plotly_chart(risk_chart, use_container_width=True)

with tab2:
    st.subheader("Evaluate Your Digital Habits")
    st.write("Use the sliders below to calculate your personal addiction score.")
    
    # Use a container for the input form to create a card-like feel
    with st.container():
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            screen_time = st.slider('📺 Screen Time (hrs/day)', 1.0, 15.0, 5.0, 0.5)
            gaming_time = st.slider('🎮 Gaming Time (hrs/day)', 0.0, 10.0, 2.0, 0.5)
            
        with col_input2:
            social_media = st.slider('📱 Social Media (hrs/day)', 0.0, 10.0, 2.0, 0.5)
            streaming_time = st.slider('🍿 Streaming (hrs/day)', 0.0, 10.0, 1.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # -----------------------------
    # PREDICTION LOGIC
    # -----------------------------
    col_btn, col_result = st.columns([1, 2])
    
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict My Risk", type="primary", use_container_width=True)

    if predict_btn:
        with col_result:
            # Formula-based prediction
            addiction_score = (
                screen_time * 0.35 +
                gaming_time * 0.25 +
                social_media * 0.25 +
                streaming_time * 0.15
            )
            
            st.metric("Your Addiction Score", f"{addiction_score:.2f} / 10")
            
            # Display highly styled results based on the score
            if addiction_score >= 7:
                st.error("🚨 **High Addiction Risk**")
                st.info("""
                **Actionable Recommendations:**
                *   📉 Drastically reduce non-essential screen time.
                *   👀 Follow the 20-20-20 rule to prevent eye strain.
                *   🛑 Use strict app timers for gaming and social media.
                *   🚶‍♂️ Substitute digital time with outdoor activities.
                """)
                
            elif addiction_score >= 4:
                st.warning("⚠️ **Medium Addiction Risk**")
                st.info("""
                **Actionable Recommendations:**
                *   ⚖️ Try to maintain healthier screen habits.
                *   📵 Institute "no-phone" zones (e.g., at the dinner table).
                *   📉 Slightly reduce your social media footprint.
                """)
                
            else:
                st.success("✅ **Low Addiction Risk**")
                st.balloons()
                st.info("""
                **Actionable Recommendations:**
                *   🌟 Keep up the great work!
                *   🧘‍♂️ You have healthy and balanced digital habits.
                *   🔋 Continue to unplug before bedtime for optimal sleep.
                """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Python, Scikit-learn, Streamlit & Plotly"
    "</div>", 
    unsafe_allow_html=True
)