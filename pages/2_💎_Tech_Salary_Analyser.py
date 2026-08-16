import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import base64
import os

# Helper to encode logo image for HTML rendering
def get_logo_b64():
    if os.path.exists("tech_analyzer_logo.png"):
        with open("tech_analyzer_logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_b64 = get_logo_b64()


# ==========================================
# 1. Custom Glowing Modern CSS Styling
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Reorder Sidebar elements so top-brand-header is at the VERY TOP */
    [data-testid="stSidebar"] > div:first-child {
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="stSidebarUserContent"] {
        order: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="stSidebarNav"] {
        order: 2 !important;
    }

    .top-brand-header {
        order: -10 !important;
    }

    /* Glowing Pulsing Animations */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); }
        100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }

    /* Glowing Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #0369A1 50%, #090D16 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 18px;
        padding: 26px 34px;
        margin-bottom: 24px;
        box-shadow: 0 10px 35px -10px rgba(56, 189, 248, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        top: -40%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, rgba(0, 0, 0, 0) 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #BAE6FD 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.4);
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        font-weight: 400;
        margin-bottom: 14px;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #7DD3FC;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #38BDF8;
        border-radius: 50%;
        animation: pulse-glow 2s infinite;
    }

    /* Output Results Styling */
    .result-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        position: relative;
        border: 1px solid rgba(56, 189, 248, 0.6);
        box-shadow: 0 0 40px rgba(56, 189, 248, 0.35), inset 0 0 25px rgba(56, 189, 248, 0.15);
        backdrop-filter: blur(12px);
    }

    .score-number {
        font-size: 3.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 12px 0 6px 0;
        color: #38BDF8;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.7);
    }

    .salary-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 14px;
        background: rgba(56, 189, 248, 0.18);
        color: #7DD3FC;
        border: 1px solid rgba(56, 189, 248, 0.5);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin-top: 18px;
    }

    .mini-metric {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: left;
        transition: all 0.25s ease;
        backdrop-filter: blur(8px);
    }

    .mini-metric:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.25);
    }

    .mini-metric-title {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .mini-metric-value {
        color: #F8FAFC;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Sidebar & Button Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19;
        border-right: 1px solid #1E293B;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.25s ease;
        background: #1E293B;
        border: 1px solid #334155;
        color: #F8FAFC;
    }

    .stButton>button:hover {
        border-color: rgba(56, 189, 248, 0.8);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Load ANN Regression Model & Preprocessors
# ==========================================

@st.cache_resource
def load_salary_artifacts():
    model = tf.keras.models.load_model("regression_model.h5", compile=False)

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    scaler_file = "scaler_salary.pkl" if os.path.exists("scaler_salary.pkl") else "scaler.pkl"
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler

try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_salary_artifacts()
except Exception as e:
    st.error(f"Error loading salary prediction artifacts: {e}")
    st.stop()


# ==========================================
# 3. Session State Management & Presets
# ==========================================

salary_defaults = {
    'sal_geography': 'France',
    'sal_gender': 'Female',
    'sal_age': 38,
    'sal_tenure': 5,
    'sal_balance': 75000.0,
    'sal_credit_score': 650,
    'sal_num_products': 2,
    'sal_has_credit_card': 'Yes',
    'sal_is_active_member': 'Active',
    'sal_exited': 'Retained (Active)'
}

for k, v in salary_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_salary_preset(preset_type):
    if preset_type == 'executive':
        st.session_state['sal_geography'] = 'France'
        st.session_state['sal_gender'] = 'Male'
        st.session_state['sal_age'] = 46
        st.session_state['sal_tenure'] = 8
        st.session_state['sal_balance'] = 145000.0
        st.session_state['sal_credit_score'] = 780
        st.session_state['sal_num_products'] = 2
        st.session_state['sal_has_credit_card'] = 'Yes'
        st.session_state['sal_is_active_member'] = 'Active'
        st.session_state['sal_exited'] = 'Retained (Active)'
    elif preset_type == 'professional':
        st.session_state['sal_geography'] = 'Germany'
        st.session_state['sal_gender'] = 'Female'
        st.session_state['sal_age'] = 34
        st.session_state['sal_tenure'] = 4
        st.session_state['sal_balance'] = 82000.0
        st.session_state['sal_credit_score'] = 680
        st.session_state['sal_num_products'] = 1
        st.session_state['sal_has_credit_card'] = 'Yes'
        st.session_state['sal_is_active_member'] = 'Active'
        st.session_state['sal_exited'] = 'Retained (Active)'
    elif preset_type == 'junior':
        st.session_state['sal_geography'] = 'Spain'
        st.session_state['sal_gender'] = 'Male'
        st.session_state['sal_age'] = 23
        st.session_state['sal_tenure'] = 1
        st.session_state['sal_balance'] = 15000.0
        st.session_state['sal_credit_score'] = 610
        st.session_state['sal_num_products'] = 1
        st.session_state['sal_has_credit_card'] = 'No'
        st.session_state['sal_is_active_member'] = 'Inactive'
        st.session_state['sal_exited'] = 'Churned (Exited)'


# ==========================================
# 4. Sidebar - Controls & Metadata
# ==========================================

with st.sidebar:
    st.markdown("### 📋 Sample Profiles")
    st.caption("Click to auto-fill customer scenario:")

    if st.button("💎 Executive Profile", use_container_width=True):
        load_salary_preset('executive')
        st.rerun()
    if st.button("💼 Senior Professional", use_container_width=True):
        load_salary_preset('professional')
        st.rerun()
    if st.button("🌱 Junior Account", use_container_width=True):
        load_salary_preset('junior')
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 Model Metadata")
    st.markdown("""
    - **Model**: Artificial Neural Network
    - **Task**: Annual Salary Regression
    - **Input Vector**: 12 Features
    - **Scaler**: StandardScaler
    - **Status**: <span style="color:#10B981; font-weight:700;">Operational</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Enterprise Valuation & Compensation Intelligence")


# ==========================================
# 5. Hero Header Banner
# ==========================================

if logo_b64:
    st.markdown(f"""
    <div class="hero-banner" style="display: flex; align-items: center; gap: 24px;">
        <img src="data:image/png;base64,{logo_b64}" style="width: 88px; height: 88px; border-radius: 18px; box-shadow: 0 0 30px rgba(56, 189, 248, 0.5); border: 2px solid rgba(255, 255, 255, 0.2); flex-shrink: 0;">
        <div>
            <div class="pill-badge">
                <span class="status-dot"></span> TECH ANALYZER Valuation Online
            </div>
            <div class="hero-title">Tech Salary & Compensation Analyser</div>
            <div class="hero-subtitle">
                Predict annual customer compensation and financial valuation using deep learning multi-variable regression metrics.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-banner">
        <div class="pill-badge">
            <span class="status-dot"></span> TECH ANALYZER Valuation Online
        </div>
        <div class="hero-title">Tech Salary & Compensation Analyser</div>
        <div class="hero-subtitle">
            Predict annual customer compensation and financial valuation using deep learning multi-variable regression metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 6. Main Layout: Input Columns & Results
# ==========================================

col_inputs, col_output = st.columns([1.15, 0.85], gap="large")

with col_inputs:
    st.markdown("### 📝 Customer Attributes")

    tab1, tab2, tab3 = st.tabs([
        "👤 Demographics", 
        "💳 Financial Profile", 
        "📊 Activity & Status"
    ])

    with tab1:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        col_t1_a, col_t1_b = st.columns(2)
        with col_t1_a:
            geography = st.selectbox(
                "Geography",
                ["France", "Germany", "Spain"],
                key="sal_geography",
                help="Customer location country"
            )
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                key="sal_gender",
                help="Customer gender"
            )
        with col_t1_b:
            age = st.slider(
                "Age (Years)",
                min_value=18,
                max_value=92,
                key="sal_age",
                help="Customer age in years"
            )
            tenure = st.slider(
                "Tenure (Years)",
                min_value=0,
                max_value=10,
                key="sal_tenure",
                help="Years customer has been with bank"
            )

    with tab2:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            key="sal_credit_score",
            step=5,
            help="Credit rating score (300 - 900)"
        )
        balance = st.number_input(
            "Account Balance ($)",
            min_value=0.0,
            key="sal_balance",
            step=1000.0,
            format="%.2f",
            help="Current bank account balance"
        )
        exited = st.selectbox(
            "Account Status",
            ["Retained (Active)", "Churned (Exited)"],
            key="sal_exited",
            help="Historical customer retention status"
        )

    with tab3:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        col_t3_a, col_t3_b = st.columns(2)
        with col_t3_a:
            num_products = st.selectbox(
                "Number of Bank Products",
                [1, 2, 3, 4],
                key="sal_num_products",
                help="Total banking products subscribed"
            )
            has_credit_card = st.selectbox(
                "Has Credit Card?",
                ["Yes", "No"],
                key="sal_has_credit_card",
                help="Is customer a credit card holder?"
            )
        with col_t3_b:
            is_active_member = st.selectbox(
                "Account Activity Status",
                ["Active", "Inactive"],
                key="sal_is_active_member",
                help="Is customer actively using bank services?"
            )


# ==========================================
# 7. Inference Pipeline & Model Prediction
# ==========================================

has_cr_card_val = 1 if has_credit_card == "Yes" else 0
is_active_val = 1 if is_active_member == "Active" else 0
exited_val = 1 if exited == "Churned (Exited)" else 0

# Encode Gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-Hot Encode Geography
geo_df_input = pd.DataFrame([{"Geography": geography}])
geo_encoded = onehot_encoder_geo.transform(geo_df_input[["Geography"]])
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Input dataframe matching regression training features
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card_val],
    "IsActiveMember": [is_active_val],
    "Exited": [exited_val]
})

input_data_encoded = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# Scale & Predict
input_scaled = scaler.transform(input_data_encoded)
prediction = model.predict(input_scaled, verbose=0)
predicted_salary = float(prediction[0][0])
monthly_salary = predicted_salary / 12.0


# ==========================================
# 8. Output Dashboard Panel
# ==========================================

with col_output:
    st.markdown("### 📊 Valuation Analysis")

    # Determine salary tier
    if predicted_salary >= 120000.0:
        tier_badge = "💎 HIGH INCOME TIER"
        tier_subtext = "Executive compensation level estimated."
    elif predicted_salary >= 70000.0:
        tier_badge = "💼 MID-UPPER EARNER"
        tier_subtext = "Senior professional compensation level estimated."
    else:
        tier_badge = "🌱 STANDARD EARNER TIER"
        tier_subtext = "Standard entry to mid compensation level estimated."

    # Render Result Card
    result_card_html = f"""<div class="result-card">
<div class="salary-badge">{tier_badge}</div>
<div style="font-size: 0.85rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">Predicted Annual Salary</div>
<div class="score-number">${predicted_salary:,.2f}</div>
<div style="font-size: 1.05rem; font-weight: 600; color: #38BDF8; margin-top: 4px; text-shadow: 0 0 15px rgba(56,189,248,0.5);">~${monthly_salary:,.2f} / month</div>
<div style="font-size: 0.88rem; color: #94A3B8; margin-top: 10px;">{tier_subtext}</div>
</div>"""
    st.markdown(result_card_html, unsafe_allow_html=True)

    # Mini Metrics Matrix
    credit_tier = "Excellent" if credit_score > 750 else ("Good" if credit_score > 650 else "Fair")
    balance_ratio = (balance / max(predicted_salary, 1.0)) * 100.0

    metrics_html = f"""<div class="metric-grid">
<div class="mini-metric">
<div class="mini-metric-title">Monthly Estimate</div>
<div class="mini-metric-value">${monthly_salary:,.0f}/mo</div>
</div>
<div class="mini-metric">
<div class="mini-metric-title">Credit Rating</div>
<div class="mini-metric-value">{credit_tier} ({credit_score})</div>
</div>
<div class="mini-metric">
<div class="mini-metric-title">Balance / Income</div>
<div class="mini-metric-value">{balance_ratio:.1f}%</div>
</div>
<div class="mini-metric">
<div class="mini-metric-title">Account Status</div>
<div class="mini-metric-value">{"Active" if is_active_val == 1 else "Inactive"}</div>
</div>
</div>"""
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Strategic Financial Insights Box
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
    with st.expander("💡 **Valuation Insights & Wealth Profile**", expanded=True):
        st.markdown(f"""
        - 📈 **Annual Compensation**: Predicted at **${predicted_salary:,.2f}** based on demographic & banking indicators.
        - 🏦 **Liquidity Ratio**: Balance of **${balance:,.2f}** represents **{balance_ratio:.1f}%** of estimated annual income.
        - 🎯 **Target Products**: Customer holds **{num_products}** product(s). Premium wealth & investment offerings recommended for this tier.
        """)
