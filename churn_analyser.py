import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
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
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes meter-shimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glowing Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 60%, #090D16 100%);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 18px;
        padding: 26px 34px;
        margin-bottom: 24px;
        box-shadow: 0 10px 35px -10px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15);
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
        background: radial-gradient(circle, rgba(129, 140, 248, 0.25) 0%, rgba(0, 0, 0, 0) 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #C7D2FE 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
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
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        animation: pulse-glow 2s infinite;
    }

    /* Glowing Output Result Cards */
    .result-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        position: relative;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }

    .result-card.high-risk {
        border: 1px solid rgba(239, 68, 68, 0.6);
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.35), inset 0 0 25px rgba(239, 68, 68, 0.15);
    }

    .result-card.low-risk {
        border: 1px solid rgba(16, 185, 129, 0.6);
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.35), inset 0 0 25px rgba(16, 185, 129, 0.15);
    }

    .result-card.medium-risk {
        border: 1px solid rgba(245, 158, 11, 0.6);
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.35), inset 0 0 25px rgba(245, 158, 11, 0.15);
    }

    .score-number {
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 12px 0 6px 0;
    }

    .high-risk .score-number { color: #F87171; text-shadow: 0 0 25px rgba(239, 68, 68, 0.6); }
    .low-risk .score-number { color: #34D399; text-shadow: 0 0 25px rgba(16, 185, 129, 0.6); }
    .medium-risk .score-number { color: #FBBF24; text-shadow: 0 0 25px rgba(245, 158, 11, 0.6); }

    .risk-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .badge-high { background: rgba(239, 68, 68, 0.18); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.5); box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
    .badge-low { background: rgba(16, 185, 129, 0.18); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.5); box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }
    .badge-medium { background: rgba(245, 158, 11, 0.18); color: #FDE047; border: 1px solid rgba(245, 158, 11, 0.5); box-shadow: 0 0 15px rgba(245, 158, 11, 0.3); }

    /* Animated Glowing Progress Bar Container */
    .meter-container {
        background: #090D16;
        border-radius: 12px;
        height: 16px;
        width: 100%;
        overflow: hidden;
        margin: 18px 0;
        border: 1px solid #334155;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.6);
    }

    .meter-fill {
        height: 100%;
        border-radius: 12px;
        background-size: 200% 200% !important;
        animation: meter-shimmer 3s ease infinite !important;
        transition: width 0.6s ease-in-out;
    }

    /* Glowing Mini Metric Cards */
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
        border-color: rgba(99, 102, 241, 0.6);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
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
        border-color: rgba(99, 102, 241, 0.8);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }

    /* Expanders & Tabs styling */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        border-radius: 10px !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Load ANN Model & Preprocessors
# ==========================================

@st.cache_resource
def load_churn_artifacts():
    model = load_model("model.h5", compile=False)

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler

try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_churn_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# ==========================================
# 3. Session State Management & Presets
# ==========================================

defaults = {
    'geography': 'France',
    'gender': 'Female',
    'age': 38,
    'tenure': 5,
    'balance': 75000.0,
    'credit_score': 650,
    'num_products': 2,
    'has_credit_card': 'Yes',
    'is_active_member': 'Active',
    'estimated_salary': 60000.0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_preset(preset_type):
    if preset_type == 'high_risk':
        st.session_state['geography'] = 'Germany'
        st.session_state['gender'] = 'Female'
        st.session_state['age'] = 54
        st.session_state['tenure'] = 2
        st.session_state['balance'] = 125000.0
        st.session_state['credit_score'] = 510
        st.session_state['num_products'] = 1
        st.session_state['has_credit_card'] = 'No'
        st.session_state['is_active_member'] = 'Inactive'
        st.session_state['estimated_salary'] = 85000.0
    elif preset_type == 'loyal':
        st.session_state['geography'] = 'France'
        st.session_state['gender'] = 'Male'
        st.session_state['age'] = 31
        st.session_state['tenure'] = 7
        st.session_state['balance'] = 60000.0
        st.session_state['credit_score'] = 760
        st.session_state['num_products'] = 2
        st.session_state['has_credit_card'] = 'Yes'
        st.session_state['is_active_member'] = 'Active'
        st.session_state['estimated_salary'] = 110000.0


# ==========================================
# 4. Sidebar - Controls & Metadata
# ==========================================

with st.sidebar:
    st.markdown("### 📋 Customer Presets")
    st.caption("Click a scenario to auto-fill inputs:")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🔴 High Risk", use_container_width=True):
            load_preset('high_risk')
            st.rerun()
    with col_p2:
        if st.button("🟢 Loyal Account", use_container_width=True):
            load_preset('loyal')
            st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 Model Metadata")
    st.markdown("""
    - **Model**: Multi-Layer ANN
    - **Input Vector**: 12 Features
    - **Encoder**: Scikit-Learn
    - **Scaler**: StandardScaler
    - **Status**: <span style="color:#10B981; font-weight:700;">Operational</span>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Enterprise Retention Intelligence Dashboard")


# ==========================================
# 5. Hero Header Banner
# ==========================================

if logo_b64:
    st.markdown(f"""
    <div class="hero-banner" style="display: flex; align-items: center; gap: 24px;">
        <img src="data:image/png;base64,{logo_b64}" style="width: 88px; height: 88px; border-radius: 18px; box-shadow: 0 0 30px rgba(99, 102, 241, 0.5); border: 2px solid rgba(255, 255, 255, 0.2); flex-shrink: 0;">
        <div>
            <div class="pill-badge">
                <span class="status-dot"></span> TECH ANALYZER AI Engine Online
            </div>
            <div class="hero-title">Customer Churn Prediction Platform</div>
            <div class="hero-subtitle">
                Analyze customer behavioral & financial metrics in real-time to forecast churn probability and deploy proactive retention strategies.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-banner">
        <div class="pill-badge">
            <span class="status-dot"></span> TECH ANALYZER AI Engine Online
        </div>
        <div class="hero-title">Customer Churn Prediction Platform</div>
        <div class="hero-subtitle">
            Analyze customer behavioral & financial metrics in real-time to forecast churn probability and deploy proactive retention strategies.
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
        "📊 Account Activity"
    ])

    with tab1:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        col_t1_a, col_t1_b = st.columns(2)
        with col_t1_a:
            geography = st.selectbox(
                "Geography",
                ["France", "Germany", "Spain"],
                key="geography",
                help="Customer location country"
            )
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                key="gender",
                help="Customer gender"
            )
        with col_t1_b:
            age = st.slider(
                "Age (Years)",
                min_value=18,
                max_value=92,
                key="age",
                help="Customer age in years"
            )
            tenure = st.slider(
                "Tenure (Years)",
                min_value=0,
                max_value=10,
                key="tenure",
                help="Years customer has been with the bank"
            )

    with tab2:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            key="credit_score",
            step=5,
            help="Credit rating score (300 - 900)"
        )
        balance = st.number_input(
            "Account Balance ($)",
            min_value=0.0,
            key="balance",
            step=1000.0,
            format="%.2f",
            help="Current bank account balance"
        )
        estimated_salary = st.number_input(
            "Estimated Annual Salary ($)",
            min_value=0.0,
            key="estimated_salary",
            step=1000.0,
            format="%.2f",
            help="Estimated yearly income"
        )

    with tab3:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        col_t3_a, col_t3_b = st.columns(2)
        with col_t3_a:
            num_products = st.selectbox(
                "Number of Bank Products",
                [1, 2, 3, 4],
                key="num_products",
                help="Total banking products subscribed"
            )
            has_credit_card = st.selectbox(
                "Has Credit Card?",
                ["Yes", "No"],
                key="has_credit_card",
                help="Is customer a credit card holder?"
            )
        with col_t3_b:
            is_active_member = st.selectbox(
                "Account Activity Status",
                ["Active", "Inactive"],
                key="is_active_member",
                help="Is customer actively using bank services?"
            )


# ==========================================
# 7. Inference Pipeline & Model Evaluation
# ==========================================

# Data mapping for model
has_cr_card_val = 1 if has_credit_card == "Yes" else 0
is_active_val = 1 if is_active_member == "Active" else 0

input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card_val,
    "IsActiveMember": is_active_val,
    "EstimatedSalary": estimated_salary
}])

# Encode Gender
input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])

# One-Hot Encode Geography
geo_encoded = onehot_encoder_geo.transform(input_df[["Geography"]])
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Concatenate Encoded Data
input_df_encoded = pd.concat(
    [input_df.drop("Geography", axis=1), geo_encoded_df],
    axis=1
)

# Scale Features
input_scaled = scaler.transform(input_df_encoded)

# Model Prediction
prediction = model.predict(input_scaled, verbose=0)
prediction_proba = float(prediction[0][0])
churn_percent = prediction_proba * 100


# ==========================================
# 8. Output Dashboard Panel
# ==========================================

with col_output:
    st.markdown("### 📊 Churn Risk Analysis")

    # Determine risk category & styling
    if churn_percent >= 50.0:
        card_class = "high-risk"
        badge_class = "badge-high"
        badge_text = "🔴 HIGH CHURN RISK"
        meter_color = "linear-gradient(90deg, #EF4444 0%, #F87171 100%)"
        headline = "Customer is likely to churn"
        subtext = "High probability of account termination. Proactive retention required."
    elif churn_percent >= 30.0:
        card_class = "medium-risk"
        badge_class = "badge-medium"
        badge_text = "🟡 MODERATE RISK"
        meter_color = "linear-gradient(90deg, #F59E0B 0%, #FBBF24 100%)"
        headline = "Moderate churn risk detected"
        subtext = "Customer exhibits potential churn signals. Monitor engagement."
    else:
        card_class = "low-risk"
        badge_class = "badge-low"
        badge_text = "🟢 LOW RISK"
        meter_color = "linear-gradient(90deg, #10B981 0%, #34D399 100%)"
        headline = "Customer is unlikely to churn"
        subtext = "Customer profile demonstrates strong loyalty & account stability."

    # Render Main Result Card
    result_card_html = f"""<div class="result-card {card_class}">
<div class="risk-badge {badge_class}">{badge_text}</div>
<div style="font-size: 0.85rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">Predicted Churn Probability</div>
<div class="score-number">{churn_percent:.1f}%</div>
<div class="meter-container">
<div class="meter-fill" style="width: {churn_percent:.1f}%; background: {meter_color}; box-shadow: 0 0 15px rgba(99,102,241,0.5);"></div>
</div>
<div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-top: 10px;">{headline}</div>
<div style="font-size: 0.88rem; color: #94A3B8; margin-top: 4px;">{subtext}</div>
</div>"""
    st.markdown(result_card_html, unsafe_allow_html=True)

    # Mini Metrics Matrix
    credit_tier = "Excellent" if credit_score > 750 else ("Good" if credit_score > 650 else "Fair")
    balance_tier = "High Wealth" if balance > 100000 else ("Moderate" if balance > 30000 else "Low Balance")

    metrics_html = f"""<div class="metric-grid">
<div class="mini-metric">
<div class="mini-metric-title">Credit Standing</div>
<div class="mini-metric-value">{credit_tier} ({credit_score})</div>
</div>
<div class="mini-metric">
<div class="mini-metric-title">Account Balance</div>
<div class="mini-metric-value">{balance_tier}</div>
</div>
<div class="mini-metric">
<div class="mini-metric-title">Product Portfolio</div>
<div class="mini-metric-value">{num_products} Product(s)</div>
</div>
<div class="mini-metric">
<div class="mini-metric-title">Activity Status</div>
<div class="mini-metric-value">{is_active_member}</div>
</div>
</div>"""
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Strategic AI Recommendations Box
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
    with st.expander("💡 **Recommended Retention Action Plan**", expanded=True):
        if churn_percent >= 50.0:
            st.markdown("""
            - 🚨 **Priority Outreach**: Schedule a call with a Senior Relationship Manager within 24 hours.
            - 🎁 **Loyalty Incentive**: Offer 0.50% APY bonus interest on savings balance or waiver on transaction fees.
            - 💳 **Product Bundling**: Customer currently has 1 product. Cross-sell wealth management or credit products to increase stickiness.
            """)
        elif churn_percent >= 30.0:
            st.markdown("""
            - 📩 **Engagement Campaign**: Include customer in automated email nurture campaign highlighting digital banking features.
            - 📊 **Feedback Survey**: Send brief customer satisfaction survey to identify pain points early.
            - 🎯 **Targeted Promotions**: Offer cashback incentives on credit card usage.
            """)
        else:
            st.markdown("""
            - ✅ **Maintain Relationship**: Standard automated relationship management.
            - 🌟 **VIP Invitation**: Invite customer to premium reward programs or exclusive credit line increases.
            """)
