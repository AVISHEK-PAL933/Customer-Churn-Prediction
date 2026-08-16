import streamlit as st
import base64
import os

# ==========================================
# 1. Page Configuration & Navigation Setup
# ==========================================

st.set_page_config(
    page_title="Tech Analyzer AI | Enterprise Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to encode logo image for HTML rendering
def get_logo_b64():
    if os.path.exists("tech_analyzer_logo.png"):
        with open("tech_analyzer_logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_b64 = get_logo_b64()

# Use official Streamlit logo API
if os.path.exists("tech_analyzer_logo.png"):
    try:
        st.logo("tech_analyzer_logo.png", icon_image="tech_analyzer_logo.png")
    except Exception:
        pass

# Custom CSS to force Top Brand Header above stSidebarNav (MODULES)
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# Render Brand Header at the VERY TOP of the Sidebar
with st.sidebar:
    if logo_b64:
        st.markdown(f"""
            <div class="top-brand-header" style="text-align: center; padding: 5px 0 15px 0;">
                <img src="data:image/png;base64,{logo_b64}" style="width: 82px; height: 82px; border-radius: 18px; box-shadow: 0 0 22px rgba(99, 102, 241, 0.5); border: 2px solid rgba(165, 180, 252, 0.4); margin-bottom: 8px;">
                <div style="font-size: 1.4rem; font-weight: 800; background: linear-gradient(90deg, #A5B4FC, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.01em;">TECH ANALYZER AI</div>
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Enterprise Intelligence Suite</div>
                <hr style="border: none; height: 1px; background: rgba(51, 65, 85, 0.6); margin: 10px 0;">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="top-brand-header" style="text-align: center; padding: 5px 0 15px 0;">
                <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">⚡ TECH ANALYZER AI</div>
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 600; text-transform: uppercase; margin-bottom: 12px;">Enterprise Intelligence Suite</div>
                <hr style="border: none; height: 1px; background: rgba(51, 65, 85, 0.6); margin: 10px 0;">
            </div>
        """, unsafe_allow_html=True)

# Define Navigation Pages
churn_page = st.Page(
    "churn_analyser.py",
    title="Churn Risk Analyser",
    icon="⚡",
    default=True
)

salary_page = st.Page(
    "pages/2_💎_Tech_Salary_Analyser.py",
    title="Tech Salary Analyser",
    icon="💎"
)

# Render Custom Sidebar Navigation Group
pg = st.navigation({
    "MODULES": [churn_page, salary_page]
})

pg.run()