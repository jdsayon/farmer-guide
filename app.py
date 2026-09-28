import streamlit as st
import pandas as pd
import requests
from utils.pricing_engine import analyze_crop_image
from utils.data_loader import get_crop_valuation, load_price_data

st.set_page_config(
    page_title="AgriPrice GenAI Philippines",
    page_icon="🌾",
    layout="wide"
)

# Application Header
st.title("🌾 AgriPrice GenAI: Philippine Market Pricing Engine")
st.caption("AI-powered produce grading & real-time Department of Agriculture (DA) market valuation.")

# All 17 Philippine Administrative Regions
PHILIPPINE_REGIONS = [
    "NCR (National Capital Region)",
    "CAR (Cordillera Administrative Region)",
    "Region I (Ilocos Region)",
    "Region II (Cagayan Valley)",
    "Region III (Central Luzon)",
    "Region IV-A (CALABARZON)",
    "MIMAROPA Region",
    "Region V (Bicol Region)",
    "Region VI (Western Visayas)",
    "Region VII (Central Visayas)",
    "Region VIII (Eastern Visayas)",
    "Region IX (Zamboanga Peninsula)",
    "Region X (Northern Mindanao)",
    "Region XI (Davao Region)",
    "Region XII (SOCCSKSARGEN)",
    "Region XIII (Caraga)",
    "BARMM (Bangsamoro Region)"
]


def ask_huggingface_llm(user_input, region):
    """
    Sends live prompt directly to Hugging Face API using HF_TOKEN.
    Uses universal auto-routing to dynamically select an available provider.
    """
    hf_token = st.secrets.get("HF_TOKEN", "").strip()

    if not hf_token or hf_token == "hf_YourActualHuggingFaceTokenHere":
        return "⚠️ Error: Missing or default HF_TOKEN in `.streamlit/secrets.toml`. Please add a valid Hugging Face Access Token."

    # Universal Hugging Face router endpoint (removed /hf-inference/)
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    # Open-ended LLM system prompt
    system_prompt = (
        "You are an intelligent, versatile AI assistant. "
        "Answer any general question accurately (science, math, history, coding, writing, general advice). "
        f"If the user asks about crops, agriculture, or market prices in the Philippines, provide accurate regional insights for {region}."
    )

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ Hugging Face API Error ({response.status_code}): {response.text}"

    except requests.exceptions.RequestException as err:
        return f"⚠️ Connection Error: Failed to reach Hugging Face API. Details: {err}"


# --- SIDEBAR CONTROLS ---
st.sidebar.header("📍 Market Settings & Filters")
selected_region = st.sidebar.selectbox(
    "Select Target Market Region",
    PHILIPPINE_REGIONS
)

st.sidebar.markdown("---")
st.sidebar.header("💬 General AI Assistant")
st.sidebar.caption("Powered live by Hugging Face API")
user_query = st.sidebar.text_input(
    "Ask anything:", 
    placeholder="e.g., Explain photosynthesis, write code, or ask crop prices"
)

if user_query:
    with st.sidebar.spinner("Connecting to Hugging Face LLM..."):
        ai_reply = ask_huggingface_llm(user_query, selected_region)
        st.sidebar.info(f"🤖 **AI Output:**\n\n{ai_reply}")

# --- MAIN CONTROLS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Produce Image")
    uploaded_file = st.file_uploader("Upload photo of crop", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)

with col2:
    st.subheader("2. AI Analysis & Market Valuation")

    if uploaded_file is not None:
        if st.button("Analyze Produce & Calculate Price", type="primary"):
            with st.spinner("Analyzing image via Hugging Face Vision API..."):
                analysis = analyze_crop_image(uploaded_file)

            crop_name = analysis.get("crop_name", "Unknown")
            grade = analysis.get("grade", "Grade A")
            score = analysis.get("quality_score", 80)
            desc = analysis.get("description", "")

            # Persist results in session state across UI reruns
            st.session_state["analysis_result"] = {
                "crop_name": crop_name,
                "grade": grade,
                "score": score,
                "desc": desc
            }

        # Render stored valuation details
        if "analysis_result" in st.session_state:
            res = st.session_state["analysis_result"]
            valuation = get_crop_valuation(res["crop_name"], res["grade"], selected_region)

            st.success("Analysis Complete!")

            # Metric Display
            m1, m2, m3 = st.columns(3)
            m1.metric("Identified Crop", res["crop_name"])
            m2.metric("Assessed Grade", res["grade"])
            m3.metric("Suggested Price", f"₱{valuation['suggested_price']:.2f} / kg")

            st.info(f"**Quality Score:** {res['score']}% | {res['desc']}")

            # Price Bar Chart
            st.markdown("---")
            st.subheader(f"📊 Price Range Distribution ({selected_region})")

            chart_df = pd.DataFrame({
                "Pricing Tier": ["Minimum Price", "Suggested Valuation", "Maximum Price"],
                "Price (₱/kg)": [valuation["min_price"], valuation["suggested_price"], valuation["max_price"]]
            }).set_index("Pricing Tier")

            st.bar_chart(chart_df)
    else:
        if "analysis_result" in st.session_state:
            del st.session_state["analysis_result"]
        st.info("Please upload an image to run the valuation engine.")

# --- DATASET TABLE ---
st.markdown("---")
with st.expander("📋 View Department of Agriculture (DA) Benchmark Price Dataset"):
    df_prices = load_price_data()
    st.dataframe(df_prices, use_container_width=True)