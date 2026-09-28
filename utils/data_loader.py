import pandas as pd
import streamlit as st

@st.cache_data
def load_price_data():
    """
    Loads and cleans crop price benchmark dataset.
    """
    try:
        df = pd.read_csv("data/crop_prices.csv")
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return df
    except Exception:
        data = {
            "crop_name": ["Talong", "Broccoli", "Pineapple", "Kalabasa", "Kamatis", "Siling Labuyo"],
            "min_price": [50.0, 120.0, 45.0, 30.0, 40.0, 150.0],
            "avg_price": [80.0, 150.0, 60.0, 45.0, 65.0, 200.0],
            "max_price": [110.0, 190.0, 80.0, 60.0, 90.0, 280.0]
        }
        return pd.DataFrame(data)


def get_crop_valuation(crop_name, grade, region="NCR (National Capital Region)"):
    """
    Calculates adjusted crop pricing across all 17 Philippine regions.
    """
    df = load_price_data()

    possible_cols = ["crop_name", "crop", "commodity", "item", "produce"]
    crop_col = next((col for col in possible_cols if col in df.columns), df.columns[0])

    matched = df[df[crop_col].astype(str).str.lower() == crop_name.lower()]

    if matched.empty:
        base_min, base_avg, base_max = 40.0, 60.0, 80.0
    else:
        row = matched.iloc[0]
        base_min = float(row.get("min_price", row.get("min", 40.0)))
        base_avg = float(row.get("avg_price", row.get("price", 60.0)))
        base_max = float(row.get("max_price", row.get("max", 80.0)))

    # Benchmark factors across all 17 administrative regions
    region_multipliers = {
        "NCR (National Capital Region)": 1.12,
        "CAR (Cordillera Administrative Region)": 0.90,
        "Region I (Ilocos Region)": 0.95,
        "Region II (Cagayan Valley)": 0.92,
        "Region III (Central Luzon)": 0.98,
        "Region IV-A (CALABARZON)": 1.02,
        "MIMAROPA Region": 0.96,
        "Region V (Bicol Region)": 0.94,
        "Region VI (Western Visayas)": 0.97,
        "Region VII (Central Visayas)": 1.05,
        "Region VIII (Eastern Visayas)": 0.95,
        "Region IX (Zamboanga Peninsula)": 0.93,
        "Region X (Northern Mindanao)": 0.94,
        "Region XI (Davao Region)": 0.91,
        "Region XII (SOCCSKSARGEN)": 0.89,
        "Region XIII (Caraga)": 0.90,
        "BARMM (Bangsamoro Region)": 0.88
    }
    multiplier = region_multipliers.get(region, 1.00)

    grade_multipliers = {
        "Grade A": 1.15,
        "Grade B": 1.00,
        "Grade C": 0.85
    }
    g_mult = grade_multipliers.get(grade, 1.00)

    final_suggested = round(base_avg * multiplier * g_mult, 2)
    final_min = round(base_min * multiplier, 2)
    final_max = round(base_max * multiplier, 2)

    return {
        "min_price": final_min,
        "suggested_price": final_suggested,
        "max_price": final_max
    }