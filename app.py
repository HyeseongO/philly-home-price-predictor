import streamlit as st
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

st.set_page_config(page_title="Philly Home Price Predictor", page_icon="🏠")

ZIP_CODES = [
    "19102", "19103", "19104", "19106", "19107", "19111", "19114", "19115", "19116", "19118",
    "19119", "19120", "19121", "19122", "19123", "19124", "19125", "19126", "19127", "19128",
    "19129", "19130", "19131", "19132", "19133", "19134", "19135", "19136", "19137", "19138",
    "19139", "19140", "19141", "19142", "19143", "19144", "19145", "19146", "19147", "19148",
    "19149", "19150", "19151", "19152", "19153", "19154",
]
HOME_TYPES = ["Row", "Twin", "Detached"]
CONDITIONS = {
    "New construction": 1,
    "Rehabbed": 2,
    "Above average": 3,
    "Average": 4,
    "Below average": 5,
    "Poor / vacant": 6,
}

st.title("Philadelphia Home Price Predictor")
st.write(
    "Estimate a home's market value in Philadelphia from its features. "
    "Built for **real estate agents** pricing a listing and **home buyers** checking whether a price is fair."
)

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        zip_code = st.selectbox("ZIP code", ZIP_CODES, index=ZIP_CODES.index("19148"))
        living_area = st.number_input("Living area (sq ft)", min_value=400, max_value=6000, value=1200, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=1)
    with col2:
        home_type = st.selectbox("Home type", HOME_TYPES)
        year_built = st.number_input("Year built", min_value=1700, max_value=2026, value=1925)
        interior = st.selectbox("Interior condition", list(CONDITIONS), index=3)

    with st.expander("More details (optional)"):
        col3, col4 = st.columns(2)
        with col3:
            stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)
            garage_spaces = st.number_input("Garage spaces", min_value=0, max_value=3, value=0)
        with col4:
            exterior = st.selectbox("Exterior condition", list(CONDITIONS), index=3)
            central_air = st.checkbox("Central air")

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    model = load_model()
    if model is None:
        price = living_area * 180 + bathrooms * 40000 + garage_spaces * 30000 + (40000 if central_air else 0)
        st.success(f"Estimated price: ${price:,.0f}")
        st.caption("This is a placeholder estimate. The real model will be connected soon.")
    else:
        X = pd.DataFrame([{
            "zip_code": zip_code,
            "total_livable_area": living_area,
            "number_of_bedrooms": bedrooms,
            "number_of_bathrooms": bathrooms,
            "year_built": year_built,
            "home_type": home_type,
            "interior_condition": CONDITIONS[interior],
            "exterior_condition": CONDITIONS[exterior],
            "number_stories": stories,
            "garage_spaces": garage_spaces,
            "central_air": "Y" if central_air else "N",
        }])
        price = model.predict(X)[0]
        st.success(f"Estimated price: ${price:,.0f}")

with st.expander("About this tool"):
    st.write(
        "This tool predicts a home's sale price using a machine learning model trained on "
        "real Philadelphia single-family home sales (2024 onward) from the City of Philadelphia "
        "Office of Property Assessment (OPA)."
    )
