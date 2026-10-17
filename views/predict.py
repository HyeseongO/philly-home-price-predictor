import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent.parent / "model"
SALE_YEAR = 2026
CATEGORICAL = ["zip_code", "interior_condition", "exterior_condition", "central_air", "quality_grade"]
NOT_SURE = "Not sure"

CONDITIONS = {
    NOT_SURE: np.nan,
    "Code 1": "1",
    "Rehabbed": "2",
    "Above average": "3",
    "Average": "4",
    "Below average": "5",
    "Code 6": "6",
    "Code 7": "7",
}
CONDITION_HELP = (
    "Codes 1, 6, and 7 are City of Philadelphia (OPA) condition codes "
    "whose official labels we are still confirming. Choose 'Not sure' if unknown."
)
CENTRAL_AIR = {NOT_SURE: np.nan, "Yes": "Y", "No": "N"}
GRADES = [NOT_SURE, "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")["model"]


@st.cache_data
def load_options():
    return json.loads((MODEL_DIR / "input_options.json").read_text())


model = load_model()
ZIP_CODES = load_options()["zip_code"]

st.title("Philadelphia Home Price Predictor")
st.write(
    "Estimate a home's market value in Philadelphia from its features. "
    "Built for **real estate agents** pricing a listing and **home buyers** checking whether a price is fair."
)

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        zip_code = st.selectbox("ZIP code", ZIP_CODES, index=ZIP_CODES.index("19143"))
        living_area = st.number_input("Living area (sq ft)", min_value=400, max_value=6000, value=1200, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=1)
    with col2:
        year_built = st.number_input("Year built", min_value=1700, max_value=2026, value=1925)
        stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)
        interior = st.selectbox("Interior condition", list(CONDITIONS), index=4, help=CONDITION_HELP)
        exterior = st.selectbox("Exterior condition", list(CONDITIONS), index=4, help=CONDITION_HELP)

    with st.expander("More details (optional)"):
        col3, col4 = st.columns(2)
        with col3:
            garage_spaces = st.number_input("Garage spaces", min_value=0, max_value=3, value=None, placeholder=NOT_SURE)
            central_air = st.selectbox("Central air", list(CENTRAL_AIR))
        with col4:
            quality_grade = st.selectbox("Quality grade (city assessment)", GRADES)

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    X = pd.DataFrame([{
        "total_livable_area": living_area,
        "number_of_bedrooms": bedrooms,
        "number_of_bathrooms": bathrooms,
        "year_built": year_built,
        "number_stories": stories,
        "garage_spaces": np.nan if garage_spaces is None else garage_spaces,
        "sale_year": SALE_YEAR,
        "zip_code": zip_code,
        "interior_condition": CONDITIONS[interior],
        "exterior_condition": CONDITIONS[exterior],
        "central_air": CENTRAL_AIR[central_air],
        "quality_grade": np.nan if quality_grade == NOT_SURE else quality_grade,
    }])
    X[CATEGORICAL] = X[CATEGORICAL].astype(object)
    price = model.predict(X)[0]
    st.success(f"Estimated price: ${price:,.0f}")

with st.expander("About this tool"):
    st.write(
        "The tool predicts a home's sale price using a machine learning model trained on "
        "real Philadelphia single-family home sales (2024 onward) from the City of Philadelphia "
        "Office of Property Assessment (OPA)."
    )

st.divider()
st.caption("Built by Owen Nguyen, Hyeseong Oh, and Ian Porter for BUSA 310 at DePauw University.")
st.page_link("views/team.py", label="Meet the team")
