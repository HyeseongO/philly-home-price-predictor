import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent.parent / "model"
SALES_PATH = Path(__file__).parent.parent / "data" / "philly_sf_sales_clean.csv"
NEIGHBORHOOD_PATH = Path(__file__).parent.parent / "data" / "zip_features.csv"
SALE_YEAR = "2026"  # sale year is a category in the model (year fixed effects)
ACCENT, RANGE_COLOR, LABEL_COLOR = "#3D6CB5", "#7C9CC9", "#8A96A6"  # readable on the light and dark themes
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
NEIGHBORHOOD_CARDS = [
    ("Median income", "zip_income", "${:,.0f}", "Median household income, U.S. Census ACS 2020-2024."),
    ("Homeowners", "zip_pct_owner", "{:.0%}", "Share of occupied homes that are owner occupied, U.S. Census ACS 2020-2024."),
    ("Bachelor's degree", "zip_pct_bachelors", "{:.0%}", "Share of adults 25 and older with a bachelor's degree or higher, U.S. Census ACS 2020-2024."),
    ("Violent crime", "zip_violent_per_1k", "{:.1f}", "Violent crimes per 1,000 residents per year, Philadelphia Police incidents 2024-2025."),
    ("Property crime", "zip_property_per_1k", "{:.1f}", "Property crimes per 1,000 residents per year, Philadelphia Police incidents 2024-2025."),
    ("School reading", "zip_school_score", "{:.0f}%", "Average share of grade 3-8 students proficient or advanced on the PSSA English test at the area's elementary schools, School District of Philadelphia SPREE 2024-25."),
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")["model"]


@st.cache_data
def load_range_table():
    return json.loads((MODEL_DIR / "price_range.json").read_text())["bands"]


def price_range(price):
    """80% range: the 10th to 90th percentile of actual/predicted for test homes with a similar estimate."""
    for band in load_range_table():
        if band["band_high"] is None or price < band["band_high"]:
            return price * band["low_mult"], price * band["high_mult"]


def range_chart(low, price, high):
    """Likely range as a bar, the estimate as a dot, and the two ends labeled."""
    pad = (high - low) * 0.06
    x = alt.X("x:Q", scale=alt.Scale(domain=[low - pad, high + pad]), axis=None)
    bar = alt.Chart(pd.DataFrame({"x": [low], "x2": [high]})).mark_bar(height=14, cornerRadius=7, color=RANGE_COLOR, opacity=0.45).encode(x=x, x2="x2:Q")
    dot = alt.Chart(pd.DataFrame({"x": [price]})).mark_point(filled=True, size=240, opacity=1, color=ACCENT).encode(x=x)
    ends = [
        alt.Chart(pd.DataFrame({"x": [value], "label": [f"${value:,.0f}"]}))
        .mark_text(align=align, dy=24, fontSize=13, color=LABEL_COLOR).encode(x=x, text="label:N")
        for value, align in [(low, "left"), (high, "right")]
    ]
    return alt.layer(bar, dot, *ends).properties(height=60).configure_view(stroke=None)


@st.cache_data
def load_options():
    return json.loads((MODEL_DIR / "input_options.json").read_text())


@st.cache_data
def load_sales():
    """Single-parcel home sales, without vacant lots or sales far below the city's assessed value."""
    columns = ["location", "zip_code", "sale_date", "sale_price", "total_livable_area", "number_of_bedrooms",
               "number_of_bathrooms", "year_built", "bundle_size", "flag_vacant", "price_to_assessed"]
    sales = pd.read_csv(SALES_PATH, usecols=columns, dtype={"zip_code": str}, parse_dates=["sale_date"])
    keep = (sales["bundle_size"] == 1) & ~sales["flag_vacant"] & (sales["price_to_assessed"] >= 0.5) & (sales["total_livable_area"] > 0)
    return sales[keep]


@st.cache_data
def load_neighborhoods():
    return pd.read_csv(NEIGHBORHOOD_PATH, dtype={"zip_code": str}).set_index("zip_code")


def similar_sales(zip_code, living_area, bedrooms, count=5):
    """Most recent sales in the same ZIP with similar living area (and bedrooms when enough exist)."""
    in_zip = load_sales().query("zip_code == @zip_code")
    for spread, bedroom_gap in [(0.25, 1), (0.25, None), (0.5, None)]:
        match = in_zip[(in_zip["total_livable_area"] - living_area).abs() <= spread * living_area]
        if bedroom_gap is not None:
            match = match[(match["number_of_bedrooms"] - bedrooms).abs() <= bedroom_gap]
        if len(match) >= 3:
            break
    return match.sort_values("sale_date", ascending=False).head(count), spread


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
            garage_spaces = st.selectbox("Garage spaces", [NOT_SURE, 0, 1, 2, 3])
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
        "garage_spaces": np.nan if garage_spaces == NOT_SURE else garage_spaces,
        "sale_year": SALE_YEAR,
        "zip_code": zip_code,
        "interior_condition": CONDITIONS[interior],
        "exterior_condition": CONDITIONS[exterior],
        "central_air": CENTRAL_AIR[central_air],
        "quality_grade": np.nan if quality_grade == NOT_SURE else quality_grade,
    }])
    X[CATEGORICAL] = X[CATEGORICAL].astype(object)
    price = model.predict(X)[0]
    low, high = price_range(price)
    with st.container(border=True):
        st.metric("Estimated price", f"${price:,.0f}")
        st.altair_chart(range_chart(low, price, high), width="stretch")
        st.caption(
            f"Likely range: \\${low:,.0f} to \\${high:,.0f}. About 8 in 10 test homes with a similar estimate "
            "sold within this range. Use it as a starting point alongside comparable sales, not as an appraisal."
        )

    by_year_tab, sales_tab, zip_tab = st.tabs(["Price by sale year", "Similar recent sales", f"About ZIP {zip_code}"])

    with by_year_tab:
        years = ["2024", "2025", "2026"]
        by_year = model.predict(pd.concat([X.assign(sale_year=year) for year in years], ignore_index=True))
        for col, year, value, previous in zip(st.columns(len(years)), years, by_year, [None, *by_year[:-1]]):
            delta = None if previous is None else f"{'+' if value >= previous else '-'}${abs(value - previous):,.0f}"
            col.metric(year, f"${value:,.0f}", delta=delta)
        st.caption(
            "How the estimate changes with the sale year, using sales from January 2024 to September 2026. "
            "This is not a forecast of future prices."
        )

    with sales_tab:
        comps, spread = similar_sales(zip_code, living_area, bedrooms)
        if comps.empty:
            st.write(f"No recent sales of a similar size were found in ZIP {zip_code}.")
        else:
            st.write(
                f"The {len(comps)} most recent sales in ZIP {zip_code} with "
                f"{living_area * (1 - spread):,.0f} to {living_area * (1 + spread):,.0f} sq ft of living area."
            )
            st.dataframe(
                pd.DataFrame({
                    "Sold": comps["sale_date"].dt.strftime("%b %Y"),
                    "Street": comps["location"].str.replace(r"^\S+\s+", "", regex=True),
                    "Sale price": comps["sale_price"].map("${:,.0f}".format),
                    "Sq ft": comps["total_livable_area"].map("{:,.0f}".format),
                    "Beds": comps["number_of_bedrooms"].astype("Int64"),
                    "Baths": comps["number_of_bathrooms"].astype("Int64"),
                    "Built": comps["year_built"].astype("Int64"),
                }),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Recorded sales from City of Philadelphia (OPA) records, January 2024 to September 2026. "
                "House numbers are hidden. Sales far below the city's assessed value are left out because "
                "they may not be regular market sales."
            )

    with zip_tab:
        neighborhoods = load_neighborhoods()
        if zip_code in neighborhoods.index:
            area, city = neighborhoods.loc[zip_code], neighborhoods.median(numeric_only=True)
            for start in range(0, len(NEIGHBORHOOD_CARDS), 3):
                for col, (label, key, fmt, source) in zip(st.columns(3), NEIGHBORHOOD_CARDS[start:start + 3]):
                    col.metric(label, fmt.format(area[key]), help=f"{source} Median across our 46 ZIP codes: {fmt.format(city[key])}.")
            st.caption(
                "Neighborhood context for the selected ZIP code. Crime rates are per 1,000 residents, "
                "so busy areas with few residents, such as Center City, can show high rates."
            )
        else:
            st.write(f"No neighborhood data is available for ZIP {zip_code}.")

with st.expander("About this tool"):
    st.write(
        "The tool predicts a home's sale price using a machine learning model trained on "
        "real Philadelphia single-family home sales (2024 onward) from the City of Philadelphia "
        "Office of Property Assessment (OPA)."
    )

st.divider()
st.caption("Built by Owen Nguyen, Hyeseong Oh, and Ian Porter for BUSA 310 at DePauw University.")
st.page_link("views/team.py", label="Meet the team")
