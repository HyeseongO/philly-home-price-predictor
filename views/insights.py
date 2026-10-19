from pathlib import Path

import pandas as pd
import streamlit as st

SALES_PATH = Path(__file__).parent.parent / "data" / "philly_sf_sales_clean.csv"
EXPLANATION_NOTEBOOK = (
    "https://colab.research.google.com/github/HyeseongO/philly-home-price-predictor/blob/main/notebook/06_explanation_model.ipynb"
)

# Explanation model (notebook 06): % change in price with everything else held the same; significant results only
HOME_FACTORS = [
    ("Living area 10% larger", "+7.7%"),
    ("One more bathroom", "+9.5%"),
    ("One more garage space", "+10.1%"),
    ("Central air (vs. none)", "+22.9%"),
    ("Interior rehabbed or above average (vs. average)", "+26%"),
    ("Interior below average (vs. average)", "-13.6%"),
    ("Exterior below average (vs. average)", "-30.1%"),
    ("One more bedroom, same living area", "-2.6%"),
]
AREA_FACTORS = [
    ("Median household income $10,000 higher", "+7.1%"),
    ("School reading score 10 points higher", "+7.2%"),
    ("One more violent crime per 1,000 residents", "-4.2%"),
    ("Sold in 2026 (vs. 2024)", "+7.7%"),
]


@st.cache_data
def zip_prices():
    """Median sale price by ZIP, using the same sales as the similar sales on the Predict page."""
    columns = ["zip_code", "sale_price", "total_livable_area", "bundle_size", "flag_vacant", "price_to_assessed"]
    sales = pd.read_csv(SALES_PATH, usecols=columns, dtype={"zip_code": str})
    sales = sales[(sales["bundle_size"] == 1) & ~sales["flag_vacant"] & (sales["price_to_assessed"] >= 0.5) & (sales["total_livable_area"] > 0)]
    sales["per_sqft"] = sales["sale_price"] / sales["total_livable_area"]
    by_zip = sales.groupby("zip_code").agg(median_price=("sale_price", "median"), per_sqft=("per_sqft", "median"), sales=("sale_price", "size"))
    return by_zip.sort_values("median_price", ascending=False), len(sales)


def zip_table(rows):
    return pd.DataFrame({
        "ZIP": rows.index,
        "Median price": rows["median_price"].map("${:,.0f}".format),
        "Per sq ft": rows["per_sqft"].map("${:,.0f}".format),
    })


def factor_table(factors):
    return pd.DataFrame(factors, columns=["Factor", "Change in price"])


st.title("Market Insights")
st.write("What our data says about home prices in Philadelphia, for agents setting a list price and buyers judging an offer.")

st.header("Which neighborhoods are most expensive?")
prices, sale_count = zip_prices()
top, bottom = prices.iloc[0], prices.iloc[-1]
st.write(
    f"The median home in ZIP {top.name} sold for \\${top['median_price']:,.0f}, about "
    f"{top['median_price'] / bottom['median_price']:.0f} times the \\${bottom['median_price']:,.0f} median in ZIP {bottom.name}. "
    "Location is the biggest driver of price in the city."
)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Most expensive")
    st.dataframe(zip_table(prices.head(5)), hide_index=True, width="stretch")
with col2:
    st.subheader("Least expensive")
    st.dataframe(zip_table(prices.tail(5).iloc[::-1]), hide_index=True, width="stretch")
st.caption(
    f"Median sale price of {sale_count:,} single-family home sales from January 2024 to September 2026, "
    f"City of Philadelphia (OPA) records. Every ZIP code has at least {prices['sales'].min():,} sales."
)

st.header("What changes the price?")
st.write(
    "How much the price changes with one feature when everything else about the home and its year of sale stays the same, "
    "from our explanation model of 27,998 sales."
)
st.subheader("The home")
st.dataframe(factor_table(HOME_FACTORS), hide_index=True, width="stretch")
st.subheader("The neighborhood and year")
st.dataframe(factor_table(AREA_FACTORS), hide_index=True, width="stretch")
st.caption(
    "Only statistically significant results are shown. Year built had no clear effect once the other features were included. "
    "These are associations, not proof of cause, and the price estimate on the Predict page comes from a separate "
    f"Random Forest model. [See the explanation model]({EXPLANATION_NOTEBOOK})."
)

st.header("What this means for agents")
st.markdown(
    """
- **Location comes first.** Compare a home with recent sales in the same ZIP code, not with the city as a whole.
- **Condition is worth a lot.** A rehabbed or above average interior sells for about a quarter more than an average one, and a below average exterior for about 30% less.
- **Space beats room count.** Living area, bathrooms, a garage, and central air add value, but splitting the same space into more bedrooms does not.
"""
)
