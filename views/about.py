import streamlit as st

REPO = "https://github.com/HyeseongO/philly-home-price-predictor"
COLAB = "https://colab.research.google.com/github/HyeseongO/philly-home-price-predictor/blob/main/notebook/"
OPA_DATA = "https://opendataphilly.org/datasets/philadelphia-properties-and-assessment-history"

INTERVIEWEES = ["two previous home buyers", "a potential first-time buyer", "a previous home seller", "a realtor"]
INTERVIEW_CHANGES = [
    "All five asked for a price range, not just one number, so every estimate now comes with a likely range.",
    "Three wanted to know how reliable the estimate is, such as how often it misses and why it might be off, so this page shows its accuracy on homes it never saw and its limitations.",
    "Three asked to see recent sales of similar homes nearby, so each estimate now lists the most recent similar sales in the same ZIP code.",
    "Three mentioned neighborhood factors such as crime, schools, and the surrounding area, so each estimate now shows income, homeownership, education, crime, and school data for the ZIP code.",
]

st.title("About This Tool")
st.write("How the estimate is made, what data it learned from, and how far you can trust it.")

st.header("How accurate is it?")
st.write("We tested the model on 5,915 Philadelphia home sales that it never saw during training.")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Typical error", "$44,432", help="Median absolute error: half of the test homes were off by less than this.")
col2.metric("Within 20% of price", "54%", help="Share of test homes whose estimate was within 20% of the actual sale price.")
col3.metric("Average error (MAE)", "$77,211", help="Mean absolute error across all test homes.")
col4.metric("R²", "0.78", help="Share of the variation in sale prices the model explains (1.0 is perfect).")
st.caption("Use the estimate as a starting point alongside comparable sales, not as a formal appraisal.")

st.header("The data")
st.markdown(
    f"""
- **Source:** City of Philadelphia, Office of Property Assessment (OPA), via [OpenDataPhilly]({OPA_DATA})
- **Homes:** properties the city classifies as single family
- **Sale dates:** January 2024 to September 2026
- **Sales used for the model:** 29,572
- **Cleaning:** kept sales above $25,000 and removed bundled sales (several parcels sold at one price)
- **Neighborhood information:** U.S. Census ACS 2020-2024, Philadelphia Police incidents 2024-2025, and School District of Philadelphia test scores (2024-25), by ZIP code
"""
)

st.header("The model")
st.write(
    "The website uses a Random Forest model. We compared it with linear regression, ridge regression, "
    "and gradient boosting on the same training and test homes."
)
st.write(
    "It uses 12 inputs: living area, bedrooms, bathrooms, year built, stories, garage spaces, ZIP code, "
    "interior condition, exterior condition, central air, quality grade, and sale year."
)
st.write(
    "ZIP code and sale year enter the model as categories (ZIP and year fixed effects), so each ZIP and each year "
    "gets its own price level. The likely range shown with each estimate covers the middle 80% of how test homes "
    "with a similar estimate actually sold."
)

st.header("Limitations")
st.markdown(
    """
- Home features come from the city's current (2026) records, which may differ from the home's condition when it sold.
- Estimates assume the home sells in 2026.
- Condition codes 1, 6, and 7 do not have confirmed labels yet.
- Inputs left as "Not sure" are filled with typical values from the training data.
- Neighborhood information is shown for context. The price estimate accounts for location through the ZIP code itself.
"""
)

if INTERVIEWEES:
    st.header("What we heard from users")
    st.write("We interviewed five people: " + ", ".join(INTERVIEWEES) + ".")
    if INTERVIEW_CHANGES:
        st.write("What we changed because of those conversations:")
        st.markdown("\n".join(f"{i}. {change}" for i, change in enumerate(INTERVIEW_CHANGES, start=1)))

st.header("Reproduce our work")
st.markdown(
    f"""
- [GitHub repository]({REPO})
- Notebooks in Google Colab: [data cleaning]({COLAB}02_data_cleaning.ipynb) · [exploratory analysis]({COLAB}Group_Project.ipynb) · [modeling]({COLAB}Group_Project_Modeling.ipynb) · [explanation model]({COLAB}06_explanation_model.ipynb)
- [Cleaned dataset (CSV)]({REPO}/blob/main/data/philly_sf_sales_clean.csv)
"""
)