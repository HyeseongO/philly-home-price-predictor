import streamlit as st

st.set_page_config(page_title="Philly Home Price Predictor")

st.title("Philadelphia Home Price Predictor")
st.write("A data-driven tool that helps real estate agents and home buyers estimate a home's market value in Philadelphia.")

st.subheader("Who is this for?")
st.write("- Real estate agents pricing a listing\n- Home buyers checking whether a price is fair")

st.subheader("How to use")
st.write("Enter the home's features on the Predict page to get an estimated price.")