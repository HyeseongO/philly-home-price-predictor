import streamlit as st

st.set_page_config(page_title="Philly Home Price Predictor")

pages = [
    st.Page("views/predict.py", title="Predict", default=True),
    st.Page("views/team.py", title="Team"),
]
st.navigation(pages, position="top").run()
