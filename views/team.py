from pathlib import Path

import streamlit as st

PHOTO_DIR = Path(__file__).parent.parent / "assets" / "team"

TEAM = [
    {
        "name": "Owen Nguyen",
        "photo": "owen.jpg",
        "major": "Economics and Business Analytics",
        "grad": "Class of 2027",
        "email": "anhnguyen_2027@depauw.edu",
        "links": {
            "LinkedIn": "https://www.linkedin.com/in/owen-nguyen-work/",
            "Portfolio": "https://owen-nguyen-portfolio.vercel.app/",
        },
    },
    {
        "name": "Hyeseong Oh",
        "photo": "hyeseong.jpg",
        "major": "Computer Science and Business Analytics",
        "grad": "Class of 2027",
        "email": "hyeseongoh_2027@depauw.edu",
        "links": {
            "LinkedIn": "https://www.linkedin.com/in/hyeseong-oh-722a712ba",
            "GitHub": "https://github.com/HyeseongO"
        },
    },
    {
        "name": "Ian Porter",
        "photo": "ian.jpg",
        "major": "Business Analytics",
        "minor": "Computer Science, Data Science, and Finance",
        "grad": "Class of 2028",
        "email": "ianporter_2028@depauw.edu",
        "links": {
            "LinkedIn": "https://www.linkedin.com/in/ian-porter-312099337/",
        },
    },
]

st.title("Meet the Team")
st.write("BUSA 310 Business Analytics III · DePauw University")

for col, member in zip(st.columns(len(TEAM), gap="large"), TEAM):
    with col:
        st.image(PHOTO_DIR / member["photo"], width="stretch")
        st.subheader(member["name"])
        st.write(member["major"])
        if "minor" in member:
            st.caption(f"Minors: {member['minor']}")
        st.caption(member["grad"])
        links = {"Email": f"mailto:{member['email']}", **member["links"]}
        st.markdown(" · ".join(f"[{label}]({url})" for label, url in links.items()))