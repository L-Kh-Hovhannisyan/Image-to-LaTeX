import requests
from PIL import Image

import streamlit as st


st.set_page_config(page_title="Image To Latex Converter")


st.title("Նկարի հիման վրա համապատասխան LaTeX շարահյուսության գեներացում")


uploaded_file = st.file_uploader(
    "Ներբեռնեք բանաձևը (նկարը) և ստացեք LaTeX շարահյուսությունը",
    type=["png", "jpg"],
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)


if st.button("Գեներացնել"):
    if uploaded_file is not None and image is not None:
        files = {"file": uploaded_file.getvalue()}
        with st.spinner("Բեռնում..."):
            response = requests.post("http://0.0.0.0:8000/predict/", files=files)
        latex_code = response.json()["data"]["pred"]
        st.code(latex_code)
        st.markdown(f"${latex_code}$")
    else:
        st.error("Ներբեռնեք բանաձևը (նկարը)")
