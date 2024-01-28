import streamlit as st

def app(df):


    # st.image(
    #         "./data/images/voiture_ciel.png",
    #         width=2000, use_column_width = "auto"
    #         )

    

    from PIL import Image
    background = Image.open("./data/images/voiture_ciel.png")
    st.image(background,
              width=10000, use_column_width = "auto")

