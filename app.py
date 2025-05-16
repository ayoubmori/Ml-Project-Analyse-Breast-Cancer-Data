import streamlit as st
import pandas as pd
from data_loader import load_data, get_features_and_target
from tabs.overview import data_overview
from tabs.modeling import modeling
from tabs.data_vis import visualization 
from tabs.Preprocessing import Preprocessing

st.set_page_config("Model Project",layout="wide")

st.title("🧬 Breast Cancer Data Visualization")
# if st.toggle("Dark Mode", value=True) is False:
#       st._config.set_option(f'theme.base', "light")
# else:
#       st._config.set_option(f'theme.base', "dark")
# if st.button("Refresh"):
#       st.rerun()


def mode_changer():
    # Initialize session state: start with toggle == False
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    # Compute a dynamic label showing the *current* theme
    label = (
        "Light Mode 🌞"
        if st.session_state.dark_mode
        else "Dark Mode 🌙"
    )

    # Render the caption and the toggle
    st.caption("Theme")
    new_mode = st.toggle(label, value=st.session_state.dark_mode)

    # If the user flipped it, update theme & rerun
    if new_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = new_mode
        # **invert**: when toggle == True → light, when False → dark
        theme = "light" if new_mode else "dark"
        st._config.set_option("theme.base", theme)
        st.rerun()

# Place it in a narrow column    
_, theme_col = st.columns([10, 1.2])
with theme_col:
    mode_changer()
    
overview, vis, preprocess, model_part = st.tabs(["Data Overview","Visualization", "Preprocessing", "Modeling"])
# Load data
df = load_data("Breast Cancer Wisconsin.csv")
X, y = get_features_and_target(df)
# Sidebar navigation
# page = st.sidebar.selectbox("Navigation", ["Data Overview", "Preprocessing", "Correlation", "Modeling"])

with overview:
    data_overview(df,X,y)

with vis :
    visualization(df,X,y)
    
with preprocess :
    Preprocessing(X)


with model_part :
    modeling(X,y)


    
