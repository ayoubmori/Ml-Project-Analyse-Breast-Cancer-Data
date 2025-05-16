import streamlit as st
import pandas as pd
from data_loader import load_data, get_features_and_target
from tabs.overview import data_overview
from tabs.modeling import modeling
from tabs.data_vis import visualization 
from tabs.Preprocessing import Preprocessing

st.set_page_config("Model Project",layout="wide")

st.title("🧬 Breast Cancer Data Visualization")
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


    
