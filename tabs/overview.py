import streamlit as st
import pandas as pd
import io

def get_features_and_target(df):
    y = df['diagnosis']
    x = df.drop(['id','diagnosis'], axis=1)
    return x, y


def data_overview(df,X,y):
    st.title("Data Overview")
    
    st.subheader("First 5 Rows :")
    st.dataframe(df[:5].style.format(precision=2))
    
    df_info,df_desc = st.columns([1.5,4])
    
    with df_info:
        # st.write("Dataset Shape:", df.shape)
        st.subheader("Dataset Info :")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_string=buffer.getvalue()
        with st.container(height=317):
            st.text(info_string)
    with df_desc:
        st.subheader("Data Descrption :")
        st.write(X.describe())
    

    
# df = pd.read_csv("Breast Cancer Wisconsin.csv")
# X, y = get_features_and_target(df)
# data_overview(df,X,y)