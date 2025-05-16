import streamlit as st
from data_loader import handle_outliers
from tabs.data_vis import plot_boxplot



def Preprocessing(X):
    col1,col2,col3 = st.columns([1,3,1])
    
    with col2:
        st.header("Data Preprocessing")
        st.subheader("Original Data")
        fig = plot_boxplot(X, "Original Features")
        st.pyplot(fig)

        if st.button("handle Outlier") :
            X_processed = handle_outliers(X)
            st.subheader("After Outlier Handling")
            fig = plot_boxplot(X_processed, "Processed Features")
            st.pyplot(fig)
