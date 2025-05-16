import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px

def get_features_and_target(df):
    y = df['diagnosis']
    x = df.drop(['id','diagnosis'], axis=1)
    return x, y

@st.fragment
def plot_boxplot(data, title):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, orient='h', palette='Set2')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

@st.fragment
def plot_correlation_heatmap(corr_matrix, title):
    plt.figure(figsize=(10,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()

@st.fragment
def plot_interval_distribution(corr_values):
    intervals = pd.cut(corr_values, bins=[-1, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1],
                       labels=["(-1, -0.8]", "(-0.8, -0.5]", "(-0.5, -0.2]", "(-0.2, 0]",
                               "(0, 0.2]", "(0.2, 0.5]", "(0.5, 0.8]", "(0.8, 1]"])
    counts = intervals.value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Correlation Coefficient Distribution')
    plt.xlabel('Interval')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

@st.fragment
def get_high_correlation_features(x, threshold=0.2):
    corr_matrix = x.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr = (corr_matrix > threshold) & mask
    cols = set(corr_matrix.columns[high_corr.any(axis=0)]).union(set(corr_matrix.columns[high_corr.any(axis=1)]))
    return x[list(cols)]

@st.fragment
def plot_box(X,column):
    plt.figure(figsize=(7, 5.5))
    sns.boxplot(data=X[column], orient='h', palette='Set2')
    # Add labels and title
    plt.xlabel('Feature Value')
    plt.tight_layout()
    return plt.gcf()


@st.fragment
def Correlation(X):
    st.header("Feature Correlation Analysis")
    corr_matrix = X.corr()
    fig = plot_correlation_heatmap(corr_matrix, "Feature Correlation Matrix")
    st.pyplot(fig)

    corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
    fig = plot_interval_distribution(corr_values)
    st.pyplot(fig)

    threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5)
    if st.button("create correlation fig"):
        high_corr_X = get_high_correlation_features(X, threshold)
        st.write("High-Correlation Features:", high_corr_X.columns.tolist())
        fig = plot_correlation_heatmap(high_corr_X.corr(), "High-Correlation Features")
        st.pyplot(fig)

def visualization (df,X,y):

    st.title("Data Vis & Feature Correlation Analysis")
        
    cl = st.columns([4,1.5])
    with cl[0]:
        feature = st.selectbox("select feature",X.columns[1:-1])
        st.line_chart(X[feature],height=400)
    with cl[1]:
        # Horizontal Boxplot of a single feature
        st.subheader(f'Box Plot for {feature}')
        fig = px.box(X, x=feature, points="outliers", color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True,config={"displayModeBar": False})
        
    st.write("---")
    cols = st.columns([2,0.2,3,0.2,3])
    with cols[0]:
        st.subheader("Diagnosis Distribution :")
        st.bar_chart(y.value_counts(),height=500)
    with cols[1]:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid #B6B09F;
                        height: 500px;
                        margin: auto;
                    }
                </style>
            '''
        )
    with cols[2]:
        # Boxplot of feature by diagnosis using Plotly
        st.subheader(f"Boxplot: {feature} by Diagnosis")
        fig2 = px.box(df, y="diagnosis", x=feature, color="diagnosis", color_discrete_sequence=px.colors.diverging.RdBu)
        st.plotly_chart(fig2, use_container_width=True)
    with cols[3]:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid #B6B09F;
                        height: 500px;
                        margin: auto;
                    }
                </style>
            '''
        )
    with cols[4]:
        # Scatter plot using Plotly
        st.subheader("Scatter Plot Between Two Features")
        sub_cols = st.columns([2, 2])
        col1 = sub_cols[0].selectbox("X-axis", df.columns[2:], index=0)
        col2 = sub_cols[1].selectbox("Y-axis", df.columns[2:], index=1)

        fig4 = px.scatter(df, x=col1, y=col2, color="diagnosis", color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig4, use_container_width=True)
        
        
    # with cols[1]:
    #     # Boxplot of feature by diagnosis
    #     st.subheader(f"Boxplot: {feature} by Diagnosis")
    #     fig2, ax2 = plt.subplots()
    #     sns.boxplot(data=df, y="diagnosis", x=feature, palette="coolwarm", ax=ax2)
    #     st.pyplot(fig2)
    # # with cols[1]:
    # with cols[2]:
    #     # Scatter plot
    #     st.subheader("Scatter Plot Between Two Features")
    #     sub_cols = st.columns([2,2])
    #     col1 = sub_cols[0].selectbox("X-axis", df.columns[2:], index=0)
    #     col2 = sub_cols[1].selectbox("Y-axis", df.columns[2:], index=1)
        
    #     fig4, ax4 = plt.subplots()
    #     sns.scatterplot(data=df, x=col1, y=col2, hue="diagnosis", ax=ax4, palette="Dark2")
    #     st.pyplot(fig4)
        
    
    
    st.write("---")
    corr_values_col,corr_col  = st.columns([3,3])
    corr_matrix = X.corr()
    with corr_values_col :
        corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
        fig = plot_interval_distribution(corr_values)
        st.pyplot(fig)
    with corr_col :
        fig = plot_correlation_heatmap(corr_matrix, "Feature Correlation Matrix")
        # st.pyplot(fig)
        threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5)
        if st.button("create correlation fig"):
            high_corr_X = get_high_correlation_features(X, threshold)
            with st.expander("High-Correlation Features:"):
                st.write(high_corr_X.columns.tolist())
            fig = plot_correlation_heatmap(high_corr_X.corr(), "High-Correlation Features")
            st.pyplot(fig)


    
    
        

# df = pd.read_csv("Breast Cancer Wisconsin.csv")
# X, y = get_features_and_target(df)
# visualization(X,y)



# st.write("This app provides visual insights into breast cancer data.")


# # Correlation heatmap
# if st.checkbox("Show Correlation Heatmap (Top 10 Features)"):
#     st.subheader("Correlation Heatmap")
#     numeric_df = df.select_dtypes(include='number')
#     top_corr = numeric_df.corr().abs().sum().sort_values(ascending=False)[1:11].index
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     sns.heatmap(numeric_df[top_corr].corr(), annot=True, cmap="coolwarm", ax=ax3)
#     st.pyplot(fig3)

