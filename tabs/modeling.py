import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from data_loader import handle_outliers, get_high_correlation_features
from model import ModelEvaluator, scale_and_encode


def modeling(X,y):
    st.header("Model Training & Evaluation")
    X_processed = handle_outliers(X)
    high_corr_X = get_high_correlation_features(X_processed, 0.2)
    X_scaled, y_encoded = scale_and_encode(high_corr_X, y)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "KNN": KNeighborsRegressor()
    }
    model_options = ["All","Logistic Regression","Decision Tree","KNN"]
    model_selected = st.selectbox("select model",
                    model_options)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5)

    if st.button("Run Models"):
        if model_selected == "All":
            cols = st.columns(3)
            i = 0
            for name, model in models.items():
                evaluator = ModelEvaluator(model, test_size=test_size, threshold=threshold)
                results = evaluator.evaluate(X_scaled, y_encoded,name,model)
                with cols[i]:
                    st.subheader(results.get('Model'),divider=True)
                    st.metric("Accuracy", f"{results.get('Accuracy'):.2f}")
                    st.metric("MAE", f"{results.get('MAE'):.2f}")
                    st.write("Classification Report:")
                    st.text(results.get('Report'))
                    st.write("Confusion Matrix:")
                    st.dataframe(pd.DataFrame(results.get('Confusion Matrix'), 
                                                columns=['Predicted 0', 'Predicted 1'], 
                                                index=['Actual 0', 'Actual 1']))
                i+=1
        else :
            model = models.get(model_selected)
            evaluator = ModelEvaluator(model, test_size=test_size, threshold=threshold)
            results = evaluator.evaluate(X_scaled, y_encoded,model_name=model_selected,model=model)
            metrics = ['Model','MAE','MSE','RMSE','R²','Accuracy','Report','Confusion Matrix']
        
            st.subheader(results.get('Model'))
            st.metric("Accuracy", f"{results.get('Accuracy'):.2f}")
            st.metric("MAE", f"{results.get('MAE'):.2f}")
            st.write("Classification Report:")
            st.text(results.get('Report'))
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(results.get('Confusion Matrix'), 
                                        columns=['Predicted 0', 'Predicted 1'], 
                                        index=['Actual 0', 'Actual 1']))