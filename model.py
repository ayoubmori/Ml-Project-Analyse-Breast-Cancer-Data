import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ModelEvaluator:
    def __init__(self, model, test_size=0.2, random_state=42, threshold=0.5):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.threshold = threshold
        self.results = {}

    def evaluate(self, X, y,model_name,model):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Classification metrics (thresholded)
        y_pred_class = (y_pred > self.threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_class)
        report = classification_report(y_test, y_pred_class)
        cm = confusion_matrix(y_test, y_pred_class)

        self.results = {
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Accuracy': accuracy,
            'Report': report,
            'Confusion Matrix': cm
        }
        return self.results

def scale_and_encode(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X_scaled, y_encoded