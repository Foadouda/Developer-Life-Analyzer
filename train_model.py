"""
Train a machine learning model to predict task success based on developer productivity metrics.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_model():
    """Train and save the best model"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("ai_dev_productivity.csv")
    
    # Features and target
    feature_cols = ['hours_coding', 'coffee_intake_mg', 'distractions', 
                    'sleep_hours', 'commits', 'bugs_reported', 
                    'ai_usage_hours', 'cognitive_load']
    
    X = df[feature_cols]
    y = df['task_success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train Random Forest model
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Train Logistic Regression model
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    # Evaluate models
    print("\n" + "="*50)
    print("RANDOM FOREST RESULTS")
    print("="*50)
    
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION RESULTS")
    print("="*50)
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    
    # Feature importance for Random Forest
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    # Save the Random Forest model (better performance)
    model_filename = "task_success_model.pkl"
    joblib.dump(rf_model, model_filename)
    print(f"\nModel saved as '{model_filename}'")
    
    # Save feature names for reference
    feature_names = {'features': feature_cols}
    joblib.dump(feature_names, "feature_names.pkl")
    
    return rf_model, X_test, y_test, feature_cols

if __name__ == "__main__":
    train_model()

