"""
train_model.py
Script for training and comparing multiple models for water potability prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optional: XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def load_and_preprocess_data():
    df = pd.read_csv('data/water_potability.csv')
    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, scaler

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    if XGBClassifier is None:
        print("XGBoost not installed.")
        return None
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def train_lightgbm(X_train, y_train):
    if LGBMClassifier is None:
        print("LightGBM not installed.")
        return None
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    return lgbm

def train_svm(X_train, y_train):
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_

def cross_validate_model(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print(f"Cross-validated ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}

def train_ensemble(X_train, y_train, models):
    estimators = [(name, mdl) for name, mdl in models.items() if mdl is not None]
    if not estimators:
        print("No models for ensemble.")
        return None
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble

def export_best_model(model, scaler, filename='water_quality_model.pkl'):
    # Save both model and scaler for later use
    joblib.dump({'model': model, 'scaler': scaler}, filename)
    print(f"Best model saved to {filename}")

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    print("\nTraining models...")
    models = {}
    models['RandomForest'] = train_random_forest(X_train, y_train)
    models['SVM'] = train_svm(X_train, y_train)
    models['XGBoost'] = train_xgboost(X_train, y_train)
    models['LightGBM'] = train_lightgbm(X_train, y_train)

    print("\nEvaluating models...")
    metrics = {}
    for name, model in models.items():
        if model is not None:
            print(f"\n{name}:")
            metrics[name] = evaluate_model(model, X_test, y_test)

    print("\nTraining ensemble model...")
    ensemble = train_ensemble(X_train, y_train, models)
    if ensemble is not None:
        print("\nEnsemble:")
        metrics['Ensemble'] = evaluate_model(ensemble, X_test, y_test)

    # Select best model by ROC-AUC
    best_model_name = max(metrics, key=lambda k: metrics[k]['roc_auc'])
    best_model = models.get(best_model_name, None)
    if best_model_name == 'Ensemble':
        best_model = ensemble
    print(f"\nBest model: {best_model_name}")

    export_best_model(best_model, scaler)

if __name__ == "__main__":
    main() 