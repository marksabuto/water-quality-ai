import pytest
import numpy as np
import pandas as pd
import joblib
from train_model import load_and_preprocess_data, train_random_forest


def test_data_loading_and_preprocessing():
    """Test that data loads, missing values are handled, and scaling works."""
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # Check no NaNs after preprocessing
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    # Check scaling (mean ~0, std ~1)
    assert np.allclose(np.mean(X_train, axis=0), 0, atol=1)
    assert np.allclose(np.std(X_train, axis=0), 1, atol=0.5)


def test_model_training_and_prediction():
    """Test that the model trains and predicts with correct output shape/type."""
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    # Check prediction shape
    assert preds.shape[0] == X_test.shape[0]
    assert proba.shape == (X_test.shape[0], 2)
    # Check prediction values are 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_loading():
    """Test that the trained model and scaler can be loaded from disk."""
    bundle = joblib.load('water_quality_model.pkl')
    assert 'model' in bundle and 'scaler' in bundle
    model = bundle['model']
    scaler = bundle['scaler']
    # Check model has predict method
    assert hasattr(model, 'predict')
    # Check scaler has transform method
    assert hasattr(scaler, 'transform') 