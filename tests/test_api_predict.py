from fastapi.testclient import TestClient
import joblib
import os
import pandas as pd

import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
from api.main import app, model, feature_names


class DummyModel:
    def __init__(self):
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        import numpy as np
        n = X.shape[0]
        out = np.tile(np.array([0.2, 0.8]), (n, 1))
        return out


def test_predict_success(monkeypatch):
    client = TestClient(app)
    dummy = DummyModel()
    monkeypatch.setattr('api.main.model', dummy)
    # ensure feature_names exists
    if not feature_names:
        # build simple feature list
        monkeypatch.setattr('api.main.feature_names', ['tenure', 'MonthlyCharges', 'SeniorCitizen'])
    resp = client.post('/predict', json={'data': {'tenure': 12, 'MonthlyCharges': 70.0, 'SeniorCitizen': 0}})
    assert resp.status_code == 200
    j = resp.json()
    assert 'probability' in j and 0.0 <= j['probability'] <= 1.0
    assert 'pred' in j and j['pred'] in (0, 1)
