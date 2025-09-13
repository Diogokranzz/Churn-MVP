import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os
try:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    mlflow_sklearn = None
    _HAS_MLFLOW = False

from preprocess import load_data, preprocess


def train(path):
    df = load_data(path)
    df = preprocess(df)
    print('data shape', df.shape)
    if 'churn' in df.columns:
        print('churn distribution', df['churn'].value_counts().to_dict())
    if 'churn' not in df.columns:
        print('churn column not found')
        return
    y = df['churn']
    X = df.drop(columns=['churn'])
    if y.nunique() < 2:
        print('Only one class present in target; provide dataset with both classes')
        return
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    n_estimators = 100
    random_state = 42
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    if len(set(y_val)) > 1:
        auc = roc_auc_score(y_val, preds)
    else:
        auc = float('nan')
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.joblib')
    if _HAS_MLFLOW:
        with mlflow.start_run():
            mlflow.log_param('n_estimators', n_estimators)
            mlflow.log_param('random_state', random_state)
            mlflow.log_param('n_features', X.shape[1])
            if not (isinstance(auc, float) and auc != auc):
                mlflow.log_metric('auc', float(auc))
            tracking_uri = mlflow.get_tracking_uri()
            try:
                if tracking_uri.startswith('file://'):
                    print(f"MLflow tracking URI is local ({tracking_uri}). Saved model to models/model.joblib instead of logging to MLflow.")
                else:
                    try:
                        mlflow_sklearn.log_model(model, 'model')
                    except Exception:
                        joblib.dump(model, 'models/model.joblib')
                        print('Failed to log model to MLflow; saved locally to models/model.joblib')
            except Exception as e:
                joblib.dump(model, 'models/model.joblib')
                print('Unexpected error while attempting to log model to MLflow:', e)
    else:
        print('mlflow is not installed; skipping MLflow logging. Model saved to models/model.joblib')
    print('AUC', auc)


if __name__ == '__main__':
    train('data/cleaned.csv')
