import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
try:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    mlflow_sklearn = None
    _HAS_MLFLOW = False
from preprocess import load_data, preprocess
from sklearn.metrics import roc_auc_score
import os
import joblib

def objective(trial):
    df = load_data('data/processed.csv')
    df = preprocess(df)
    y = df['churn']
    X = df.drop(columns=['churn'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    gbm = lgb.train(param, dtrain, valid_sets=[dval], num_boost_round=100)
    preds = gbm.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    return auc

def run_study(n_trials=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    df = load_data('data/processed.csv')
    df = preprocess(df)
    y = df['churn']
    X = df.drop(columns=['churn'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dtrain = lgb.Dataset(X_train, label=y_train)
    gbm = lgb.train({**best, 'objective':'binary','metric':'auc','verbosity':-1}, dtrain, num_boost_round=200)
    preds = gbm.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    if _HAS_MLFLOW:
        mlflow.start_run()
        mlflow.log_params(best)
        mlflow.log_metric('auc', float(auc))
        tracking_uri = mlflow.get_tracking_uri()
        try:
            if tracking_uri.startswith('file://'):
                os.makedirs('models', exist_ok=True)
                joblib.dump(gbm, 'models/model_lgbm.joblib')
                print(f"MLflow tracking URI is local ({tracking_uri}). Saved model to models/model_lgbm.joblib instead of logging to MLflow.")
            else:
                try:
                    mlflow_sklearn.log_model(gbm, 'model')
                except Exception:
                    os.makedirs('models', exist_ok=True)
                    joblib.dump(gbm, 'models/model_lgbm.joblib')
                    print('Failed to log model to MLflow; saved locally to models/model_lgbm.joblib')
        except Exception as e:
            os.makedirs('models', exist_ok=True)
            joblib.dump(gbm, 'models/model_lgbm.joblib')
            print('Unexpected error while attempting to log model to MLflow:', e)
        mlflow.end_run()
    else:
        os.makedirs('models', exist_ok=True)
        joblib.dump(gbm, 'models/model_lgbm.joblib')
        print('mlflow is not installed; saved model to models/model_lgbm.joblib')

if __name__ == '__main__':
    run_study(30)
