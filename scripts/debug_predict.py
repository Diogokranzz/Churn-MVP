
import joblib
import pandas as pd
import traceback
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_paths = [os.path.join(root, 'models', 'model.joblib'), os.path.join(root, 'models', 'model_lgbm.joblib')]

def find_model():
    for p in model_paths:
        if os.path.exists(p):
            return p
    return None

mp = find_model()
print('model_path=', mp)
model = joblib.load(mp)
print('Model type:', type(model))


def _apply_monotonic_shim(m):
    try:
        if hasattr(m, 'estimators_'):
            for est in m.estimators_:
                if not hasattr(est, 'monotonic_cst'):
                    setattr(est, 'monotonic_cst', None)
    except Exception:
        pass


def _positive_class_index(classes):
    if classes is None:
        return -1
    for i, c in enumerate(classes):
        try:
            if c is True or c == 1 or c == 1.0:
                return i
            s = str(c).lower()
            if s in ('1', 'true', 'yes', 'y', 'positive', 'pos'):
                return i
        except Exception:
            continue
    return len(classes) - 1



_apply_monotonic_shim(model)

data_path = os.path.join(root, 'data', 'cleaned.csv')
print('data_path=', data_path)
df_sample = pd.read_csv(data_path)
feature_names = [c for c in df_sample.columns if c != 'churn']
print('feature_names (count):', len(feature_names))
print(feature_names)


payload = {'tenure': 12, 'MonthlyCharges': 70.0, 'SeniorCitizen': 0}
print('payload keys:', payload.keys())


df = pd.DataFrame([payload])
print('raw df:')
print(df)
df = df.reindex(columns=feature_names, fill_value=0)
print('reindexed df head:')
print(df.head())
print('dtypes:')
print(df.dtypes)


print('any NaN?', df.isna().any().any())


for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass

print('dtypes after to_numeric:')
print(df.dtypes)


try:
    preds = model.predict_proba(df)
   
    try:
        if getattr(preds, 'ndim', 2) > 1:
            s = float(preds.sum(axis=1)[0])
            if not (0.999 <= s <= 1.001):
                preds = preds / s
                print(f'Normalized debug predict_proba row by sum {s}')
    except Exception:
        pass

    print('predict_proba shape:', preds.shape)
    print('predict_proba first row:', preds[0][:5] if preds.ndim>1 else preds[0])
except Exception:
    print('predict_proba failed:')
    traceback.print_exc()

try:
    pred = model.predict(df)
    print('predict shape:', getattr(pred, 'shape', None))
    print('predict first:', pred[:5])
except Exception:
    print('predict failed:')
    traceback.print_exc()


try:
    import shap
    try:
        expl = shap.Explainer(model, df_sample[feature_names])
    except Exception:
        try:
            expl = shap.TreeExplainer(model)
        except Exception:
            expl = None
    print('explainer type:', type(expl) if expl is not None else None)
    if expl is not None:
        out = expl(df)
        print('shap output shape:', out.values.shape)
except Exception:
    print('SHAP failed:')
    traceback.print_exc()