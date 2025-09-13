import joblib, pandas as pd, traceback
try:
    import shap
except Exception as e:
    print("Erro ao importar shap:", e)
    raise

try:
    model = joblib.load('models/model.joblib')
    df = pd.read_csv('data/cleaned.csv')
    feature_names = [c for c in df.columns if c != 'churn']
    print("feature_names:", feature_names[:10], "... (total", len(feature_names), ")")
    explainer = shap.Explainer(model, df[feature_names])
    print("Explainer criado com sucesso")
    out = explainer(df[feature_names].iloc[:1])
    print("Exemplo SHAP values:", out.values.shape)
except Exception:
    traceback.print_exc()