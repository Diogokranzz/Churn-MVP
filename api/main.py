from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os
import traceback
from contextlib import asynccontextmanager
import logging
import uuid
import json


try:
    from pinecone import Pinecone, ServerlessSpec
    pinecone_enabled = True
except ImportError:
    pinecone_enabled = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    root = _project_root()
    model_paths = [
        os.path.join(root, 'models', 'model.joblib'),
        os.path.join(root, 'models', 'model_lgbm.joblib'),
    ]
    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break

    global model, explainer, feature_names
    if model_path is None:
        model = None
    else:
        try:
            model = joblib.load(model_path)
            try:
                if hasattr(model, 'estimators_'):
                    for est in model.estimators_:
                        if not hasattr(est, 'monotonic_cst'):
                            setattr(est, 'monotonic_cst', None)
            except Exception:
                pass
        except Exception:
            model = None

    data_path = os.path.join(root, 'data', 'cleaned.csv')
    if not os.path.exists(data_path):
        explainer = None
        feature_names = None
    else:
        try:
            sample = pd.read_csv(data_path)
            feature_names = [c for c in sample.columns if c != 'churn']
            try:
                explainer = shap.Explainer(model, sample[feature_names])
            except Exception:
                try:
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    explainer = None
        except Exception:
            explainer = None
            feature_names = None

    yield



app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO, format='%(message)s')


if pinecone_enabled:
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY', 'demo-key'))
    index_name = os.environ.get('PINECONE_INDEX', 'churn-traces')
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index = pc.Index(index_name)


from fastapi.responses import JSONResponse
import csv
@app.get('/traces')
def get_traces():
    traces = []
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trace_modelo.csv'))
    if not os.path.exists(csv_path):
        return JSONResponse(content={"error": "Arquivo trace_modelo.csv não encontrado."}, status_code=404)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            traces.append(row)
    return traces


class PredictRequest(BaseModel):
    data: dict


model = None
explainer = None
feature_names = None


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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


@app.on_event('startup')
def load_model():
    global model, explainer, feature_names
    root = _project_root()
    model_paths = [
        os.path.join(root, 'models', 'model.joblib'),
        os.path.join(root, 'models', 'model_lgbm.joblib'),
    ]
    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        model = None
        return
    try:
        model = joblib.load(model_path)
        try:
            if hasattr(model, 'estimators_'):
                for est in model.estimators_:
                    if not hasattr(est, 'monotonic_cst'):
                        setattr(est, 'monotonic_cst', None)
        except Exception:
            pass
    except Exception:
        model = None
        return
    data_path = os.path.join(root, 'data', 'cleaned.csv')
    if not os.path.exists(data_path):
        explainer = None
        feature_names = None
        return
    try:
        sample = pd.read_csv(data_path)
        feature_names = [c for c in sample.columns if c != 'churn']
        try:
            explainer = shap.Explainer(model, sample[feature_names])
        except Exception:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = None
    except Exception:
        explainer = None
        feature_names = None


@app.post('/predict')
async def predict(req: PredictRequest, request: Request):
    trace_id = str(uuid.uuid4())
    numero_requisicao = request.headers.get('x-request-number', str(uuid.uuid4())[:8])
    if model is None:
        logging.error(json.dumps({"trace_id": trace_id, "numero_requisicao": numero_requisicao, "descricao": "Modelo não carregado"}, ensure_ascii=False))
        raise HTTPException(status_code=503, detail='model not loaded')
    if not isinstance(req.data, dict):
        logging.error(json.dumps({"trace_id": trace_id, "numero_requisicao": numero_requisicao, "descricao": "Payload inválido"}, ensure_ascii=False))
        raise HTTPException(status_code=422, detail='data must be um objeto')
    if feature_names is None:
        logging.error(json.dumps({"trace_id": trace_id, "numero_requisicao": numero_requisicao, "descricao": "Metadados de features não disponíveis"}, ensure_ascii=False))
        raise HTTPException(status_code=503, detail='feature metadata not available')
    df = pd.DataFrame([req.data])
    df = df.reindex(columns=feature_names, fill_value=0)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    try:
        probs = model.predict_proba(df)
        if probs is None:
            raise ValueError('predict_proba returned None')
        try:
            if getattr(probs, 'ndim', 2) > 1:
                row_sum = float(probs.sum(axis=1)[0])
            else:
                row_sum = float(probs[0])
        except Exception:
            row_sum = None
        if row_sum is not None and getattr(probs, 'ndim', 2) > 1:
            if not (0.999 <= row_sum <= 1.001):
                try:
                    probs = probs / row_sum
                except Exception:
                    pass
        if getattr(probs, 'ndim', 2) == 1:
            prob = float(probs[0])
        else:
            classes = getattr(model, 'classes_', None)
            idx = _positive_class_index(classes)
            if idx >= 0 and idx < probs.shape[1]:
                prob = float(probs[0, idx])
            else:
                prob = float(probs[0, -1])
        if not isinstance(prob, float):
            prob = float(prob)
        prob = max(0.0, min(1.0, prob))
    except Exception as e:
        logging.error(json.dumps({"trace_id": trace_id, "numero_requisicao": numero_requisicao, "descricao": "Erro ao calcular probabilidade", "erro": str(e)}, ensure_ascii=False))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail='prediction failed')
    pred = int(prob >= 0.5)
    trace = {
        "trace_id": trace_id,
        "numero_requisicao": numero_requisicao,
        "descricao": "Inferência de churn realizada",
        "payload": req.data,
        "score": prob,
        "classe_prevista": pred
    }
    logging.info(json.dumps(trace, ensure_ascii=False))
    if pinecone_enabled:
        try:
            pinecone_index.upsert([(trace_id, [prob], {"payload": json.dumps(req.data, ensure_ascii=False), "score": str(prob), "classe_prevista": str(pred), "numero_requisicao": numero_requisicao})])
        except Exception as e:
            logging.error(json.dumps({"trace_id": trace_id, "descricao": "Falha ao enviar trace para Pinecone", "erro": str(e)}, ensure_ascii=False))
    return {"probability": prob, "pred": pred, "trace_id": trace_id, "numero_requisicao": numero_requisicao}


@app.post('/explain')
async def explain(req: PredictRequest, request: Request):
    trace_id = str(uuid.uuid4())
    numero_requisicao = request.headers.get('x-request-number', str(uuid.uuid4())[:8])
    if explainer is None:
        logging.error(json.dumps({"trace_id": trace_id, "numero_requisicao": numero_requisicao, "descricao": "Explainer não disponível"}, ensure_ascii=False))
        raise HTTPException(status_code=503, detail='explainer not available')
    if not isinstance(req.data, dict):
        logging.error(json.dumps({"trace_id": trace_id, "numero_requisicao": numero_requisicao, "descricao": "Payload inválido"}, ensure_ascii=False))
        raise HTTPException(status_code=422, detail='data must be um objeto')
    df = pd.DataFrame([req.data])
    df = df.reindex(columns=feature_names, fill_value=0)
    try:
        shap_values = explainer(df)
        vals = shap_values.values[0]
        feature_importance = sorted(list(zip(feature_names, vals)), key=lambda x: abs(x[1]), reverse=True)
        top5 = feature_importance[:5]
        trace = {
            "trace_id": trace_id,
            "numero_requisicao": numero_requisicao,
            "descricao": "Explicação SHAP gerada",
            "payload": req.data,
            "top5_features": [{"feature": f, "valor": float(v)} for f, v in top5]
        }
        logging.info(json.dumps(trace, ensure_ascii=False))
        if pinecone_enabled:
            try:
                pinecone_index.upsert([(trace_id, [0.0], {"payload": json.dumps(req.data, ensure_ascii=False), "top5": json.dumps(trace["top5_features"], ensure_ascii=False), "numero_requisicao": numero_requisicao})])
            except Exception as e:
                logging.error(json.dumps({"trace_id": trace_id, "descricao": "Falha ao enviar trace para Pinecone", "erro": str(e)}, ensure_ascii=False))
        return {"explanation": [{"feature": f, "value": float(v)} for f, v in top5], "trace_id": trace_id, "numero_requisicao": numero_requisicao}
    except Exception as e:
        logging.error(json.dumps({"trace_id": trace_id, "numero_requisicao": numero_requisicao, "descricao": "Erro ao gerar explicação SHAP", "erro": str(e)}, ensure_ascii=False))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail='failed to compute explanation')
