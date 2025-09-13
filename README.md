# Projeto: Churn Prediction - MVP

## Visão Geral
Este projeto é um MVP completo para previsão de churn (cancelamento de clientes), pensado para portfólio e demonstração profissional. Ele cobre todo o ciclo de machine learning aplicado a churn:

- **Preprocessamento de dados**: scripts para limpeza, transformação e preparação dos dados.
- **Treinamento e tuning**: modelos treinados com scikit-learn/LightGBM, busca de hiperparâmetros com Optuna.
- **Tracking de experimentos**: integração com MLflow para registro de runs, métricas e artefatos.
- **API robusta**: FastAPI com endpoints para inferência (`/predict`), explicação SHAP (`/explain`) e consulta de traces (`/traces`).
- **Observabilidade e tracing**: cada requisição gera um trace detalhado, logado e (opcionalmente) enviado para Pinecone para busca e análise.
- **Explicabilidade**: integração com SHAP para explicar as previsões do modelo.
- **Automação e testes**: scripts utilitários, testes automatizados com pytest e CI via GitHub Actions.
- **Portabilidade**: fácil de rodar localmente, com instruções para ambiente virtual, dependências e execução.

## Funcionalidades
- Pipeline completo de ML: preprocessamento, treino, tuning, tracking e deploy.
- API RESTful pronta para produção (FastAPI).
- Tracking de experimentos e modelos com MLflow (local ou remoto).
- Explicabilidade de previsões com SHAP.
- Tracing detalhado de cada requisição, com integração opcional ao Pinecone.
- Scripts para upload e limpeza de traces no Pinecone.
- Testes automatizados e integração contínua.
- Pronto para portfólio: documentação, exemplos, dicas de apresentação e automação.

Estrutura principal
- `data/` - arquivos de dados processados
- `models/` - modelos salvos (`model.joblib`, `model_lgbm.joblib`)
- `src/` - scripts de preprocess, treino e tuning
- `api/` - servidor FastAPI
- `scripts/` - utilitários de debug e testes manuais
- `tests/` - testes automatizados
- `requirements.txt` - dependências do projeto

Instalação e execução local (PowerShell)
1) Criar e ativar ambiente virtual
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Preparar dados
Coloque o CSV processado em `data/cleaned.csv` com a coluna alvo `churn`.

3) Treinar modelo (recomendado para compatibilidade local)
```powershell
python .\src\train.py
```

4) Iniciar API
```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Exemplos de requests
- Inferência:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Body (ConvertTo-Json @{data=@{tenure=12; MonthlyCharges=70.0; SeniorCitizen=0}}) -ContentType 'application/json'
```
- Explicação (SHAP):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/explain -Body (ConvertTo-Json @{data=@{tenure=12; MonthlyCharges=70.0; SeniorCitizen=0}}) -ContentType 'application/json'
```

MLflow
- Modo local (padrão): os runs são salvos em `mlruns/`. Scripts fazem fallback para salvar `models/*.joblib` quando o tracking URI local impede upload de artefatos.
- Modo servidor: rode um servidor MLflow e aponte `MLFLOW_TRACKING_URI` para o endpoint HTTP para uploads automáticos.

CI
O workflow `.github/workflows/ci.yml` executa preprocess e testes com `pytest`.

Sugestões para portfólio
- Documente métricas e decisões de modelagem no README.
- Adicione screenshots ou um vídeo curto demonstrando o fluxo.

Comandos úteis
```powershell
# congelar dependências
& ".\.venv\Scripts\python.exe" -m pip freeze > requirements.txt

# rodar testes
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Licença
Adicione uma licença se desejar publicar o projeto.
Projeto: Churn Prediction - MVP

Objetivo
Construir um MVP de modelo preditivo de churn com endpoint de inferência e notebook de baseline.

Estrutura sugerida
- data/
- models/
- notebooks/
- src/
- api/
- requirements.txt

Como usar localmente
1. Criar ambiente virtual
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2. Preparar dados
Colocar arquivo processado em `data/processed.csv` com coluna alvo `churn`.

3. Treinar modelo
```powershell
python src/train.py
```

4. Rodar API
```powershell
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

Portfólio
Inclua no README descrição do problema, métricas usadas, resultados e um link para demo em vídeo.

MLflow
- instalar e rodar UI:
```powershell
python -m pip install mlflow
mlflow ui
```
o painel fica em `http://localhost:5000` e os runs são gravados em `mlruns/`.

Configuração do MLflow (local vs server)

- **Modo local (padrão)**: por padrão o MLflow grava em `mlruns/` usando um tracking URI local `file://.../mlruns`. Nesse modo nosso script `src/tune.py` salva o modelo final em `models/model_lgbm.joblib` e registra métricas/params no diretório `mlruns/`.

- **Modo servidor (recomendado para teams)**: rode o servidor MLflow e aponte `MLFLOW_TRACKING_URI` para o endpoint HTTP do servidor para que `mlflow.sklearn.log_model` funcione corretamente. Exemplo:

```powershell
setx MLFLOW_TRACKING_URI "http://localhost:5000"

$env:MLFLOW_TRACKING_URI = 'http://localhost:5000'
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Depois a execução do `python src/tune.py` irá tentar fazer upload do artefato para o servidor MLflow em vez de salvar somente localmente.

CI
o repositório contém um workflow GitHub Actions em `.github/workflows/ci.yml` que roda `preprocess` e os testes com `pytest`.

Automação
rode `.\run_all.ps1` para criar ambiente, instalar dependências, executar preprocess, treinar e iniciar a API.

Tuning
rode `python src/tune.py` para executar uma busca de hiperparâmetros com Optuna e treinar um modelo LightGBM.
o script salva o melhor modelo em `models/model_lgbm.joblib` e registra o run no MLflow.

## Tracing e Observabilidade

Cada requisição para `/predict` e `/explain` gera um trace detalhado, logado no console e (opcionalmente) enviado para Pinecone.

### Exemplo de trace de inferência
```json
{
  "trace_id": "e9b4e6e5-472d-4736-80e5-f21c4e3aab56",
  "numero_requisicao": "4e4f9b99",
  "descricao": "Inferência de churn realizada",
  "payload": {"tenure": 12, "MonthlyCharges": 70.0, "SeniorCitizen": 0},
  "score": 0.52,
  "classe_prevista": 1
}
```
- **trace_id**: identificador único da requisição
- **numero_requisicao**: número curto para rastreio
- **descricao**: descrição em português da ação
- **payload**: dados recebidos
- **score**: probabilidade prevista de churn
- **classe_prevista**: 0 (não churn) ou 1 (churn)

### Como visualizar
- Os traces aparecem no terminal do Uvicorn a cada requisição.
- Se Pinecone estiver configurado, também são enviados para o índice remoto.
- O retorno da API inclui `trace_id` e `numero_requisicao` para rastreabilidade.

### Exemplo de uso
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Body (ConvertTo-Json @{data=@{tenure=12; MonthlyCharges=70.0; SeniorCitizen=0}}) -ContentType 'application/json'
```

O log completo da trace será exibido no console e pode ser consultado por ID.

## Comandos essenciais para testar o funcionamento

```powershell
# 1. Criar e ativar ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Treinar o modelo
python src/train.py

# 4. Iniciar a API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Fazer uma requisição de inferência (em outro terminal)
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{ "data": { "age": 30, "plan": "premium", "usage": 100 } }'

# 6. Fazer uma requisição de explicação (SHAP)
curl -X POST "http://localhost:8000/explain" -H "Content-Type: application/json" -d '{ "data": { "age": 30, "plan": "premium", "usage": 100 } }'

# 7. Visualizar traces do CSV via API
curl http://localhost:8000/traces

# 8. Rodar testes automatizados
.venv\Scripts\python.exe -m pytest -q

# 9. (Opcional) Upload das traces do CSV para o Pinecone
python scripts/upload_traces_to_pinecone.py

# 10. (Opcional) Limpar todos os registros do Pinecone
python scripts/limpar_pinecone.py
```

> Todos os comandos acima podem ser executados no PowerShell, na raiz do projeto.
