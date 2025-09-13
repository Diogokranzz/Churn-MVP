if (-Not (Test-Path -Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r api\requirements.txt
python src\preprocess.py
python src\train.py
cd api
Start-Process -NoNewWindow -FilePath powershell -ArgumentList "-NoExit","uvicorn main:app --host 0.0.0.0 --port 8000"