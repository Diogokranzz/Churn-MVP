import os
import csv
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "churn-traces")

if not API_KEY:
    raise RuntimeError("PINECONE_API_KEY não encontrado no ambiente!")

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trace_modelo.csv'))

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    batch = []
    for row in reader:
       
     
        if row["score"] == "score":
            continue
        vector = [float(row["score"])]
        
        metadata = {}
        for key in ["description", "payload", "class", "timestamp", "grau_dificuldade"]:
            value = row.get(key)
            if value is not None and value != "":
                metadata[key] = value
        
        batch.append((row["trace_id"], vector, metadata))
       
        if len(batch) == 100:
            index.upsert(batch)
            batch = []
    if batch:
        index.upsert(batch)

print("Upload das traces do CSV para o Pinecone concluído!")
