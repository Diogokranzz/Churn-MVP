import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "churn-traces")

if not API_KEY:
    raise RuntimeError("PINECONE_API_KEY não encontrado no ambiente!")

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)


index.delete(delete_all=True)

print("Todos os registros do index foram removidos!")
