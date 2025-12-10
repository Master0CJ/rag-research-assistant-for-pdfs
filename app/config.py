import os
from dotenv import load_dotenv
load_dotenv()
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    VECTOR_DB_PATH = 'vector_db'
    # LangSmith configuration
    LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING', 'true')
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
    LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', 'rag-kms-hansa')
