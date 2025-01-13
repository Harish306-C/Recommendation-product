import openai
import pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from typing import Dict, List

# FastAPI initialization
app = FastAPI()

# OpenAI API Key
openai.api_key = "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"  # Replace with your OpenAI API Key

# Initialize Sentence Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone Initialization
PINECONE_API_KEY = "pcsk_3ZyWhu_AyLNPgS9ZaerfM7ysrBVHmzVfwdzxDDfp1mRzuisKTUPtnZjYARPvHBpY3Vn45q"  # Replace with your Pinecone API Key
PINECONE_ENV = "us-west1-gcp"  # Replace with your Pinecone environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create or connect to Pinecone index
vector_db_name = "crm-recommendation-db"
if vector_db_name not in pinecone.list_indexes():
    pinecone.create_index(vector_db_name, dimension=384)

vector_db = pinecone.Index(vector_db_name)

# Function to process data into chunks
def chunk_data(data: str, chunk_size: int = 512):
    chunks = [data[i: i + chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks

# Function to generate embeddings and store in Pinecone
def store_embeddings(data_chunks: List[str], metadata: Dict = None):
    embeddings = embedding_model.encode(data_chunks, convert_to_tensor=False)
    vectors = [{"id": f"chunk-{i}", "values": embeddings[i], "metadata": metadata or {}} for i in range(len(data_chunks))]
    vector_db.upsert(vectors)

# Function to retrieve relevant documents based on query
def retrieve_documents(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
    response = vector_db.query(query_embedding.tolist(), top_k=top_k, include_metadata=True)
    return response["matches"]

# Function to generate responses using LLM
def generate_response(query: str, documents: List[Dict]):
    document_texts = "\n".join([doc["metadata"].get("text", "") for doc in documents])
    prompt = f"Query: {query}\nRelevant Documents:\n{document_texts}\n\nGenerate a detailed response:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# FastAPI endpoint for CRM recommendations
@app.post("/api/recommendation")
async def recommendation(query: str):
    documents = retrieve_documents(query)
    response = generate_response(query, documents)
    return {"query": query, "recommendations": response}

# FastAPI endpoint for real-time sentiment analysis
@app.post("/api/sentiment")
async def sentiment_analysis(text: str):
    prompt = f"Analyze the sentiment of the following text: {text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"text": text, "sentiment_analysis": response["choices"][0]["message"]["content"]}