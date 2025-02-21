# main.py

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    ServiceContext, 
    load_index_from_storage
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
import chainlit as cl
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Load environment variables
load_dotenv()

# Initialize LLM and Embedding Model
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define Settings
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900
Settings.chunk_size = 512
Settings.chunk_overlap = 64

# Load Documents
reader = SimpleDirectoryReader(input_dir="data/")
documents = reader.load_data(num_workers=4)

# Build the Index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create Query Engine
query_engine = index.as_query_engine(llm=llm)

# Chainlit Application
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Welcome! Ask any question about the documents."
    ).send()

@cl.on_message
async def on_message(message):
    query = message.content
    resp = query_engine.query(query)
    await cl.Message(
        content=f"Response: {resp.response}"
    ).send()
