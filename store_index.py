import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set current working directory
os.chdir('/Users/dnyanesh/Desktop/PDFReader/ChatBot-PDF-GenAI')

# LangChain and helper functions
from src.helper import load_pdf_file, split_texts, download_huggingface_embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pinecone SDK
from pinecone import Pinecone, ServerlessSpec

# Step 1: Load and split PDFs
print("📄 Loading PDF files...")
extracted_data = load_pdf_file(Data='Data/')
text_chunks = split_texts(extracted_data)

# Step 2: Load embeddings
print("🔍 Downloading embeddings model...")
embeddings = download_huggingface_embeddings()

# Step 3: Initialize Pinecone
print("🧠 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdf-test"
dimension = 384  # Make sure this matches your embedding model

# Step 4: Check if index exists
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    print(f"📦 Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"✅ Index '{index_name}' already exists.")

# Step 5: Upload documents to Pinecone
print("🚀 Uploading documents to Pinecone...")
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Indexing complete.")