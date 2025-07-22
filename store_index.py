from src.helper import load_pdf_file, split_texts, download_huggingface_embeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter   
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


import os
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

import os
os.chdir('/Users/dnyanesh/Desktop/PDFReader/ChatBot-PDF-GenAI')

extracted_data = load_pdf_file(Data='Data/')
text_chunks = split_texts(extracted_data)
embeddings = download_huggingface_embeddings()

from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
import os
pc = Pinecone(api_key = PINECONE_API_KEY)
index_name = "pdf-test"

pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1")

)

from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,       
    index_name=index_name,
    
)