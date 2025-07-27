#from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter   
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.chdir('/Users/dnyanesh/Desktop/PDFReader/ChatBot-PDF-GenAI')

def load_pdf_file (Data):
    loader = DirectoryLoader(Data, glob ="*.pdf", loader_cls=PyPDFLoader)
    data = loader.load()
    return data


def split_texts (extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_huggingface_embeddings():
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

#pip install sentence-transformers