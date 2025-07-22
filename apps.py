from flask import Flask, request, jsonify, render_template
from src.helper import load_pdf_file, split_texts, download_huggingface_embeddings
#from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter   
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import Ollama
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from src.prompt import *
import os
app = Flask(__name__)
load_dotenv()   
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY   
os.chdir('/Users/dnyanesh/Desktop/PDFReader/ChatBot-PDF-GenAI')
embeddings = download_huggingface_embeddings()

index_name = "pdf-test"

docsearch = PineconeVectorStore.from_existing_index (
    index_name=index_name,
    embedding=embeddings,
    
)

retrive = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        
    }
)

llm = Ollama(model="llama3", temperature=0.7)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retrive, question_answer_chain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get('msg', '')
    print(f"User input: {msg}")
    
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I didn't understand.")
        print(f"Bot response: {answer}")
    except Exception as e:
        answer = f"Error: {str(e)}"

    return jsonify({"answer": answer})

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'Data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files.get('pdf_file')
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({"message": "PDF uploaded successfully!"})
    return jsonify({"message": "Invalid file format"}), 400

@app.route('/clear_pdfs', methods=['POST'])
def clear_pdfs():
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.pdf'):
                os.remove(os.path.join(UPLOAD_FOLDER, filename))
        return jsonify({"message": "All PDFs cleared."})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
    