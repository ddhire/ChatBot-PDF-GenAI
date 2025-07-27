from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore

from src.helper import download_huggingface_embeddings
from src.prompt import system_prompt

import os
import subprocess

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Constants
UPLOAD_FOLDER = 'Data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
index_name = "pdf-test"

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Change working directory if needed
os.chdir('/Users/dnyanesh/Desktop/PDFReader/ChatBot-PDF-GenAI')

# Load embeddings
embeddings = download_huggingface_embeddings()

# Try to load existing index or create it
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
except Exception as e:
    print("[INFO] Index not found. Executing store_index.py to create it...")
    result = subprocess.run(
        ['python', 'store_index.py'],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("[INFO] store_index.py executed:", result.stdout.strip())

    # Retry loading after creation
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

# Build Retriever and Chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = Ollama(model="llama3", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get('msg', '')
    print(f"[USER]: {msg}")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I didn't understand.")
        print(f"[BOT]: {answer}")
    except Exception as e:
        answer = f"Error: {str(e)}"

    return jsonify({"answer": answer})


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


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    try:
        result = subprocess.run(
            ['python', 'store_index.py'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return jsonify({"message": result.stdout.strip()})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.strip()}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)