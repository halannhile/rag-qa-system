from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from utils import allowed_file
from pinecone import Pinecone, ServerlessSpec
import logging

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure the upload settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit.

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Delete the existing index
if "document-store" in pc.list_indexes().names():
    pc.delete_index("document-store")

# Create a new index with the correct dimension
pc.create_index(
    name="document-store",
    dimension=384,  # Set this to match your current embedding model
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Get the index
index = pc.Index("document-store")

# # Initialize LangChain components
# embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

# Replace the OpenAI embeddings with HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create LangChain vectorstore
vectorstore = LangChainPinecone.from_existing_index("document-store", embeddings)

# Initialize OpenAI LLM
llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))

# Create a custom prompt template
template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Initialize the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET'])
def index():
    app.logger.info('Index route accessed')
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f'Error rendering template: {str(e)}')
        return f"An error occurred: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process and index the file using LangChain
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Use LangChain's Pinecone integration to index the documents
        index_name = "document-store"  # Define the index name here
        LangChainPinecone.from_documents(texts, embeddings, index_name=index_name)
        
        return jsonify({"message": "File uploaded and indexed successfully"}), 200
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Use the RetrievalQA chain to get the answer
    result = qa_chain({"query": question})
    
    answer = result['result']
    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
    
    return jsonify({"answer": answer, "sources": sources}), 200

if __name__ == '__main__':
    app.run(debug=True)
