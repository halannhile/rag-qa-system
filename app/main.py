from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from utils import allowed_file

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
index_name = "document-store"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # OpenAI embedding dimension

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
vectorstore = Pinecone.from_existing_index(index_name, embeddings)
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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

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
        
        Pinecone.from_documents(texts, embeddings, index_name=index_name)
        
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