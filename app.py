import os
import pandas as pd
import tempfile
from flask import Flask, request, jsonify
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


class RAGChatbot:
    def __init__(self, data_path=None, df=None, model_name="gpt-3.5-turbo"):
        """
        Initialize the RAG Chatbot
        
        Args:
            data_path (str, optional): Path to the data file (CSV, JSON, etc.)
            df (DataFrame, optional): DataFrame containing the data
            model_name (str): Name of the LLM model to use
        """
        self.data_path = data_path
        self.df = df
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.vectorstore = None
        self.rag_chain = None
        
    def load_data(self):
        """Load the dataset and return a DataFrame"""
        if self.df is not None:
            return self.df
            
        file_extension = self.data_path.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(self.data_path)
        elif file_extension == 'json':
            df = pd.read_json(self.data_path)
        elif file_extension == 'xlsx' or file_extension == 'xls':
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns.")
        self.df = df
        return df
        
    def prepare_documents(self, df):
        """Convert DataFrame to documents and split into chunks"""
        # Combine text columns if there are multiple
        if 'text' not in df.columns:
            # Let's create a text column by combining all string columns
            string_columns = df.select_dtypes(include=['object']).columns.tolist()
            df['text'] = df[string_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            
        # Load documents from DataFrame
        loader = DataFrameLoader(df, page_content_column="text")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"Split data into {len(splits)} chunks")
        return splits
        
    def create_vectorstore(self, splits):
        """Create a vector store from the document chunks"""
        # Create embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        
        self.vectorstore = vectorstore
        print("Vector store created successfully")
        return self.vectorstore
        
    def setup_rag_pipeline(self):
        """Set up the RAG pipeline using LangChain"""
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create template for the prompt
        template = """You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question using only the provided context. If you cannot answer based on the context, say "I don't have enough information to answer this question."
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Set up the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.rag_chain = rag_chain
        return rag_chain
        
    def initialize(self):
        """Initialize the RAG chatbot"""
        df = self.load_data()
        splits = self.prepare_documents(df)
        self.create_vectorstore(splits)
        rag_chain = self.setup_rag_pipeline()
        return rag_chain
        
    def query(self, question):
        """Process a single query"""
        if self.rag_chain is None:
            self.initialize()
            
        try:
            response = self.rag_chain.invoke(question)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

# Global chatbot instance
chatbot_instance = None

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "RAG Chatbot API is running. Use /upload to upload a CSV file and /query to ask questions."
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    global chatbot_instance
    
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file part in the request"
        }), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected"
        }), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({
            "status": "error",
            "message": "Only CSV files are supported"
        }), 400
    
    # Save the file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Create a new chatbot instance
        chatbot_instance = RAGChatbot(data_path=temp_file.name)
        chatbot_instance.initialize()
        
        return jsonify({
            "status": "success",
            "message": "File uploaded and processed successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing file: {str(e)}"
        }), 500
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)

@app.route('/query', methods=['POST'])
def query():
    global chatbot_instance
    
    if chatbot_instance is None:
        return jsonify({
            "status": "error",
            "message": "Please upload a CSV file first using the /upload endpoint"
        }), 400
        
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide a question in the request body"
        }), 400
        
    question = data['question']
    response = chatbot_instance.query(question)
    
    return jsonify({
        "status": "success",
        "question": question,
        "answer": response
    })

# Adding a route to directly upload a DataFrame as JSON
@app.route('/upload-json', methods=['POST'])
def upload_json():
    global chatbot_instance
    
    data = request.get_json()
    
    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400
        
    try:
        df = pd.DataFrame(data)
        chatbot_instance = RAGChatbot(df=df)
        chatbot_instance.initialize()
        
        return jsonify({
            "status": "success",
            "message": "JSON data uploaded and processed successfully",
            "rows": len(df),
            "columns": len(df.columns)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing JSON data: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
