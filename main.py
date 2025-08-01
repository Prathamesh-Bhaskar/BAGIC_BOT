from flask import Flask, request, jsonify, session, render_template
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
from typing import List, Dict, Any
import json
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here-change-this')

# Global RAG system instance
rag_system = None

class TravelInsuranceRAG:
    def __init__(self):
        # Astra DB configuration
        self.ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "")
        self.ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT", "")
        self.COLLECTION_NAME = os.environ.get("ASTRA_DB_COLLECTION_NAME", "travel_insurance_docs")
        
        # Azure OpenAI Configuration
        self.AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")
        self.AZURE_MODEL_NAME = os.environ.get("AZURE_MODEL_NAME", "")
        self.AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT", "")
        self.AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
        self.AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "")
        
        # Document paths
        pdf_files_env = os.environ.get("PDF_FILES", "")
        if pdf_files_env:
            self.PDF_FILES = [file.strip() for file in pdf_files_env.split(",")]
        else:
            self.PDF_FILES = [
                "Travel-Ace-PF.pdf",
                "Travel-Ace-Policy-brochure.pdf",
            ]
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self.plan_rag_chain = None
        self.documents = None  # For fallback mode
        self.fallback_mode = False
        
    def initialize_models(self):
        """Initialize Azure OpenAI models for embeddings and chat"""
        logger.info("🚀 Initializing Azure OpenAI models...")
        
        # Validate required environment variables
        required_vars = {
            'AZURE_ENDPOINT': self.AZURE_ENDPOINT,
            'AZURE_API_KEY': self.AZURE_API_KEY,
            'AZURE_API_VERSION': self.AZURE_API_VERSION,
            'AZURE_DEPLOYMENT': self.AZURE_DEPLOYMENT
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        try:
            # Initialize embeddings model
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.AZURE_ENDPOINT,
                api_key=self.AZURE_API_KEY,
                api_version=self.AZURE_API_VERSION,
                azure_deployment="text-embedding-ada-002",
                chunk_size=1000
            )
            
            # Initialize chat model
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.AZURE_ENDPOINT,
                api_key=self.AZURE_API_KEY,
                api_version=self.AZURE_API_VERSION,
                azure_deployment=self.AZURE_DEPLOYMENT,
                temperature=0.0,
                max_tokens=2000
            )
            
            logger.info("✅ Models initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
        
    def load_and_process_documents(self) -> List:
        """Load PDFs and split with table-aware chunking"""
        logger.info("📄 Loading and processing PDF documents...")
        
        all_docs = []
        
        for pdf_file in self.PDF_FILES:
            if not os.path.exists(pdf_file):
                logger.warning(f"⚠️  Warning: {pdf_file} not found, skipping...")
                continue
                
            logger.info(f"   Loading {pdf_file}...")
            
            try:
                # Load PDF
                loader = PyPDFLoader(pdf_file)
                pages = loader.load()
                
                # Enhanced metadata extraction
                for page_num, page in enumerate(pages):
                    page.metadata['document_name'] = pdf_file
                    page.metadata['document_type'] = 'travel_insurance'
                    page.metadata['page_number'] = page_num + 1
                    
                    # Content type classification
                    content = page.page_content.lower()
                    if 'summary of coverage' in content or 'coverage | plan' in content:
                        page.metadata['content_type'] = 'coverage_table'
                    elif 'premium' in content and ('travel days' in content or 'age' in content):
                        page.metadata['content_type'] = 'premium_table'
                    else:
                        page.metadata['content_type'] = 'general'
                
                all_docs.extend(pages)
                logger.info(f"   ✅ Loaded {len(pages)} pages from {pdf_file}")
                
            except Exception as e:
                logger.error(f"❌ Error loading {pdf_file}: {e}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully loaded")
        
        logger.info(f"📚 Total pages loaded: {len(all_docs)}")
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for doc in all_docs:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        
        # Enhanced metadata for chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_length': len(chunk.page_content),
                'word_count': len(chunk.page_content.split())
            })
        
        logger.info(f"✅ Created {len(chunks)} chunks")
        return chunks
        
    def create_vector_store(self, documents: List):
        """Create vector store with AstraDB and FAISS fallback"""
        logger.info("🗄️  Creating vector store...")
        
        # Try AstraDB first
        if self.ASTRA_DB_APPLICATION_TOKEN and self.ASTRA_DB_API_ENDPOINT:
            try:
                logger.info("🔌 Attempting AstraDB connection...")
                logger.info(f"📋 Collection: {self.COLLECTION_NAME}")
                logger.info(f"🔗 Endpoint: {self.ASTRA_DB_API_ENDPOINT}")
                
                from langchain_astradb import AstraDBVectorStore
                
                # Test connection with timeout
                self.vectorstore = AstraDBVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.COLLECTION_NAME,
                    token=self.ASTRA_DB_APPLICATION_TOKEN,
                    api_endpoint=self.ASTRA_DB_API_ENDPOINT,
                )
                
                logger.info(f"✅ AstraDB vector store created with {len(documents)} documents")
                
            except Exception as astra_error:
                logger.error(f"❌ AstraDB connection failed: {astra_error}")
                logger.info("🔄 Falling back to FAISS...")
                self._create_faiss_fallback(documents)
        else:
            logger.warning("⚠️  AstraDB credentials missing, using FAISS...")
            self._create_faiss_fallback(documents)
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
    def _create_faiss_fallback(self, documents: List):
        """Create FAISS vector store as fallback"""
        try:
            from langchain_community.vectorstores import FAISS
            
            logger.info("🗄️  Creating FAISS vector store (fallback mode)...")
            
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            self.fallback_mode = True
            logger.info(f"✅ FAISS vector store created with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"❌ FAISS fallback also failed: {e}")
            # Ultimate fallback: simple text search
            self._create_simple_search_fallback(documents)
    
    def _create_simple_search_fallback(self, documents: List):
        """Create simple text-based search as ultimate fallback"""
        logger.warning("⚠️  Using simple text search (limited functionality)")
        
        self.documents = documents
        self.fallback_mode = True
        
        # Simple search function
        def simple_search(query, k=5):
            scored_docs = []
            query_lower = query.lower()
            
            for doc in self.documents:
                content_lower = doc.page_content.lower()
                score = sum(1 for word in query_lower.split() if word in content_lower)
                
                if score > 0:
                    scored_docs.append((score, doc))
            
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scored_docs[:k]]
        
        # Create a mock retriever
        class SimpleRetriever:
            def __init__(self, search_func):
                self.search_func = search_func
            
            def get_relevant_documents(self, query):
                return self.search_func(query)
        
        self.retriever = SimpleRetriever(simple_search)
        logger.info("✅ Simple search fallback ready")
        
    def setup_rag_chains(self):
        """Set up RAG chains with error handling"""
        logger.info("🔗 Setting up RAG chains...")
        
        try:
            # General prompt
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert travel insurance assistant for Bajaj Allianz Travel Ace policies.
                
                Extract information from the provided context documents. Use only the information available in the context.
                If specific information is not found, state clearly that it's not available in the documents.
                
                Context: {context}"""),
                ("human", "{input}")
            ])
            
            # Plan-specific prompt
            plan_specific_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are answering questions specifically about the {selected_plan} plan.
                
                Use only information from the provided context that relates to {selected_plan}.
                Include specific coverage amounts and conditions for this plan.
                
                Context: {context}"""),
                ("human", "Question about {selected_plan}: {input}")
            ])
            
            # Create document chains
            if self.fallback_mode and hasattr(self, 'documents'):
                # Simple mode for fallback
                self.rag_chain = self._create_simple_chain(general_prompt)
                self.plan_rag_chain = self._create_simple_chain(plan_specific_prompt)
            else:
                # Full RAG chains
                general_document_chain = create_stuff_documents_chain(self.llm, general_prompt)
                plan_specific_document_chain = create_stuff_documents_chain(self.llm, plan_specific_prompt)
                
                self.rag_chain = create_retrieval_chain(self.retriever, general_document_chain)
                self.plan_rag_chain = create_retrieval_chain(self.retriever, plan_specific_document_chain)
            
            logger.info("✅ RAG chains setup complete!")
            
        except Exception as e:
            logger.error(f"❌ RAG chain setup failed: {e}")
            raise
    
    def _create_simple_chain(self, prompt_template):
        """Create simple chain for fallback mode"""
        def simple_chain_func(inputs):
            try:
                question = inputs.get('input', '')
                selected_plan = inputs.get('selected_plan', '')
                
                # Get relevant documents
                relevant_docs = self.retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs[:5]])
                
                # Format the prompt
                if selected_plan:
                    formatted_prompt = f"Context: {context}\n\nQuestion about {selected_plan}: {question}"
                else:
                    formatted_prompt = f"Context: {context}\n\nQuestion: {question}"
                
                response = self.llm.invoke([
                    ("system", "You are a travel insurance assistant. Answer based only on the provided context."),
                    ("human", formatted_prompt)
                ])
                
                return {
                    'answer': response.content,
                    'source_documents': relevant_docs
                }
                
            except Exception as e:
                logger.error(f"Simple chain error: {e}")
                return {
                    'answer': f"I apologize, but I encountered an error: {str(e)}",
                    'source_documents': []
                }
        
        return simple_chain_func
        
    def query_plan_specific(self, question: str, selected_plan: str) -> dict:
        """Query with plan-specific context"""
        try:
            logger.info(f"🔍 Querying for: {question} (Plan: {selected_plan})")
            
            if self.fallback_mode:
                result = self.plan_rag_chain({
                    'input': question,
                    'selected_plan': selected_plan
                })
            else:
                result = self.plan_rag_chain.invoke({
                    'input': question,
                    'selected_plan': selected_plan
                })
            
            return {
                'answer': result.get('answer', 'No answer generated'),
                'source_documents': result.get('source_documents', []),
                'query': question,
                'selected_plan': selected_plan
            }
            
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return {
                'answer': f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
                'source_documents': [],
                'query': question,
                'selected_plan': selected_plan
            }
    
    def query_general(self, question: str) -> dict:
        """Query for general information"""
        try:
            logger.info(f"🔍 General query: {question}")
            
            if self.fallback_mode:
                result = self.rag_chain({'input': question})
            else:
                result = self.rag_chain.invoke({'input': question})
            
            return {
                'answer': result.get('answer', 'No answer generated'),
                'source_documents': result.get('source_documents', []),
                'query': question
            }
            
        except Exception as e:
            logger.error(f"❌ General query error: {e}")
            return {
                'answer': f"I apologize, but I encountered an error: {str(e)}",
                'source_documents': [],
                'query': question
            }
    
    def get_available_plans(self) -> dict:
        """Get available plans information"""
        try:
            query = "List all Travel Ace insurance plans with their coverage amounts and key features"
            result = self.query_general(query)
            
            return {
                'success': True,
                'plans_info': result.get('answer', 'Plans information not available'),
                'source_documents': result.get('source_documents', [])
            }
            
        except Exception as e:
            logger.error(f"❌ Error retrieving plans: {e}")
            return {
                'success': False,
                'error': str(e),
                'plans_info': 'Unable to retrieve plan information'
            }
    
    def setup(self):
        """Complete setup process with fallback capabilities"""
        logger.info("🏗️  Setting up Travel Insurance RAG System...")
        logger.info("=" * 60)
        
        try:
            # Initialize models
            self.initialize_models()
            
            # Load and process documents
            documents = self.load_and_process_documents()
            
            if not documents:
                raise ValueError("No documents were loaded. Please check your PDF files.")
            
            # Create vector store (with fallback)
            self.create_vector_store(documents)
            
            # Setup RAG chains
            self.setup_rag_chains()
            
            mode = "FALLBACK MODE" if self.fallback_mode else "FULL MODE"
            logger.info("=" * 60)
            logger.info(f"🎉 RAG System setup complete! Running in {mode}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

# [Keep all existing Flask routes unchanged - they're fine]

@app.route('/')
def home():
    """Home page with plan selection"""
    return render_template('index.html')

@app.route('/api/plans', methods=['GET'])
def get_plans():
    """Get all available plans"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized',
                'message': 'System is still initializing. Please try again.'
            }), 503
        
        plans_result = rag_system.get_available_plans()
        
        if plans_result['success']:
            return jsonify({
                'success': True,
                'plans_info': plans_result['plans_info'],
                'message': 'Plans retrieved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': plans_result.get('error', 'Unknown error'),
                'message': 'Failed to retrieve plans'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve plans'
        }), 500

@app.route('/api/select-plan', methods=['POST'])
def select_plan():
    """Select a specific plan"""
    try:
        data = request.get_json()
        plan_name = data.get('plan_name', '').strip()
        
        if not plan_name:
            return jsonify({
                'success': False,
                'error': 'Plan name required'
            }), 400
        
        # Store selected plan in session
        session['selected_plan'] = plan_name
        
        return jsonify({
            'success': True,
            'selected_plan': plan_name,
            'plan_details': f'✅ {plan_name} selected successfully. You can now ask questions about this plan.',
            'sources': [],
            'message': f'{plan_name} plan selected successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to select plan'
        }), 500

@app.route('/api/query', methods=['POST'])
def query_plan():
    """Query for plan-specific information"""
    try:
        selected_plan = session.get('selected_plan')
        if not selected_plan:
            return jsonify({
                'success': False,
                'error': 'No plan selected',
                'message': 'Please select a plan first'
            }), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Empty question'
            }), 400
        
        global rag_system
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized'
            }), 503
        
        # Query the RAG system
        result = rag_system.query_plan_specific(question, selected_plan)
        
        # Process sources
        sources = []
        if result.get('source_documents'):
            for doc in result['source_documents'][:5]:
                sources.append({
                    'document': doc.metadata.get('document_name', 'Unknown'),
                    'page': doc.metadata.get('page_number', 'Unknown'),
                    'content_type': doc.metadata.get('content_type', 'general'),
                    'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'question': result['query'],
            'selected_plan': selected_plan,
            'sources': sources,
            'message': 'Query processed successfully'
        })
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process query'
        }), 500

@app.route('/api/current-plan', methods=['GET'])
def get_current_plan():
    """Get currently selected plan"""
    selected_plan = session.get('selected_plan')
    
    if not selected_plan:
        return jsonify({
            'success': False,
            'message': 'No plan selected'
        })
    
    return jsonify({
        'success': True,
        'selected_plan': selected_plan,
        'message': 'Current plan retrieved'
    })

@app.route('/api/reset-plan', methods=['POST'])
def reset_plan():
    """Reset plan selection"""
    session.pop('selected_plan', None)
    return jsonify({
        'success': True,
        'message': 'Plan selection reset'
    })

@app.route('/api/general-query', methods=['POST'])
def general_query():
    """Query for general information"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Empty question'
            }), 400
        
        global rag_system
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized'
            }), 503
        
        result = rag_system.query_general(question)
        
        # Process sources
        sources = []
        if result.get('source_documents'):
            for doc in result['source_documents'][:5]:
                sources.append({
                    'document': doc.metadata.get('document_name', 'Unknown'),
                    'page': doc.metadata.get('page_number', 'Unknown'),
                    'content_type': doc.metadata.get('content_type', 'general'),
                    'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'question': result['query'],
            'sources': sources,
            'message': 'General query processed successfully'
        })
        
    except Exception as e:
        logger.error(f"General query error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process general query'
        }), 500

def initialize_rag_system():
    """Initialize RAG system with enhanced error handling"""
    global rag_system
    try:
        logger.info("🔄 Initializing RAG system...")
        rag_system = TravelInsuranceRAG()
        rag_system.setup()
        logger.info("✅ RAG system initialized successfully!")
        return rag_system
        
    except Exception as e:
        logger.error(f"❌ RAG system initialization failed: {e}")
        
        # Try to create a minimal system for basic functionality
        try:
            logger.info("🔄 Attempting minimal system setup...")
            rag_system = TravelInsuranceRAG()
            rag_system.initialize_models()  # At least get the LLM working
            logger.info("⚠️  Minimal system ready (limited functionality)")
            return rag_system
        except Exception as minimal_error:
            logger.error(f"❌ Even minimal setup failed: {minimal_error}")
            return None

if __name__ == '__main__':
    # Get port for Render deployment
    port = int(os.environ.get("PORT", 5000))
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if rag_system:
        # Import and initialize chatbot
        try:
            from chatbot import chatbot_bp, initialize_chatbot
            initialize_chatbot(rag_system)
            app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
            logger.info("✅ Chatbot initialized")
        except Exception as e:
            logger.error(f"⚠️  Chatbot initialization failed: {e}")
    
    # Start the application
    logger.info(f"🚀 Starting Flask application on port {port}...")
    app.run(
        debug=False,  # Always False for production
        host='0.0.0.0', 
        port=port
    )
