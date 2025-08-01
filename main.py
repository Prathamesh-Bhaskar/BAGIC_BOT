from flask import Flask, request, jsonify, session, render_template
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_astradb import AstraDBVectorStore
import time
from typing import List, Dict, Any
import json
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this'  # Change this in production

# Global RAG system instance (this will be shared with the chatbot)
rag_system = None

class TravelInsuranceRAG:
    def __init__(self):
        # Astra DB configuration
        self.ASTRA_DB_APPLICATION_TOKEN = ""
        self.ASTRA_DB_API_ENDPOINT = ""
        self.COLLECTION_NAME = ""  # New version for table-aware processing
        
        # Azure OpenAI Configuration
        self.AZURE_ENDPOINT = ""
        self.AZURE_MODEL_NAME = ""
        self.AZURE_DEPLOYMENT = ""
        self.AZURE_API_KEY = ""
        self.AZURE_API_VERSION = ""
        
        # Document paths
        self.PDF_FILES = [
            "Travel-Ace-PF.pdf",
            "Travel-Ace-Policy-brochure.pdf",
            # "Travel-Ace-Policy-Wordings.pdf",
            # "trave;-ace-int-cies.pdf"
        ]
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self.plan_rag_chain = None
        
    def initialize_models(self):
        """Initialize Azure OpenAI models for embeddings and chat"""
        print("🚀 Initializing Azure OpenAI models...")
        
        # Initialize embeddings model
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.AZURE_ENDPOINT,
            api_key=self.AZURE_API_KEY,
            api_version=self.AZURE_API_VERSION,
            azure_deployment="text-embedding-ada-002",
            chunk_size=1000
        )
        
        # Initialize chat model with better parameters for structured data
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.AZURE_ENDPOINT,
            api_key=self.AZURE_API_KEY,
            api_version=self.AZURE_API_VERSION,
            azure_deployment=self.AZURE_DEPLOYMENT,
            temperature=0.0,  # Zero temperature for consistent factual responses
            max_tokens=2000   # More tokens for detailed table responses
        )
        
        print("✅ Models initialized successfully!")
        
    def extract_section_info(self, text: str) -> Dict[str, Any]:
        """Extract section numbers and coverage information from text"""
        section_info = {
            'sections': [],
            'coverage_types': [],
            'plans_mentioned': [],
            'amounts_mentioned': [],
            'has_table': False
        }
        
        # Extract section numbers
        section_patterns = [
            r'Section (\d+)',
            r'Sub\s+Section (\d+)',
            r'Extension (\d+)',
            r'Section ([A-Z])'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            section_info['sections'].extend(matches)
        
        # Extract plan names
        plan_patterns = [
            r'(Standard|Silver|Gold|Platinum|Super Age|Corporate Lite|Corporate Plus)',
            r'Travel Ace (Standard|Silver|Gold|Platinum|Super Age|Corporate Lite|Corporate Plus)'
        ]
        
        for pattern in plan_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            section_info['plans_mentioned'].extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # Extract coverage types
        coverage_patterns = [
            r'(Medical Exigencies|Personal Accident|Trip Cancellation|Baggage|Laptop|Mobile|Equipment)',
            r'(Home Burglary|Fire and Special Perils|Emergency Dental|Hospitalization)',
            r'(Portable Equipment|Personal Belonging|Personal Liability)'
        ]
        
        for pattern in coverage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            section_info['coverage_types'].extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # Extract monetary amounts
        amount_patterns = [
            r'USD\s*[\d,]+',
            r'INR\s*[\d,]+',
            r'₹\s*[\d,]+',
            r'\$\s*[\d,]+'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            section_info['amounts_mentioned'].extend(matches)
        
        # Check if text contains tabular data
        table_indicators = ['|', 'Standard', 'Silver', 'Gold', 'Platinum', 'Coverage', 'Plan']
        section_info['has_table'] = sum(1 for indicator in table_indicators if indicator in text) >= 3
        
        return section_info
        
    def load_and_process_documents(self) -> List:
        """Load PDFs and split with table-aware chunking"""
        print("📄 Loading and processing PDF documents with table awareness...")
        
        all_docs = []
        
        for pdf_file in self.PDF_FILES:
            if not os.path.exists(pdf_file):
                print(f"⚠️  Warning: {pdf_file} not found, skipping...")
                continue
                
            print(f"   Loading {pdf_file}...")
            
            # Load PDF
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            
            # Enhanced metadata extraction
            for page_num, page in enumerate(pages):
                page.metadata['document_name'] = pdf_file
                page.metadata['document_type'] = 'travel_insurance'
                page.metadata['page_number'] = page_num + 1
                
                # Extract structured information
                section_info = self.extract_section_info(page.page_content)
                page.metadata.update({
                    'sections': section_info['sections'],
                    'coverage_types': section_info['coverage_types'],
                    'plans_mentioned': section_info['plans_mentioned'],
                    'amounts_mentioned': section_info['amounts_mentioned'],
                    'has_table': section_info['has_table']
                })
                
                # Content type classification
                content = page.page_content.lower()
                if 'summary of coverage' in content or 'coverage | plan' in content:
                    page.metadata['content_type'] = 'coverage_table'
                elif 'premium' in content and ('travel days' in content or 'age' in content):
                    page.metadata['content_type'] = 'premium_table'
                elif 'section' in content and any(term in content for term in ['laptop', 'equipment', 'portable']):
                    page.metadata['content_type'] = 'equipment_coverage'
                elif 'home burglary' in content or 'fire and special perils' in content:
                    page.metadata['content_type'] = 'property_coverage'
                elif 'exclusion' in content:
                    page.metadata['content_type'] = 'exclusions'
                elif 'medical' in content and 'exigencies' in content:
                    page.metadata['content_type'] = 'medical_coverage'
                else:
                    page.metadata['content_type'] = 'general'
            
            all_docs.extend(pages)
            print(f"   ✅ Loaded {len(pages)} pages from {pdf_file}")
        
        print(f"📚 Total pages loaded: {len(all_docs)}")
        
        # Advanced chunking strategy for tables and structured data
        print("✂️  Applying table-aware chunking...")
        
        chunks = []
        for doc in all_docs:
            if doc.metadata.get('has_table', False):
                # For documents with tables, use larger chunks to preserve table structure
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,  # Larger chunks for tables
                    chunk_overlap=400,
                    length_function=len,
                    separators=["\n\n\n", "\n\n", "\n", "|", " ", ""]
                )
            else:
                # Standard chunking for regular text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        
        # Enhanced metadata for chunks
        for i, chunk in enumerate(chunks):
            try:
                original_metadata = chunk.metadata.copy() if chunk.metadata else {}
                
                # Re-extract section info for chunk-level metadata
                chunk_section_info = self.extract_section_info(chunk.page_content)
                
                chunk.metadata = {
                    'chunk_id': i,
                    'source': original_metadata.get('source', 'unknown'),
                    'page': original_metadata.get('page', 0),
                    'page_number': original_metadata.get('page_number', 0),
                    'document_name': original_metadata.get('document_name', 'unknown'),
                    'document_type': original_metadata.get('document_type', 'travel_insurance'),
                    'content_type': original_metadata.get('content_type', 'general'),
                    'chunk_length': len(chunk.page_content),
                    'sections': chunk_section_info['sections'],
                    'coverage_types': chunk_section_info['coverage_types'],
                    'plans_mentioned': chunk_section_info['plans_mentioned'],
                    'amounts_mentioned': chunk_section_info['amounts_mentioned'],
                    'has_table': chunk_section_info['has_table'],
                    'word_count': len(chunk.page_content.split())
                }
            except Exception as e:
                print(f"Warning: Error processing chunk {i}: {e}")
                chunk.metadata = {
                    'chunk_id': i,
                    'source': 'unknown',
                    'page': 0,
                    'document_name': 'unknown',
                    'document_type': 'travel_insurance',
                    'content_type': 'general',
                    'chunk_length': len(chunk.page_content)
                }
        
        print(f"✅ Created {len(chunks)} enhanced chunks")
        
        # Debug: Print sample chunk info
        if chunks:
            table_chunks = [c for c in chunks if c.metadata.get('has_table', False)]
            print(f"📊 Table chunks: {len(table_chunks)}")
            print(f"📋 Coverage chunks: {len([c for c in chunks if 'coverage' in c.metadata.get('content_type', '')])}")
            
            # Sample chunk preview
            if table_chunks:
                sample = table_chunks[0]
                print(f"📋 Sample table chunk metadata: {sample.metadata}")
                print(f"📋 Sample table content preview: {sample.page_content[:200]}...")
        
        return chunks
        
    def create_vector_store(self, documents: List):
        """Create vector store with enhanced search capabilities"""
        print("🗄️  Creating enhanced vector store in AstraDB...")
        
        try:
            # Create vector store
            self.vectorstore = AstraDBVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.COLLECTION_NAME,
                token=self.ASTRA_DB_APPLICATION_TOKEN,
                api_endpoint=self.ASTRA_DB_API_ENDPOINT,
            )
            print(f"✅ Successfully created AstraDB collection with {len(documents)} chunks: {self.COLLECTION_NAME}")
            
            # Create retriever with enhanced parameters
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 15,  # Retrieve more documents for comprehensive coverage
                }
            )
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
            
    def expand_query(self, question: str, selected_plan: str = None) -> List[str]:
        """Expand query with related terms and variations"""
        query_variations = [question]
        
        # Add plan-specific variations
        if selected_plan:
            plan_name = selected_plan.replace('Travel Ace ', '').strip()
            query_variations.extend([
                f"{plan_name} {question}",
                f"{question} {plan_name} plan",
                f"Section coverage {question} {plan_name}"
            ])
        
        # Add coverage-specific terms
        coverage_synonyms = {
            'laptop': ['laptop', 'portable equipment', 'equipment', 'electronic device', 'computer'],
            'medical': ['medical', 'medical exigencies', 'sickness', 'illness', 'medical emergency'],
            'baggage': ['baggage', 'luggage', 'checked baggage', 'personal belongings'],
            'trip cancellation': ['trip cancellation', 'cancellation', 'trip interruption'],
            'premium': ['premium', 'cost', 'price', 'rate', 'charges']
        }
        
        for term, synonyms in coverage_synonyms.items():
            if term in question.lower():
                for synonym in synonyms:
                    if synonym not in question.lower():
                        query_variations.append(question.replace(term, synonym))
        
        return query_variations[:5]  # Limit to 5 variations
        
    def multi_retrieval_search(self, question: str, selected_plan: str = None) -> List:
        """Perform multiple retrieval attempts with different strategies"""
        all_docs = []
        seen_chunks = set()
        
        # Expand query variations
        query_variations = self.expand_query(question, selected_plan)
        
        for query in query_variations:
            try:
                # Standard similarity search
                docs = self.vectorstore.similarity_search(query, k=5)
                
                for doc in docs:
                    chunk_id = doc.metadata.get('chunk_id', id(doc))
                    if chunk_id not in seen_chunks:
                        all_docs.append(doc)
                        seen_chunks.add(chunk_id)
                        
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")
                continue
        
        # Sort by relevance (prioritize table chunks and coverage-specific content)
        def relevance_score(doc):
            score = 0
            metadata = doc.metadata
            
            # Boost table chunks
            if metadata.get('has_table', False):
                score += 3
                
            # Boost chunks with plan mentions
            if selected_plan and selected_plan.lower() in str(metadata.get('plans_mentioned', [])).lower():
                score += 2
                
            # Boost chunks with coverage types matching question
            coverage_types = metadata.get('coverage_types', [])
            if any(ctype.lower() in question.lower() for ctype in coverage_types):
                score += 2
                
            # Boost chunks with relevant content types
            content_type = metadata.get('content_type', '')
            if any(term in question.lower() for term in ['laptop', 'equipment']) and 'equipment' in content_type:
                score += 3
            if 'medical' in question.lower() and 'medical' in content_type:
                score += 3
                
            return score
        
        # Sort and return top documents
        all_docs.sort(key=relevance_score, reverse=True)
        return all_docs[:10]  # Return top 10 most relevant
        
    def setup_rag_chains(self):
        """Set up RAG chains optimized for structured insurance data"""
        print("🔗 Setting up enhanced RAG chains...")
        
        # General RAG chain for plan information extraction
        plan_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert travel insurance analyst specializing in Bajaj Allianz Travel Ace policies.

Your task is to extract and present information from insurance policy documents with tables and structured data.

CRITICAL INSTRUCTIONS:
1. **Use ONLY information from the provided context documents**
2. **For tabular data**: Read tables carefully, identify column headers and row data precisely
3. **Section References**: When information mentions "Section X", include the section number in your response
4. **Monetary Values**: Present amounts exactly as shown in documents (USD/INR format)
5. **Plan-Specific Data**: When extracting data for specific plans, check table columns carefully
6. **Coverage Details**: Include specific coverage amounts, limits, and conditions
7. **If information not found**: State clearly "This information is not available in the provided documents"

RESPONSE FORMAT:
- Start with direct answer to the question
- Include specific amounts/limits from tables
- Reference relevant sections when applicable
- Present data in structured format when from tables

Context: {context}"""),
            ("human", "{input}")
        ])
        
        # Plan-specific RAG chain with enhanced table handling
        plan_specific_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized travel insurance assistant for Bajaj Allianz Travel Ace policies.

You are answering questions about the "{selected_plan}" plan specifically.

CRITICAL INSTRUCTIONS FOR {selected_plan}:
1. **Focus EXCLUSIVELY on {selected_plan} plan data from the provided context**
2. **Table Reading**: In coverage tables, find the column for {selected_plan} and extract exact values
3. **Section-Specific**: Reference specific sections (e.g., "Section 33 Sub-Section 2") when mentioned
4. **Exact Amounts**: Extract precise coverage amounts for {selected_plan} from tables
5. **Coverage Verification**: Ensure the coverage exists for {selected_plan} (some are marked "Optional" or "NA")
6. **Multi-Document Context**: Information may span across multiple document chunks - synthesize appropriately

TABLE READING GUIDANCE:
- Look for plan columns: Standard, Silver, Gold, Platinum, Super Age, Corporate Lite, Corporate Plus
- Match the {selected_plan} plan with the correct column
- Extract the exact value from that column
- Note any conditions, deductibles, or limitations

If specific information about {selected_plan} is not in the context, state: "This information for {selected_plan} is not available in the provided documents"

Context: {context}"""),
            ("human", "Question about {selected_plan}: {input}")
        ])
        
        # Create document chains
        general_document_chain = create_stuff_documents_chain(self.llm, plan_extraction_prompt)
        plan_specific_document_chain = create_stuff_documents_chain(self.llm, plan_specific_prompt)
        
        # Create retrieval chains with custom retrieval
        self.rag_chain = create_retrieval_chain(self.retriever, general_document_chain)
        self.plan_rag_chain = create_retrieval_chain(self.retriever, plan_specific_document_chain)
        
        print("✅ Enhanced RAG chains setup complete!")
        
    def get_available_plans(self) -> dict:
        """Extract available plans using enhanced retrieval"""
        try:
            query = """List all Travel Ace insurance plans with their coverage amounts in USD, age groups, and key features. 
            Include Standard, Silver, Gold, Platinum, Super Age, Corporate Lite, and Corporate Plus plans with specific details from coverage tables."""
            
            # Use multi-retrieval for comprehensive plan information
            relevant_docs = self.multi_retrieval_search(query)
            
            # Create a custom context from the most relevant documents
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs[:8]])
            
            response = self.llm.invoke([
                ("system", """Extract comprehensive information about all Travel Ace insurance plans from the provided context. 
                Focus on coverage tables and structured plan data. Present in clear, organized format."""),
                ("human", f"Context: {context_text}\n\nQuery: {query}")
            ])
            
            return {
                'success': True,
                'plans_info': response.content,
                'source_documents': relevant_docs[:5]
            }
        except Exception as e:
            print(f"❌ Error retrieving plans: {e}")
            return {
                'success': False,
                'error': str(e),
                'plans_info': 'Unable to retrieve plan information'
            }
    
    def query_plan_specific(self, question: str, selected_plan: str) -> dict:
        """Query with enhanced multi-retrieval and plan-specific context"""
        if not self.plan_rag_chain:
            raise ValueError("RAG chain not initialized. Call setup() first.")
        
        try:
            print(f"🔍 Enhanced search for: {question} (Plan: {selected_plan})")
            
            # Use multi-retrieval for comprehensive results
            relevant_docs = self.multi_retrieval_search(question, selected_plan)
            
            print(f"📄 Retrieved {len(relevant_docs)} relevant documents")
            
            # Debug: Print retrieved document info
            for i, doc in enumerate(relevant_docs[:5]):
                print(f"Doc {i+1}: {doc.metadata.get('document_name', 'Unknown')} - "
                      f"Page {doc.metadata.get('page', 'Unknown')} - "
                      f"Type: {doc.metadata.get('content_type', 'general')} - "
                      f"Table: {doc.metadata.get('has_table', False)}")
            
            # Enhanced query with plan context
            enhanced_query = f"For the {selected_plan} plan specifically: {question}"
            
            # Manual context preparation for better results
            context_text = "\n\n---DOCUMENT---\n\n".join([doc.page_content for doc in relevant_docs[:8]])
            
            response = self.llm.invoke([
                ("system", f"""You are answering questions about the {selected_plan} plan specifically.

CRITICAL INSTRUCTIONS:
1. Focus ONLY on {selected_plan} plan details from the context
2. For tables, find the {selected_plan} column and extract exact values
3. Include section numbers when referenced
4. Present exact amounts and conditions
5. If information not found, state clearly

Context: {context_text}"""),
                ("human", enhanced_query)
            ])
            
            return {
                'answer': response.content,
                'source_documents': relevant_docs[:8],
                'query': question,
                'selected_plan': selected_plan
            }
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or contact support.",
                'source_documents': [],
                'query': question,
                'selected_plan': selected_plan
            }
    
    def query_general(self, question: str) -> dict:
        """Query with enhanced retrieval for general information"""
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call setup() first.")
        
        try:
            print(f"🔍 Enhanced general search for: {question}")
            
            # Use multi-retrieval
            relevant_docs = self.multi_retrieval_search(question)
            
            print(f"📄 Retrieved {len(relevant_docs)} relevant documents")
            
            # Manual context preparation
            context_text = "\n\n---DOCUMENT---\n\n".join([doc.page_content for doc in relevant_docs[:8]])
            
            response = self.llm.invoke([
                ("system", """Extract information from insurance policy documents with tables and structured data.
                Use only the provided context. For tables, read carefully and extract precise data.
                Include section references and exact amounts."""),
                ("human", f"Context: {context_text}\n\nQuestion: {question}")
            ])
            
            return {
                'answer': response.content,
                'source_documents': relevant_docs[:8],
                'query': question
            }
        except Exception as e:
            print(f"❌ Error processing general query: {e}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question.",
                'source_documents': [],
                'query': question
            }
    
    def setup(self):
        """Complete setup process with enhanced capabilities"""
        print("🏗️  Setting up Enhanced Travel Insurance RAG System...")
        print("=" * 60)
        
        # Initialize models
        self.initialize_models()
        
        # Load and process documents with table awareness
        documents = self.load_and_process_documents()
        
        if not documents:
            raise ValueError("No documents were loaded. Please check your PDF files.")
        
        # Create enhanced vector store
        self.create_vector_store(documents)
        
        # Setup enhanced RAG chains
        self.setup_rag_chains()
        
        print("=" * 60)
        print("🎉 Enhanced RAG System setup complete! Ready for complex queries.")
        print("=" * 60)

# Flask Routes (keeping the existing routes unchanged)

@app.route('/')
def home():
    """Home page with plan selection"""
    return render_template('index.html')

@app.route('/api/plans', methods=['GET'])
def get_plans():
    """Get all available plans from enhanced RAG system"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized',
                'message': 'System is still initializing. Please try again.'
            }), 503
        
        # Get plans information from enhanced RAG system
        plans_result = rag_system.get_available_plans()
        
        if plans_result['success']:
            return jsonify({
                'success': True,
                'plans_info': plans_result['plans_info'],
                'message': 'Plans retrieved successfully from documents'
            })
        else:
            return jsonify({
                'success': False,
                'error': plans_result.get('error', 'Unknown error'),
                'message': 'Failed to retrieve plans from documents'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve plans'
        }), 500

@app.route('/api/select-plan', methods=['POST'])
def select_plan():
    """Select a specific plan and get basic plan confirmation"""
    try:
        data = request.get_json()
        plan_name = data.get('plan_name', '').strip()
        
        if not plan_name:
            return jsonify({
                'success': False,
                'error': 'Plan name required',
                'message': 'Please provide a plan name'
            }), 400
        
        global rag_system
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized',
                'message': 'System is still initializing. Please try again.'
            }), 503
        
        # Store selected plan in session
        session['selected_plan'] = plan_name
        
        # Simple confirmation without detailed overview
        return jsonify({
            'success': True,
            'selected_plan': plan_name,
            'plan_details': f'✅ {plan_name} has been selected successfully. You can now ask detailed questions about this plan\'s coverage, premiums, and features.',
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
    """Query the enhanced RAG system for plan-specific information"""
    try:
        # Check if plan is selected
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
                'error': 'Empty question',
                'message': 'Please provide a question'
            }), 400
        
        # Check if RAG system is initialized
        global rag_system
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized',
                'message': 'System is still initializing. Please try again.'
            }), 503
        
        # Query the enhanced RAG system
        result = rag_system.query_plan_specific(question, selected_plan)
        
        # Process source documents with enhanced metadata
        sources = []
        if result.get('source_documents'):
            for doc in result['source_documents'][:6]:  # Show top 6 sources
                try:
                    source_info = {
                        'document': doc.metadata.get('document_name', 'Unknown Document'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'content_type': doc.metadata.get('content_type', 'general'),
                        'has_table': doc.metadata.get('has_table', False),
                        'sections': doc.metadata.get('sections', []),
                        'coverage_types': doc.metadata.get('coverage_types', []),
                        'content_preview': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                        'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
                    }
                    sources.append(source_info)
                except Exception as e:
                    print(f"Error processing source document: {e}")
                    continue
        
        # Enhanced response data
        response_data = {
            'success': True,
            'answer': result['answer'],
            'question': result['query'],
            'selected_plan': selected_plan,
            'sources': sources,
            'source_count': len(sources),
            'message': 'Query processed successfully with enhanced retrieval'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in enhanced query endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process query'
        }), 500

@app.route('/api/current-plan', methods=['GET'])
def get_current_plan():
    """Get currently selected plan"""
    try:
        selected_plan = session.get('selected_plan')
        
        if not selected_plan:
            return jsonify({
                'success': False,
                'message': 'No plan selected'
            })
        
        return jsonify({
            'success': True,
            'selected_plan': selected_plan,
            'message': 'Current plan retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve current plan'
        }), 500

@app.route('/api/reset-plan', methods=['POST'])
def reset_plan():
    """Reset plan selection"""
    try:
        session.pop('selected_plan', None)
        
        return jsonify({
            'success': True,
            'message': 'Plan selection reset successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to reset plan selection'
        }), 500

@app.route('/api/general-query', methods=['POST'])
def general_query():
    """Query the enhanced RAG system for general travel insurance information"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Empty question',
                'message': 'Please provide a question'
            }), 400
        
        global rag_system
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized',
                'message': 'System is still initializing. Please try again.'
            }), 503
        
        # Query the enhanced RAG system
        result = rag_system.query_general(question)
        
        # Process source documents with enhanced metadata
        sources = []
        if result.get('source_documents'):
            for doc in result['source_documents'][:6]:
                try:
                    source_info = {
                        'document': doc.metadata.get('document_name', 'Unknown Document'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'content_type': doc.metadata.get('content_type', 'general'),
                        'has_table': doc.metadata.get('has_table', False),
                        'sections': doc.metadata.get('sections', []),
                        'coverage_types': doc.metadata.get('coverage_types', []),
                        'content_preview': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                        'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
                    }
                    sources.append(source_info)
                except Exception as e:
                    print(f"Error processing source document: {e}")
                    continue
        
        response_data = {
            'success': True,
            'answer': result['answer'],
            'question': result['query'],
            'sources': sources,
            'source_count': len(sources),
            'message': 'General query processed successfully with enhanced retrieval'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in enhanced general query endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process general query'
        }), 500

# Initialize RAG system function that will be imported by app.py
def initialize_rag_system():
    """Initialize the enhanced RAG system on startup"""
    global rag_system
    try:
        print("🔄 Initializing Enhanced RAG system...")
        rag_system = TravelInsuranceRAG()
        rag_system.setup()
        print("✅ Enhanced RAG system initialized successfully!")
        return rag_system
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced RAG system: {e}")
        return None

if __name__ == '__main__':
    # Initialize enhanced RAG system
    rag_system = initialize_rag_system()
    
    if rag_system:
        # Import chatbot blueprint after initializing RAG system to avoid circular imports
        from chatbot import chatbot_bp, initialize_chatbot
        
        # Initialize chatbot with RAG system
        initialize_chatbot(rag_system)
        
        # Register chatbot blueprint
        app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
        
        print("🚀 Starting Flask application with enhanced RAG capabilities and chatbot...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start application due to Enhanced RAG system initialization error")