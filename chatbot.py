from flask import Blueprint, request, jsonify, session
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import re
import time

# Create a blueprint instead of a Flask app
chatbot_bp = Blueprint('chatbot', __name__)

# Global RAG system instance (shared with main app)
rag_system = None
chatbot = None

class TravelChatbot:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
        # Azure OpenAI Configuration (using the same configuration as RAG)
        self.AZURE_ENDPOINT = ""
        self.AZURE_MODEL_NAME = ""
        self.AZURE_DEPLOYMENT = ""
        self.AZURE_API_KEY = ""
        self.AZURE_API_VERSION = ""
        
        # Initialize chat model
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.AZURE_ENDPOINT,
            api_key=self.AZURE_API_KEY,
            api_version=self.AZURE_API_VERSION,
            azure_deployment=self.AZURE_DEPLOYMENT,
            temperature=0.7,  # Slightly higher temperature for more natural conversations
            max_tokens=1000
        )
        
        # Store conversation history
        self.conversation_history = {}
        
        # Plan recommendation rules
        self.plan_rules = {
            'Travel Ace Standard': {
                'description': 'Basic coverage for short trips to low-risk destinations',
                'recommended_for': 'Budget travelers, short trips to Asian countries',
                'key_features': 'Basic medical coverage, baggage loss, trip cancellation'
            },
            'Travel Ace Silver': {
                'description': 'Mid-level coverage for longer trips with additional benefits',
                'recommended_for': 'Regular travelers, trips between 15-30 days',
                'key_features': 'Enhanced medical coverage, electronics protection, trip delay'
            },
            'Travel Ace Gold': {
                'description': 'Comprehensive coverage for international travel with high limits',
                'recommended_for': 'Frequent travelers to USA, Europe or high-cost regions',
                'key_features': 'High medical coverage limits, adventure sports coverage, extended electronics protection'
            },
            'Travel Ace Platinum': {
                'description': 'Premium all-inclusive coverage with highest limits and benefits',
                'recommended_for': 'Premium travelers, business executives, families',
                'key_features': 'Maximum coverage on all benefits, lowest deductibles, VIP assistance'
            },
            'Travel Ace Super Age': {
                'description': 'Specialized coverage for senior travelers with focus on medical',
                'recommended_for': 'Travelers aged 61+ to any destination',
                'key_features': 'Enhanced medical coverage, pre-existing condition coverage, prescription assistance'
            },
            'Travel Ace Corporate Lite': {
                'description': 'Basic business travel coverage for employees',
                'recommended_for': 'Companies with frequent employee travel needs',
                'key_features': 'Basic business equipment coverage, meeting cancellation, work trip essentials'
            },
            'Travel Ace Corporate Plus': {
                'description': 'Premium business travel coverage with executive benefits',
                'recommended_for': 'Executives and VIP corporate travelers',
                'key_features': 'Premium business equipment coverage, executive assistance, VIP medical'
            }
        }
    
    def get_session_id(self):
        """Get a unique session ID for the current user"""
        if 'chat_session_id' not in session:
            session['chat_session_id'] = f"chat_{int(time.time())}"
        return session['chat_session_id']
    
    def add_to_history(self, session_id, role, content):
        """Add a message to the conversation history"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            "role": role,
            "content": content
        })
    
    def get_conversation_history(self, session_id):
        """Get the conversation history for a session"""
        return self.conversation_history.get(session_id, [])
    
    def reset_conversation(self, session_id):
        """Reset the conversation history"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
    
    def recommend_plan(self, user_preferences):
        """
        Recommend a travel insurance plan based on user preferences
        
        user_preferences format:
        {
            "destination": "USA",
            "duration": "14 days",
            "age": "35",
            "coverage": "Medical coverage"
        }
        """
        # Extract key information
        destination = user_preferences.get('destination', '').lower()
        duration_text = user_preferences.get('duration', '').lower()
        age_text = user_preferences.get('age', '0')
        coverage_pref = user_preferences.get('coverage', '').lower()
        
        # Parse duration to days
        days = 0
        if 'day' in duration_text:
            days = int(re.search(r'(\d+)', duration_text).group(1))
        elif 'week' in duration_text:
            weeks = int(re.search(r'(\d+)', duration_text).group(1))
            days = weeks * 7
        elif 'month' in duration_text:
            months = int(re.search(r'(\d+)', duration_text).group(1))
            days = months * 30
        
        # Parse age
        try:
            age = int(re.search(r'(\d+)', age_text).group(1))
        except:
            age = 35  # Default age if parsing fails
        
        # Recommendation logic
        if age > 60:
            return "Travel Ace Super Age"
        
        if 'business' in coverage_pref:
            if 'executive' in coverage_pref or 'vip' in coverage_pref:
                return "Travel Ace Corporate Plus"
            else:
                return "Travel Ace Corporate Lite"
        
        high_cost_destinations = ['usa', 'united states', 'america', 'europe', 'canada', 'australia']
        is_high_cost = any(dest in destination for dest in high_cost_destinations)
        
        if is_high_cost:
            if 'all' in coverage_pref or 'premium' in coverage_pref:
                return "Travel Ace Platinum"
            else:
                return "Travel Ace Gold"
        else:
            if days > 15:
                return "Travel Ace Silver"
            else:
                return "Travel Ace Standard"
    
    def get_plan_details(self, plan_name):
        """Get detailed information about a specific plan"""
        try:
            # First, try to get basic plan details from our rules
            basic_details = self.plan_rules.get(plan_name, {})

            # Then, use the RAG system to get comprehensive plan details
            if self.rag_system:
                # Create a query to get plan details
                query = f"Provide a comprehensive summary of the {plan_name} plan, including coverage amounts, benefits, and key features."

                # Use the existing RAG system to get plan details
                try:
                    result = self.rag_system.query_plan_specific(query, plan_name)

                    # Process source documents to make them JSON serializable
                    serializable_sources = []
                    if 'source_documents' in result:
                        for doc in result['source_documents']:
                            if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                                serializable_sources.append({
                                    'document': doc.metadata.get('document_name', ''),
                                    'page': doc.metadata.get('page_number', ''),
                                    'content_type': doc.metadata.get('content_type', ''),
                                    'content_preview': doc.page_content[:200] if hasattr(doc, 'page_content') else ''
                                })

                    # Combine our basic details with RAG results
                    combined_details = f"""
                    📋 Description:
                    • {basic_details.get('description', '').strip()}

                    💡 Plan Details:
                    {result.get('answer', '').strip().replace('\n', '\n• ')}

                    👥 Recommended for:
                    • {basic_details.get('recommended_for', '').strip()}

                    ✨ Key Features:
                    • {basic_details.get('key_features', '').strip().replace(', ', '\n• ')}
                    """

                    return {
                        'success': True,
                        'plan_name': plan_name,
                        'plan_details': combined_details.strip(),
                        'sources': serializable_sources
                    }
                except Exception as e:
                    print(f"Error getting RAG details: {e}")
                    # Fall back to basic details if RAG fails
                    return {
                        'success': True,
                        'plan_name': plan_name,
                        'plan_details': f"""
                        {basic_details.get('description', '')}

                        Recommended for: {basic_details.get('recommended_for', '')}
                        Key features: {basic_details.get('key_features', '')}
                        """.strip(),
                        'sources': []
                    }
            else:
                # If RAG system is not available, use basic details
                return {
                    'success': True,
                    'plan_name': plan_name,
                    'plan_details': f"""
                    {basic_details.get('description', '')}

                    Recommended for: {basic_details.get('recommended_for', '')}
                    Key features: {basic_details.get('key_features', '')}
                    """.strip(),
                    'sources': []
                }
        except Exception as e:
            print(f"Error getting plan details: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve plan details'
            }
            
    def analyze_objection(self, objection, current_plan, user_preferences):
        """Analyze user objection and suggest alternative plan"""
        try:
            # Create a prompt to analyze the objection
            analysis_prompt = f"""
            The user has concerns about the {current_plan} plan.
            User's objection: {objection}
            User preferences: {user_preferences}

            Please analyze the objection and suggest the most suitable alternative plan.
            Focus on addressing the specific concerns while maintaining coverage needs.
            """

            # Use RAG to get alternative plan suggestions
            if self.rag_system:
                result = self.rag_system.query_general(analysis_prompt)
                
                # Extract key concerns and find better matching plan
                concerns = self.extract_concerns(objection)
                alternative_plan = self.find_alternative_plan(concerns, current_plan, user_preferences)
                
                return {
                    'success': True,
                    'analysis': result.get('answer', 'Based on your concerns, let me suggest an alternative plan.'),
                    'suggested_plan': alternative_plan
                }
            
            return {
                'success': True,
                'analysis': 'I understand your concerns. Let me suggest a different plan that might better suit your needs.',
                'suggested_plan': self.find_alternative_plan([], current_plan, user_preferences)
            }
            
        except Exception as e:
            print(f"Error analyzing objection: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to analyze concerns'
            }

    def extract_concerns(self, objection):
        """Extract main concerns from user objection"""
        concerns = {
            'cost': ['expensive', 'costly', 'price', 'premium'],
            'coverage': ['coverage', 'benefits', 'not enough', 'limited'],
            'restrictions': ['restrictions', 'limitations', 'age', 'excluded'],
            'features': ['features', 'missing', 'need', 'want']
        }
        
        user_concerns = []
        for category, keywords in concerns.items():
            if any(keyword in objection.lower() for keyword in keywords):
                user_concerns.append(category)
        
        return user_concerns

    def find_alternative_plan(self, concerns, current_plan, user_preferences):
        """Find alternative plan based on concerns and preferences"""
        if 'cost' in concerns:
            # Suggest next lower tier plan
            plan_tiers = ['Travel Ace Platinum', 'Travel Ace Gold', 'Travel Ace Silver', 'Travel Ace Standard']
            try:
                current_index = plan_tiers.index(current_plan)
                if current_index < len(plan_tiers) - 1:
                    return plan_tiers[current_index + 1]
            except ValueError:
                pass
        
        if 'coverage' in concerns or 'features' in concerns:
            # Suggest next higher tier plan
            plan_tiers = ['Travel Ace Standard', 'Travel Ace Silver', 'Travel Ace Gold', 'Travel Ace Platinum']
            try:
                current_index = plan_tiers.index(current_plan)
                if current_index > 0:
                    return plan_tiers[current_index - 1]
            except ValueError:
                pass
        
        # Default logic based on user preferences
        destination = user_preferences.get('destination', '').lower()
        age = int(user_preferences.get('age', '0'))
        
        if age > 60:
            return "Travel Ace Super Age"
        if 'business' in user_preferences.get('coverage', '').lower():
            return "Travel Ace Corporate Plus"
        if any(dest in destination for dest in ['usa', 'europe']):
            return "Travel Ace Gold"
        
        return "Travel Ace Silver"

    def process_message(self, message, selected_plan=None):
        """Process a message from the user and generate a response"""
        try:
            session_id = self.get_session_id()
            
            # Add user message to history
            self.add_to_history(session_id, "user", message)
            
            # Get the conversation history
            history = self.get_conversation_history(session_id)
            
            # Check for negative feedback or objections about selected plan
            negative_indicators = ['expensive', 'costly', 'high', 'not good', 'don\'t like', 'don\'t want',
                                'too much', 'bad', 'poor', 'terrible', 'awful', 'worse', 'worst',
                                'limited', 'insufficient', 'inadequate', 'restrict']
            
            if selected_plan and any(indicator in message.lower() for indicator in negative_indicators):
                # Handle objection for the selected plan
                result = self.analyze_objection(message, selected_plan, {})
                if result['success']:
                    alternative_plan = result['suggested_plan']
                    response = f"{result['analysis']}\n\nWould you like to see details of the {alternative_plan} plan?"
                    self.add_to_history(session_id, "assistant", response)
                    return {
                        'success': True,
                        'response': response,
                        'suggested_plan': alternative_plan,
                        'options': [f"Show me {alternative_plan} details", "Keep current plan", "Explore other options"]
                    }
            
            # Check if this is the first message
            if len(history) <= 1:
                if message.lower() in ['hi', 'hello', 'hey']:
                    response = "I am an assistant to help you choose your travel plan. I can guide you through selecting the right plan based on your needs."
                    self.add_to_history(session_id, "assistant", response)
                    return {
                        'success': True,
                        'response': response,
                        'options': ["I need help choosing a plan", "Tell me about available plans", "What questions do you have?"]
                    }
            
            # Check for plan-related queries
            if any(keyword in message.lower() for keyword in ['show', 'tell', 'what', 'details', 'benefits']) and \
               any(keyword in message.lower() for keyword in ['plan', 'policy', 'package']):
                # Extract plan name if mentioned
                for plan_name in self.plan_rules.keys():
                    if plan_name.lower() in message.lower():
                        result = self.get_plan_details(plan_name)
                        if result['success']:
                            self.add_to_history(session_id, "assistant", result['plan_details'])
                            return result

            # If a plan is already selected, use RAG to answer specific questions
            if selected_plan:
                try:
                    result = self.rag_system.query_plan_specific(message, selected_plan)
                    response = result.get('answer', "I couldn't find specific information about that in the selected plan.")
                    self.add_to_history(session_id, "assistant", response)
                    return {
                        'success': True,
                        'response': response,
                        'options': ["Show full plan details", "Compare with other plans", "I have concerns about this plan"]
                    }
                except Exception as e:
                    print(f"Error using RAG: {e}")
            
            # Create a prompt for the language model
            system_prompt = """You are a helpful travel insurance assistant for Bajaj Allianz.
            You help users choose the right travel insurance plan based on their needs.
            Be conversational, friendly, and informative.
            
            Available plans include:
            - Travel Ace Standard: Basic coverage for short trips to low-risk destinations
            - Travel Ace Silver: Mid-level coverage with better benefits
            - Travel Ace Gold: Comprehensive coverage for international travel
            - Travel Ace Platinum: Premium all-inclusive coverage
            - Travel Ace Super Age: Specialized coverage for senior travelers
            - Travel Ace Corporate Lite: Basic business travel coverage
            - Travel Ace Corporate Plus: Premium business travel coverage
            
            If the user asks about coverage details you don't know, suggest they select a plan to see the specifics."""
            
            # Prepare messages for the LLM
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (limit to last 10 messages)
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Add assistant response to history
            self.add_to_history(session_id, "assistant", response_text)
            
            # Generate options based on the conversation state
            options = []
            
            # Add suggestion options based on context
            if "plan" in message.lower() and "recommend" in message.lower():
                options = ["Tell me about available plans", "What documents do I need for claims?", "How do I compare plans?"]
            elif "coverage" in message.lower():
                options = ["Medical coverage", "Baggage coverage", "Trip cancellation"]
            elif "destination" in message.lower():
                options = ["USA", "Europe", "Asia", "Australia"]
            
            return {
                'success': True,
                'response': response_text,
                'options': options
            }
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to process message'
            }


# Route for chatbot messages
@chatbot_bp.route('/message', methods=['POST'])
def process_message():
    """Process a message from the user"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Empty message',
                'message': 'Please provide a message'
            }), 400
        
        # Get the selected plan from session if available
        selected_plan = session.get('selected_plan')
        
        # Process the message
        result = chatbot.process_message(message, selected_plan)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing chatbot message: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process message'
        }), 500

# Route for getting plan details
@chatbot_bp.route('/get_plan_details', methods=['POST'])
def get_plan_details():
    """Get details for a specific plan"""
    try:
        data = request.get_json()
        plan_name = data.get('plan_name', '').strip()
        
        if not plan_name:
            return jsonify({
                'success': False,
                'error': 'Plan name required',
                'message': 'Please provide a plan name'
            }), 400
        
        # Get plan details
        result = chatbot.get_plan_details(plan_name)
        
        # Store selected plan in session
        session['selected_plan'] = plan_name
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error getting plan details: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get plan details'
        }), 500

# Route for analyzing objections and suggesting alternatives
@chatbot_bp.route('/analyze_objection', methods=['POST'])
def analyze_objection():
    """Analyze user objection and suggest alternative plan"""
    try:
        data = request.get_json()
        objection = data.get('objection', '').strip()
        current_plan = data.get('current_plan', '').strip()
        user_preferences = data.get('user_preferences', {})
        
        if not objection or not current_plan:
            return jsonify({
                'success': False,
                'error': 'Missing required information',
                'message': 'Please provide objection and current plan'
            }), 400
        
        # Analyze objection and get suggestion
        result = chatbot.analyze_objection(objection, current_plan, user_preferences)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error analyzing objection: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to analyze objection'
        }), 500

# Route for resetting conversation
@chatbot_bp.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation history"""
    try:
        session_id = chatbot.get_session_id()
        chatbot.reset_conversation(session_id)
        
        # Also reset selected plan
        if 'selected_plan' in session:
            session.pop('selected_plan')
        
        return jsonify({
            'success': True,
            'message': 'Conversation reset successfully'
        })
        
    except Exception as e:
        print(f"Error resetting conversation: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to reset conversation'
        }), 500

def initialize_chatbot(rag_system_instance):
    """Initialize the chatbot with the RAG system instance"""
    global chatbot
    global rag_system
    
    rag_system = rag_system_instance
    chatbot = TravelChatbot(rag_system)
    print("✅ Chatbot initialized successfully!")