from flask import Flask, render_template
import os

# Import from main.py and chatbot.py
from main import app, initialize_rag_system
from chatbot import chatbot_bp, initialize_chatbot

# Register the chatbot blueprint
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')

if __name__ == '__main__':
    # Initialize the RAG system
    rag_system = initialize_rag_system()
    
    if rag_system:
        # Initialize the chatbot with the RAG system
        initialize_chatbot(rag_system)
        
        # Start the Flask application
        print("🚀 Starting Flask application with RAG and chatbot...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start application due to RAG system initialization error")