from flask import Flask, render_template
import os

# Import from main.py and chatbot.py
from main import app, initialize_rag_system
from chatbot import chatbot_bp, initialize_chatbot

# Register the chatbot blueprint
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')

if __name__ == '__main__':
    # Get port from environment (Render uses PORT env var)
    port = int(os.environ.get("PORT", 5000))
    
    # Initialize the RAG system with better error handling
    print("🔄 Starting application initialization...")
    
    try:
        rag_system = initialize_rag_system()
        
        if rag_system:
            # Initialize the chatbot with the RAG system
            initialize_chatbot(rag_system)
            
            # Start the Flask application
            print(f"🚀 Starting Flask application on port {port}...")
            app.run(
                debug=False,  # Disable debug in production
                host='0.0.0.0', 
                port=port
            )
        else:
            print("❌ RAG system initialization failed, starting with limited functionality...")
            # Start app anyway with basic functionality
            app.run(
                debug=False,
                host='0.0.0.0', 
                port=port
            )
            
    except Exception as e:
        print(f"❌ Application startup error: {e}")
        print("🔄 Attempting to start with minimal functionality...")
        
        # Last resort: start basic Flask app
        app.run(
            debug=False,
            host='0.0.0.0', 
            port=port
        )
