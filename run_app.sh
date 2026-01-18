#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
VENV_DIR="./venv"
APP_FILE="./app.py"

# ==========================================
# CLEANUP FUNCTION
# ==========================================
# This function runs when you press Ctrl+C or close the app
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    
    # Check if we started a background Ollama process
    if [ -n "$OLLAMA_PID" ]; then
        echo "killing local Ollama instance (PID: $OLLAMA_PID)..."
        kill $OLLAMA_PID 2>/dev/null
    fi
    
    deactivate 2>/dev/null
    echo "👋 Goodbye!"
    exit
}

# Register the cleanup function to run on exit signals
trap cleanup SIGINT SIGTERM EXIT

# ==========================================
# MAIN EXECUTION
# ==========================================

echo "🚀 Initializing Nutrition Analyzer..."

# 1. Activate Virtual Environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "✅ Virtual environment activated."
else
    echo "❌ Error: Virtual environment directory '$VENV_DIR' not found."
    echo "Please create it first: python3 -m venv venv"
    exit 1
fi

# 2. Install/Update Dependencies
echo "📦 Checking dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dependencies are up to date."
else
    echo "⚠️  Warning: Dependency installation had issues. Proceeding anyway..."
fi

# 3. Start Ollama (Background Process)
# We check if it's already running to avoid conflicts
if curl -s http://localhost:11434 > /dev/null; then
    echo "✅ Ollama is already running globally."
else
    echo "🦙 Starting local Ollama instance..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$! # Capture the Process ID so we can kill it later
    
    # Wait a few seconds for Ollama to wake up
    echo "⏳ Waiting for Ollama to initialize..."
    sleep 5
fi

# 4. Start Streamlit
if [ -f "$APP_FILE" ]; then
    echo "📊 Launching Streamlit..."
    echo "--------------------------------------------------"
    streamlit run "$APP_FILE"
else
    echo "❌ Error: $APP_FILE not found!"
    cleanup
fi