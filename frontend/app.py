"""
Door Specifications RAG - Streamlit Frontend
Navy Blue & White themed interface with VTX branding
"""
import streamlit as st
import requests
import os
import base64

# Page configuration
st.set_page_config(
    page_title="Door Specifications Assistant",
    page_icon="🚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'input_value' not in st.session_state:
    st.session_state.input_value = ""
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

# Backend URL - use API_URL from docker-compose or default to localhost
BACKEND_URL = os.getenv("API_URL", os.getenv("BACKEND_URL", "http://localhost:8000"))

# Custom CSS for Navy Blue theme
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1e3a5f;
    }
    .message-assistant {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Navy Blue buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c4f7c 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2c4f7c 0%, #3d5a80 100%);
        box-shadow: 0 4px 8px rgba(30, 58, 95, 0.3);
        transform: translateY(-1px);
    }
    /* Primary button (Search) - Navy Blue */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c4f7c 100%) !important;
        color: white !important;
        border: 2px solid #1e3a5f !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2c4f7c 0%, #3d5a80 100%) !important;
    }
    /* Secondary button (Clear) - White with Navy border */
    button[kind="secondary"] {
        background: white !important;
        color: #1e3a5f !important;
        border: 2px solid #1e3a5f !important;
    }
    button[kind="secondary"]:hover {
        background: #f0f4f8 !important;
        color: #2c4f7c !important;
        border: 2px solid #2c4f7c !important;
    }
    /* Sidebar buttons */
    .css-1d391kg button, [data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c4f7c 100%);
        color: white;
        border: none;
    }
    .css-1d391kg button:hover, [data-testid="stSidebar"] button:hover {
        background: linear-gradient(135deg, #2c4f7c 0%, #3d5a80 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Function to query the backend
def query_backend(query_text):
    """Send query to backend API"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/query",
            json={"query": query_text},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Error: Server returned status {response.status_code}",
                "confidence": "Error",
                "confidence_score": 0.0,
                "sources": [],
                "conflicts": []
            }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "⚠️ Cannot connect to backend. Please ensure the backend is running on port 8000.",
            "confidence": "Error",
            "confidence_score": 0.0,
            "sources": [],
            "conflicts": []
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence": "Error",
            "confidence_score": 0.0,
            "sources": [],
            "conflicts": []
        }

# Function to handle search
def perform_search(query_text):
    """Perform search and add to messages"""
    if query_text and query_text.strip():
        st.session_state.messages.append({"role": "user", "content": query_text})

        with st.spinner("🔍 Searching documents..."):
            response = query_backend(query_text)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.query_count += 1

        # Clear the input
        st.session_state.input_value = ""
        st.session_state.clear_input = True

# Function to clear chat
def clear_chat():
    """Clear all messages"""
    st.session_state.messages = []
    st.session_state.input_value = ""

# Sidebar with simplified functionalities
with st.sidebar:
    # Logo - the files are mounted at frontend/vtx_logo1.png from /app directory
    logo_path = "frontend/vtx_logo2.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        # Fallback to check script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_full_path = os.path.join(script_dir, "vtx_logo2.png")
        if os.path.exists(logo_full_path):
            st.image(logo_full_path, width=150)

  

    # System Stats
    st.subheader("System Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", "670+")
    with col2:
        st.metric("Total Queries", st.session_state.query_count)

    st.divider()

    # Quick Door Lookup
    st.subheader("Quick Door Lookup")
    door_number = st.text_input("Enter Door Number:", placeholder="e.g., 148A", key="sidebar_door")
    if st.button("🔍 Look Up Door", type="primary", use_container_width=True):
        if door_number:
            query = f"What are the specifications for door {door_number}?"
            perform_search(query)
            st.rerun()

    st.divider()

    # Sample Queries
    st.subheader("Sample Queries")
    sample_queries = [
        "What are the fire rating requirements?",
        "List all doors with 90 MIN rating",
        "What materials are used for doors?"
    ]

    for idx, query in enumerate(sample_queries):
        if st.button(query, key=f"sample_{idx}", use_container_width=True):
            perform_search(query)
            st.rerun()

    st.divider()

    # API Status Check
    st.subheader("System Status")
    try:
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ Backend Online")
        else:
            st.error("❌ Backend Error")
    except:
        st.warning("⚠️ Backend Offline")

# Main content area with logo inside header container
# Check if logo exists and encode it
logo1_path = "frontend/vtx_logo1.png"
logo_html = ""
if os.path.exists(logo1_path):
    import base64
    with open(logo1_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: 80px; margin-right: 20px; vertical-align: middle;">'

# Header with logo on far left
st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2c4f7c 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                justify-content: space-between;">
        <div style="display: flex; align-items: center;">
            {logo_html}
        </div>
        <div style="flex-grow: 1; text-align: center; margin-right: 80px;">
            <h1 style="margin: 0; color: white;">🚪 Door Specifications RAG System</h1>
            <p style="margin: 5px 0 0 0; color: #e0e0e0;">Intelligent search across door documentation</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Search Container with horizontally aligned input and buttons
st.markdown("### 🔍 Search Documents")
col1, col2, col3 = st.columns([7, 1.5, 1.5])

with col1:
    query_input = st.text_input(
        label="Search",
        value=st.session_state.input_value if not st.session_state.clear_input else "",
        placeholder="Ask about door specifications... (e.g., What are the specifications for door 148A?)",
        label_visibility="collapsed",
        key="query_input"
    )
    if st.session_state.clear_input:
        st.session_state.clear_input = False
        st.session_state.input_value = ""

with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

with col3:
    clear_button = st.button("🗑️ Clear", type="secondary", use_container_width=True)

st.markdown("---")  # Separator line

# Handle button clicks
if search_button and query_input:
    perform_search(query_input)
    st.rerun()

if clear_button:
    clear_chat()
    st.rerun()

# Handle Enter key press automatically through text_input's on_change
if query_input and query_input != st.session_state.input_value:
    st.session_state.input_value = query_input

# Display chat messages
if st.session_state.messages:
    st.markdown("### 💬 Conversation History")

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="message-user"><strong>👤 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            response = message["content"]

            # Format the response with better structure
            answer_html = f'<div class="message-assistant"><strong>🤖 Assistant:</strong><br><br>'

            # Add the answer
            answer_text = response.get('answer', 'No answer available')
            # Convert markdown-style formatting to HTML
            answer_text = answer_text.replace('\n', '<br>')
            answer_text = answer_text.replace('**', '<strong>').replace('**', '</strong>')
            answer_html += answer_text

            # Add confidence if available
            if 'confidence' in response and response['confidence'] != 'Error':
                confidence = response['confidence']
                score = response.get('confidence_score', 0.0)
                color = "#4caf50" if confidence == "High" else "#ff9800" if confidence == "Medium" else "#f44336"
                answer_html += f'<br><br>📊 <strong>Confidence:</strong> <span style="color: {color}">{confidence} ({score:.0%})</span>'

            # Add sources if available
            if 'sources' in response and response['sources']:
                answer_html += '<br><br>📚 <strong>Sources:</strong><ul style="margin-top: 0.5rem;">'
                for source in response['sources']:
                    answer_html += f'<li>{source}</li>'
                answer_html += '</ul>'

            # Add conflicts if available
            if 'conflicts' in response and response['conflicts']:
                answer_html += '<br><br>⚠️ <strong>Conflicts Detected:</strong><ul style="margin-top: 0.5rem; color: #ff9800;">'
                for conflict in response['conflicts']:
                    answer_html += f'<li>{conflict}</li>'
                answer_html += '</ul>'

            answer_html += '</div>'
            st.markdown(answer_html, unsafe_allow_html=True)
else:
    # Show welcome message when no messages
    st.info("""
    👋 **Welcome to the Door Specifications RAG System!**

    You can:
    - Type a question in the search box above and click **Search**
    - Use the **Quick Door Lookup** in the sidebar for specific door numbers
    - Try one of the **Sample Queries** in the sidebar
    - Click **Clear** to start a new conversation

    **Example queries:**
    - What are the specifications for door 148A?
    - What fire rating standards are used?
    - List all doors with 90 MIN fire rating
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
    🚪 Door Specifications RAG System | Powered by OpenAI & Qdrant<br>
    📚 670+ document chunks indexed | 🔍 Semantic search with entity recognition<br>
        Developed by MLawali@versatexmsp.com | © 2025 All rights reserved.
    </small>
</div>
""", unsafe_allow_html=True)