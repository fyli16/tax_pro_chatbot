import streamlit as st
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# load api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]

# Reconnect to your existing index
INDEX_NAME = "tax-rag"
pc = Pinecone(api_key=PINECONE_API_KEY)
index_info = pc.describe_index(INDEX_NAME)
index = pc.Index(host=index_info.host)

embedding_model = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
client = OpenAI()

# Build the Retrieval-Augmented Generation (RAG) pipeline
def retrieve_context(query):
    embedded_query = embedding_model.embed_query(query)
    results = index.query(vector=embedded_query, top_k=5, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_answer(query, conversation_history=None):
    context = "\n\n".join(retrieve_context(query))
    
    # Build conversation context
    if conversation_history:
        # Include previous conversation for context
        conversation_text = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in conversation_history])
        prompt = f"""You are a tax assistant. Answer based only on the following documents:\n\n{context}\n\nPrevious conversation:\n{conversation_text}\n\nCurrent question: {query}\n\nPlease provide a helpful answer based on the tax documents:"""
    else:
        prompt = f"""You are a tax assistant. Answer based only on the following documents:\n\n{context}\n\nQ: {query}\nA:"""
    
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Tax Filing Chatbot")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me a tax question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching tax documents..."):
            # Prepare conversation history for context
            conversation_history = []
            for i in range(0, len(st.session_state.messages) - 1, 2):  # Skip the current message
                if i + 1 < len(st.session_state.messages):
                    conversation_history.append({
                        'user': st.session_state.messages[i]['content'],
                        'assistant': st.session_state.messages[i + 1]['content']
                    })
            
            response = generate_answer(prompt, conversation_history)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Add some helpful information in the sidebar
st.sidebar.markdown("### About this Chatbot")
st.sidebar.markdown("""
This chatbot can help you with questions about:
- Form 1040 and 1040-SR filing
- Tax filing status
- Deductions and credits
- Tax deadlines and requirements
- And more!

The bot uses the official IRS instructions for tax filing.
""")