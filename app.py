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
INDEX_NAME = "tax-rag3"
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

def generate_answer(query):
    context = "\n\n".join(retrieve_context(query))
    prompt = f"""You are a tax assistant. Answer based only on the following documents:\n\n{context}\n\nQ: {query}\nA:"""
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.title("Tax Filing Chatbot 2025")

query = st.text_input("Ask me a tax question:")
if query:
    with st.spinner("Searching..."):
        answer = generate_answer(query)
        st.write("**Answer:**", answer)