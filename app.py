import streamlit as st
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI


# load api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "tax-rag"

client = OpenAI()

embedding_model = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

# Reconnect to your existing index
index_info = pc.describe_index(INDEX_NAME)
index = pc.Index(host=index_info.host)

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

st.title("Chatbot")

query = st.text_input("Ask me a tax question:")
if query:
    with st.spinner("Searching..."):
        answer = generate_answer(query)
        st.write("**Answer:**", answer)