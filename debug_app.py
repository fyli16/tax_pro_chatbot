import streamlit as st
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# load api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "tax-rag2"

client = OpenAI()
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Reconnect to your existing index
index_info = pc.describe_index(INDEX_NAME)
index = pc.Index(host=index_info.host)

def retrieve_context(query):
    embedded_query = embedding_model.embed_query(query)
    results = index.query(vector=embedded_query, top_k=5, include_metadata=True)
    return results['matches']

def generate_answer(query):
    matches = retrieve_context(query)
    context_parts = [match['metadata']['text'] for match in matches]
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a tax assistant. Answer based only on the following documents:\n\n{context}\n\nQ: {query}\nA:"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using a more reliable model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.title("Tax Assistant Debug Mode")

query = st.text_input("Ask me a tax question:")
if query:
    with st.spinner("Searching..."):
        # Debug: Show retrieval results
        matches = retrieve_context(query)
        
        st.write("### Debug Information:")
        st.write(f"**Number of matches found:** {len(matches)}")
        
        if len(matches) > 0:
            st.write("**Top match scores:**")
            for i, match in enumerate(matches[:3]):
                st.write(f"Match {i+1}: Score = {match['score']:.4f}")
                st.write(f"Text preview: {match['metadata']['text'][:200]}...")
                st.write("---")
            
            # Show full context being sent to GPT
            context_parts = [match['metadata']['text'] for match in matches]
            context = "\n\n".join(context_parts)
            st.write("**Full context being sent to GPT:**")
            st.text_area("Context", context, height=300)
            
            # Generate answer
            answer = generate_answer(query)
            st.write("**Answer:**", answer)
        else:
            st.error("No matches found in the vector database!")
            st.write("This could mean:")
            st.write("1. The index is empty")
            st.write("2. The embeddings weren't created properly")
            st.write("3. The query embedding is too different from stored embeddings")
