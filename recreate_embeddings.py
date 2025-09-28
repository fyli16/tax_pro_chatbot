import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import time
import tqdm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def extract_text_pymupdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_pdf_text(text):
    # Replace multiple newlines with single spaces
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

# Extract and clean text from PDF
print("📄 Extracting text from PDF...")
pdf_text = extract_text_pymupdf("./i1040gi.pdf")
pdf_text = clean_pdf_text(pdf_text)
print(f"Text length: {len(pdf_text)} characters")

# Split text into chunks
print("✂️ Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.create_documents([pdf_text])
print(f"Created {len(chunks)} chunks")

# Create embeddings
print("🧠 Creating embeddings...")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
print(f"Created {len(vectors)} embeddings")

# Initialize Pinecone
print("🌲 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "tax-rag3"

# Check if index exists, create if not
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI text-embedding-3-small dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Waiting for index to be ready...")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
    print("Index is ready!")

# Connect to index
index_info = pc.describe_index(INDEX_NAME)
index = pc.Index(host=index_info.host)

# Clear existing data (optional) - skip if index is empty
print("🗑️ Checking for existing data...")
stats = index.describe_index_stats()
if stats.total_vector_count > 0:
    print(f"Clearing {stats.total_vector_count} existing vectors...")
    index.delete(delete_all=True)
else:
    print("Index is already empty, proceeding with upload...")

# Upload embeddings
print("📤 Uploading embeddings to Pinecone...")
for i in tqdm.tqdm(range(len(vectors))):
    index.upsert([
        (f"id-{i}", vectors[i], {"text": chunks[i].page_content})
    ])

print("✅ Done! Embeddings uploaded successfully.")

# Verify upload
stats = index.describe_index_stats()
print(f"📊 Index now contains {stats.total_vector_count} vectors")
