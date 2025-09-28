from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# List current indexes
print("📋 Current indexes before deletion:")
indexes = pc.list_indexes()
for idx in indexes:
    print(f"  - {idx.name}")

# Delete specific indexes (uncomment the ones you want to delete)
print("\n🗑️ Deleting indexes...")

# Delete old indexes
if 'tax-rag' in [idx.name for idx in indexes]:
    print("Deleting 'tax-rag'...")
    pc.delete_index('tax-rag')

if 'tax-rag2' in [idx.name for idx in indexes]:
    print("Deleting 'tax-rag2'...")
    pc.delete_index('tax-rag2')

# Keep tax-rag3 (your current working index)
print("Keeping 'tax-rag3' (current working index)")

print("\n📋 Remaining indexes:")
remaining_indexes = pc.list_indexes()
for idx in remaining_indexes:
    print(f"  - {idx.name}")
