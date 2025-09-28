from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# List all indexes
print("📋 Current indexes:")
indexes = pc.list_indexes()
for idx in indexes:
    print(f"  - {idx.name}")

if len(indexes) == 0:
    print("No indexes found.")
else:
    print(f"\nTotal indexes: {len(indexes)}")
    
    # Ask which index to delete
    print("\n🗑️ To delete an index, uncomment the line below and specify the index name:")
    for idx in indexes:
        print(f"# pc.delete_index('{idx.name}')")
    
    # Example: Delete a specific index
    # pc.delete_index('tax-rag2')
    
    print("\n⚠️  Warning: Deleting an index is permanent and cannot be undone!")
    print("All vectors and metadata in the deleted index will be lost.")
