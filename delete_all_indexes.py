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

if len(indexes) == 0:
    print("No indexes to delete.")
else:
    # Confirm deletion
    print(f"\n⚠️  WARNING: This will delete ALL {len(indexes)} indexes!")
    print("This action is permanent and cannot be undone.")
    
    # Uncomment the line below to actually delete all indexes
    # for idx in indexes:
    #     print(f"Deleting {idx.name}...")
    #     pc.delete_index(idx.name)
    
    print("\nTo actually delete all indexes, uncomment the deletion code above.")
    
    print("\n📋 Indexes remain unchanged:")
    remaining_indexes = pc.list_indexes()
    for idx in remaining_indexes:
        print(f"  - {idx.name}")
