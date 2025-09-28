from pinecone import Pinecone
import os
from dotenv import load_dotenv

# load api key
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "tax-rag2"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
try:
    index_info = pc.describe_index(INDEX_NAME)
    print(f"✅ Index '{INDEX_NAME}' exists")
    print(f"Index status: {index_info.status}")
    print(f"Index dimension: {index_info.dimension}")
    print(f"Index metric: {index_info.metric}")
    
    # Connect to index
    index = pc.Index(host=index_info.host)
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"\n📊 Index Statistics:")
    print(f"Total vector count: {stats.total_vector_count}")
    print(f"Index dimension: {stats.dimension}")
    
    if stats.total_vector_count > 0:
        print("\n✅ Index has data!")
        
        # Try a simple query to see if it works
        print("\n🔍 Testing a simple query...")
        test_query = [0.1] * stats.dimension  # Create a dummy vector
        results = index.query(vector=test_query, top_k=1, include_metadata=True)
        print(f"Query returned {len(results['matches'])} matches")
        
        if len(results['matches']) > 0:
            print("Sample metadata:", results['matches'][0]['metadata'])
        else:
            print("No matches found for test query")
    else:
        print("\n❌ Index is empty! No vectors found.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("This could mean:")
    print("1. The index doesn't exist")
    print("2. The index name is wrong")
    print("3. There's an API key issue")
