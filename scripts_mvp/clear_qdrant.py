import qdrant_client

# Initialize the Qdrant client
# If your Qdrant instance is running on the default port, the URL is http://localhost:6333
client = qdrant_client.QdrantClient(url="http://localhost:6333")

try:
    # Get the list of all collections
    collections_response = client.get_collections()
    
    # The actual list of collections is in the 'collections' attribute
    collections = collections_response.collections
    
    if not collections:
        print("No collections found.")
    else:
        print(f"Found {len(collections)} collections. Deleting them now...")
        
        # Iterate over the collections and delete each one
        for collection in collections:
            collection_name = collection.name
            try:
                delete_result = client.delete_collection(collection_name=collection_name)
                if delete_result:
                    print(f"Collection '{collection_name}' deleted successfully.")
                else:
                    print(f"Failed to delete collection '{collection_name}'.")
            except Exception as e:
                print(f"An error occurred while deleting collection '{collection_name}': {e}")
                
    print("All collections have been deleted.")

except Exception as e:
    print(f"An error occurred: {e}")