import os
import sys
import requests
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import time

def init_pinecone(api_key, index_name, pinecone_dimension):
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            deletion_protection="disabled",
            tags={
                "environment": "development"
            }
        )

    print("INDEX:", pc.describe_index(index_name))
    index = pc.Index(host="dense-image-index-qa9q792.svc.aped-4627-b74a.pinecone.io")
    return index


def main_client(index,clip_url="http://localhost:8000", query=None):

    # 3️ Get text query & obtain its embedding from your CLIP server
    if not query:
        query = input("Enter your search query: ").strip()
        if not query:
            sys.exit("No query entered; exiting.")

    try:
        resp = requests.get(
            clip_url,
            params={"text": query},
            timeout=10
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding")
    except Exception as e:
        sys.exit(f"Failed to get embedding from CLIP server: {e}")

    if not embedding:
        sys.exit("CLIP server did not return an embedding.")

    # 4️ Query Pinecone for the top‐1 match (include our stored metadata)
    result = index.query(
        namespace="image_embeddings",
        vector=embedding,
        top_k=1,
        include_metadata=True,
        include_values=False,
    )

    if not result.matches:
        print("No matches found in Pinecone Database.")
        return

    match   = result.matches[0]
    score   = match.score
    meta    = match.metadata or {}
    img_path = meta.get("image_path", match.id)

    # 5️ Load & display the image
    try:
         # if img_path is a filesystem path:
        img = Image.open(img_path)
        img.show()
        print(f"Original query: {query}\n Displayed `{img_path}` (score: {score:.4f})")
    except FileNotFoundError:
        # fallback: maybe it's relative to your images/ directory
        fallback = os.path.join("images", os.path.basename(img_path))
        try:
            img = Image.open(fallback)
            img.show()
            print(f"Original query: {query}\n Displayed `{fallback}` (score: {score:.4f})")
        except Exception as e:
            print(f"Original query: {query}\n Could not open image `{img_path}`: {e}")

if __name__ == "__main__":
    api_key    = os.getenv("PINECONE_API_KEY", "pcsk_3eJ2cq_MYjUEN6HXnQN9X8RTjidvSrKT3AmYsij6oD2URuEccV5r8CyKgz7gNQ6QLf2P7x")
    pinecone_env   = os.getenv("PINECONE_ENV", "us-east1-aws")
    index_name = os.getenv("PINECONE_INDEX", "dense-image-index")
    clip_url   = os.getenv("CLIP_SERVER_URL", "http://localhost:8000/query_embedding")

    print(api_key, pinecone_env, index_name, clip_url)

    if not all(map(len, [api_key, pinecone_env, index_name])):
        sys.exit(
            "Error: please export PINECONE_API_KEY, PINECONE_ENV, and PINECONE_INDEX in your shell."
        )

    # 2️ Initialize Pinecone client
    index = init_pinecone(api_key, index_name, 512)

    # Test with multiple queries
    queries = [
        "A cat sitting on a couch",
        "A dog playing with a ball",
        "A beautiful sunset over the mountains",
        "A person riding a bicycle in the park",
        "A group of friends having a picnic"
    ]
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(main_client, index, clip_url, query) for query in queries]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing query: {e}")

    end_time = time.time()
    print(f"All queries processed in {end_time - start_time:.2f} seconds.")
