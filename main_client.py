import os
import sys
import requests
import pinecone
from PIL import Image
from io import BytesIO


def main_client():

    # 1️ Load & validate environment variables
    api_key    = os.getenv("pcsk_3eJ2cq_MYjUEN6HXnQN9X8RTjidvSrKT3AmYsij6oD2URuEccV5r8CyKgz7gNQ6QLf2P7x")
    pinecone_env   = os.getenv("us-east1-aws")
    index_name = os.getenv("dense-image-index")
    clip_url   = os.getenv("CLIP_SERVER_URL", "http://localhost:8000/query_embedding")

    if not all([api_key, pinecone_env, index_name]):
        sys.exit(
            "Error: please export PINECONE_API_KEY, PINECONE_ENV, and PINECONE_INDEX in your shell."
        )

    # 2️ Initialize Pinecone client
    pinecone.init(api_key=api_key, environment=pinecone_env)
    index = pinecone.Index(index_name)

    # 3️ Get text query & obtain its embedding from your CLIP server
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
        vector=embedding,
        top_k=1,
        include_metadata=True
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
        print(f"Displayed `{img_path}` (score: {score:.4f})")
    except FileNotFoundError:
        # fallback: maybe it's relative to your images/ directory
        fallback = os.path.join("images", os.path.basename(img_path))
        try:
            img = Image.open(fallback)
            img.show()
            print(f"Displayed `{fallback}` (score: {score:.4f})")
        except Exception as e:
            print(f"Could not open image `{img_path}`: {e}")

if __name__ == "__main__":
    main_client()
