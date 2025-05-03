import os
import sys
import requests
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import random # Add this import

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


def main_client(index,clip_url="http://localhost:8000", query=None, display=True):

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
    if display:
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
    num_workers = 100 # Represents how many "people" are sending queries at a time
    custom_query = True

    print(api_key, pinecone_env, index_name, clip_url)

    if not all(map(len, [api_key, pinecone_env, index_name])):
        sys.exit(
            "Error: please export PINECONE_API_KEY, PINECONE_ENV, and PINECONE_INDEX in your shell."
        )

    # 2️ Initialize Pinecone client
    index = init_pinecone(api_key, index_name, 512)

    if custom_query:
        query = input("Enter your search query: ").strip()
        main_client(index, clip_url, query)
    else:
        # Test with multiple queries
        queries = [
            "A cat sitting on a couch",
            "A dog playing with a ball",
            "A beautiful sunset over the mountains",
            "A person riding a bicycle in the park",
            "A group of friends having a picnic"
        ]
        queries *= 20 # Keep the multiplication if you want 100 total queries

        # --- Configuration for Batch Submission ---
        batch_size = 50
        interval_seconds = 5.0 # Submit queries randomly within this time window per batch
        # --- End Configuration ---

        # Remove shuffling if you want the original order within batches
        # random.shuffle(queries) # Commented out based on the request

        start_time = time.time()
        futures = []
        num_queries = len(queries)
        num_batches = (num_queries + batch_size - 1) // batch_size

        with ThreadPoolExecutor(max_workers=max(100, batch_size)) as executor: # Ensure enough workers for a batch
            print(f"Processing {num_queries} queries in {num_batches} batches of size {batch_size} over {interval_seconds}s intervals.")

            for i in tqdm(range(num_batches), desc="Processing Batches"):
                batch_start_index = i * batch_size
                batch_end_index = min((i + 1) * batch_size, num_queries)
                current_batch_queries = queries[batch_start_index:batch_end_index]

                if not current_batch_queries:
                    continue

                interval_start_time = time.time()
                tasks_for_interval = []
                for query in current_batch_queries:
                    # Calculate when the task should START within the interval
                    random_start_offset = random.uniform(0, interval_seconds)
                    tasks_for_interval.append((random_start_offset, query))

                # Sort tasks by their scheduled start time
                tasks_for_interval.sort(key=lambda x: x[0])

                # Submit tasks at their scheduled time within the interval
                for start_offset, query in tasks_for_interval:
                    current_time_in_interval = time.time() - interval_start_time
                    sleep_duration = max(0, start_offset - current_time_in_interval)
                    time.sleep(sleep_duration)

                    # Submit the actual work
                    future = executor.submit(main_client, index, clip_url, query, False)
                    futures.append(future)

                # Wait for the remainder of the interval before starting the next batch
                interval_end_time = time.time()
                elapsed_in_interval = interval_end_time - interval_start_time
                sleep_remainder = max(0, interval_seconds - elapsed_in_interval)
                time.sleep(sleep_remainder)


            print("All queries submitted. Waiting for results...")
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
                try:
                    future.result() # Retrieve result or raise exception
                except Exception as e:
                    print(f"Error processing query: {e}")

        end_time = time.time()
        print(f"Processed {len(queries)} queries in {end_time - start_time:.2f} seconds (including interval delays).")
        # Note: The queries/sec calculation needs careful interpretation due to batching and delays.
        # Effective processing rate might be calculated differently depending on what you want to measure.
