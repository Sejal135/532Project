import sys, os
import time
import signal
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pyspark.ml.linalg import DenseVector

sys.path.append(os.getcwd())

# from contextlib import asynccontextmanager # Removed FastAPI import
from CLIPModel import ImageCLIP
from preprocessing.kafka import KafkaConfig, SparkRowConsumer

image_model = ImageCLIP(
    batch_size=32, timeout=5.0, image_dir="./images"
)

# Setup Kafka consumer
kafka_config = KafkaConfig(bootstrap_servers="localhost:9092")  # update if needed
consumer = SparkRowConsumer(config=kafka_config, topics=["image_embedding_requests"])

api_key    = os.getenv("PINECONE_API_KEY", "pcsk_3eJ2cq_MYjUEN6HXnQN9X8RTjidvSrKT3AmYsij6oD2URuEccV5r8CyKgz7gNQ6QLf2P7x")
pinecone_env   = os.getenv("PINECONE_ENV", "us-east1-aws")
index_name = os.getenv("PINECONE_INDEX", "dense-image-index")
clip_url   = os.getenv("CLIP_SERVER_URL", "http://localhost:8000/query_embedding")
index_host = os.getenv("PINECONE_INDEX_HOST", "dense-image-index-qa9q792.svc.aped-4627-b74a.pinecone.io")
pinecone_dimension = int(os.getenv("PINECONE_DIM", "512"))

index = None

def init_pinecone():
    global index
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


# TODO: add multiprocessing
def kafka_image_embedding_handler(row, metadata):
    """
    Handler for Kafka image embedding requests.
    row: dict with keys:
         'image' (struct, row["image"]["origin"] is the path),
         'tensors' (DenseVector),
         'preprocess_time' (Timestamp)
    metadata: headers from Kafka (e.g. row_index)
    """
    # Extract iamge path & tensor vector from vectorized dataframe
    print("ROW:", type(row))
    print(row.keys())
    image_struct = row["image"]
    image_path   = image_struct["origin"]
    print(image_path)
    if not image_path:
        raise ValueError("Missing image.origin in row from Kafka message")

    vec = row["tensors"]
    print("VECTOR:", len(vec), type(vec))
    if vec is None:
        raise ValueError("Missing 'tensors in Kafka message")
    
    embedding = image_model.get_embedding(vec, timeout=10)
    print("EMBEDDING:", len(embedding), type(embedding))
    
    # Build metadata
    pinecone_metadata = {
        "image_path": image_path,
        "preprocess_time": row["preprocess_time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    print("METADATA:", pinecone_metadata)
    
    # Upsert into Pinecone
    vid = os.path.basename(image_path)
    print("Upserting to Pinecone...")
    index.upsert(vectors=[{
        "id": vid,
        "values": embedding.tolist(),
        "metadata": pinecone_metadata
    }],
    namespace="image_embeddings")
    print("Upserted to Pinecone:", vid)
    print(f"[Kafka→Pinecone] Upserted and Successfully processed embedding: {vid}")
    
   
# Removed FastAPI lifespan function

# Register signal handlers for graceful shutdown


if __name__ == "__main__":
     # Validate env
    if not all([api_key, pinecone_env, index_name]):
        sys.exit("Error: export PINECONE_API_KEY, PINECONE_ENV, and PINECONE_INDEX first in your terminal")

    # Init Pinecone
    init_pinecone()

    print("Starting Kafka consumer...")
    # Stop consuming when reaching max queue, resuming when there is more space
    processing = False
    while True:
        # Start consuming in a background thread (assuming start_consuming does this)
        if not processing and image_model.queue.qsize() < 1000:
            processing = True
            consumer.start_consuming(handler=kafka_image_embedding_handler)
        if processing and image_model.queue.qsize() >= 1000:
            processing = False
            consumer.stop_consuming()

    # except Exception as e:
    #     print(f"An error occurred during consumer operation: {e}")
    # finally:
    #     # Ensure consumer is stopped and closed on exit
    #     print("Stopping Kafka consumer...")
    #     consumer.stop_consuming()
    #     consumer.close()
    #     print("Consumer stopped.")