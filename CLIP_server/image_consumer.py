import sys, os
import time
import signal
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pyspark.ml.linalg import DenseVector
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

sys.path.append(os.getcwd())

# from contextlib import asynccontextmanager # Removed FastAPI import
from CLIPModel import ImageCLIP
from preprocessing.kafka import KafkaConfig, SparkRowConsumer
import threading

image_model = ImageCLIP(
    batch_size=50, timeout=5.0, image_dir="./images"
)
queue_lock = Lock()
job_queue = []
job_max = 100
max_workers = 100
num_running = 0
running_lock = Lock()

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


# TODO: add multithreading. Each time send row function to queue (block if quue full). Threadpool constantly works on the queue
def kafka_image_embedding_handler(row, metadata):
    """
    Handler for Kafka image embedding requests.
    row: dict with keys:
         'image' (struct, row["image"]["origin"] is the path),
         'tensors' (DenseVector),
         'preprocess_time' (Timestamp)
    metadata: headers from Kafka (e.g. row_index)
    """
    with running_lock:
        global num_running
        num_running += 1
    # Extract iamge path & tensor vector from vectorized dataframe
    image_struct = row["image"]
    image_path   = image_struct["origin"]
    if not image_path:
        raise ValueError("Missing image.origin in row from Kafka message")

    vec = row["tensors"]
    if vec is None:
        raise ValueError("Missing 'tensors in Kafka message")
    
    embedding = image_model.get_embedding(vec, timeout=10)
    
    # Build metadata
    pinecone_metadata = {
        "image_path": image_path,
        "preprocess_time": row["preprocess_time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    
    # Upsert into Pinecone
    vid = os.path.basename(image_path)
    index.upsert(vectors=[{
        "id": vid,
        "values": embedding.tolist(),
        "metadata": pinecone_metadata
    }],
    namespace="image_embeddings")
    print(f"[Kafka→Pinecone] Upserted and Successfully processed embedding: {vid}")
    with running_lock:
        num_running -= 1
    

def run():
     init_pinecone()
     executor = ThreadPoolExecutor(max_workers=max_workers)
     futures = []

     while True:
        with queue_lock, running_lock:
            if len(job_queue) > 0 and num_running < job_max:
                row, metadata = job_queue.pop(0)
                futures.append(executor.submit(kafka_image_embedding_handler, row, metadata))
        time.sleep(0.1)

     


def send_to_thread(row, metadata):
    # Put into queue if queue is not full, otherwise wait
    while True:
        with queue_lock:
            if len(job_queue) < job_max:
                job_queue.append((row, metadata))
                break
        time.sleep(0.1)



# Removed FastAPI lifespan function

# Register signal handlers for graceful shutdown


if __name__ == "__main__":
     # Validate env
    if not all([api_key, pinecone_env, index_name]):
        sys.exit("Error: export PINECONE_API_KEY, PINECONE_ENV, and PINECONE_INDEX first in your terminal")

    print("Starting Kafka consumer...")
    processing = False
    def kafka_consumer_loop():
        global processing
        while True:
            # Start consuming in a background thread (assuming start_consuming does this)
            if not processing and image_model.queue.qsize() < 1000:
                processing = True
                consumer.start_consuming(handler=send_to_thread)
            if processing and image_model.queue.qsize() >= 1000:
                processing = False
                consumer.stop_consuming()
                break

    kafka_future = ThreadPoolExecutor(max_workers=1).submit(kafka_consumer_loop)
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        consumer.stop_consuming()
        exit(1)
        print("Consumer stopped.")
    
