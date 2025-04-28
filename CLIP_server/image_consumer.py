import sys, os
import time
import signal
import pinecone
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

api_key = os.getenv("pcsk_3eJ2cq_MYjUEN6HXnQN9X8RTjidvSrKT3AmYsij6oD2URuEccV5r8CyKgz7gNQ6QLf2P7x")
pinecone_env = os.getenv("us-east1-aws")
index_name = os.getenv("dense-image-index")
pinecone_dimension = int(os.getenv("PINECONE_DIM", "512"))

index = None

def init_pinecone():
    global index
    pinecone.init(
        api_key=api_key,
        environment=pinecone_env
    )
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=pinecone_dimension,
            metric="cosine"
        )

    index = pinecone.Index(index_name)    
    print(f"[Pinecone] Initialized index '{index_name}' with dim={pinecone_dimension}")

def kafka_image_embedding_handler(row, metadata):
    """
    Handler for Kafka image embedding requests.
    row: dict with keys:
         'image' (struct, row["image"]["origin"] is the path),
         'tensors' (DenseVector),
         'preprocess_time' (Timestamp)
    metadata: headers from Kafka (e.g. row_index)
    """

    try:
        # Extract iamge path & tensor vector from vectorized dataframe
        image_struct = row.get("image", {})
        image_path   = image_struct.get("origin")
        if not image_path:
            raise ValueError("Missing image.origin in row from Kafka message")

        vec = row["tensors"]
        if vec is None:
            raise ValueError("Missing 'tensors in Kafka message")
        
        embedding = image_model.get_embedding(image_path, timeout=10)
        vector = embedding.tolist()  # Convert to Python list

        # Convert to Python list
        if isinstance(vec, DenseVector):
            vector = vec.toArray().tolist()
        else:
            vector = list(vec)
        
        # Build metadata
        pinecone_metadata = {
            "image_path": image_path,
            "preprocess_time": row.get("preprocess_time")
        }
        
        # Upsert into Pinecone
        vid = os.path.basename(image_path)
        index.upsert(vectors=[(vid, vec, pinecone_metadata)])
        print(f"[Kafka→Pinecone] Upserted and Successfully processed embedding: {vid}")

    except Exception as e:
        print(f"[Kafka Error] Error processing image embedding in kafka_image_embedding_handler: {e}")
        
   
# Removed FastAPI lifespan function

# Flag to control the main loop
running = True

def signal_handler(sig, frame):
    """Handles termination signals for graceful shutdown."""
    global running
    print("Termination signal received. Shutting down consumer...")
    running = False

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
     # Validate env
    if not all([api_key, pinecone_env, index_name]):
        sys.exit("Error: export PINECONE_API_KEY, PINECONE_ENV, and PINECONE_INDEX first in your terminal")

    # Init Pinecone
    init_pinecone()

    print("Starting Kafka consumer...")
    try:
        # Stop consuming when reaching max queue, resuming when there is more space
        while True:
            # Start consuming in a background thread (assuming start_consuming does this)
            if not processing and image_model.queue.qsize() < 1000:
                processing = True
                consumer.start_consuming(handler=kafka_image_embedding_handler)
            if processing and image_model.queue.qsize() == 1000:
                processing = False
                consumer.stop_consuming()

        # Keep the main thread alive while the consumer runs
        while running:
            time.sleep(1)  # Check the running flag every second

    except Exception as e:
        print(f"An error occurred during consumer operation: {e}")
    finally:
        # Ensure consumer is stopped and closed on exit
        print("Stopping Kafka consumer...")
        consumer.stop_consuming()
        consumer.close()
        print("Consumer stopped.")