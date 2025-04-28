import sys, os
import time
import signal

sys.path.append(os.getcwd())

# from contextlib import asynccontextmanager # Removed FastAPI import
from CLIPModel import ImageCLIP
from preprocessing.kafka import KafkaConfig, SparkRowConsumer

image_model = ImageCLIP(
    batch_size=32, timeout=5.0, image_dir="/Users/samonuallain/532Project/images"
)

# Setup Kafka consumer
kafka_config = KafkaConfig(bootstrap_servers="localhost:9092")  # update if needed
consumer = SparkRowConsumer(config=kafka_config, topics=["image_embedding_requests"])


def kafka_image_embedding_handler(row, metadata):
    """
    Handler for Kafka image embedding requests.
    row: dict with 'image_name' key (and optionally 'metadata')
    metadata: additional info sent with the message
    """
    try:
        image_name = row.get("image_name")
        if not image_name:
            raise ValueError("No image_name provided in Kafka message.")

        print(f"Received request for: {image_name}")
        # embedding = image_model.get_embedding(image_name, timeout=10) # Uncomment when ready
        # TODO: Store embedding and metadata in Pinecone or your storage here

        print(f"[Kafka] Successfully processed embedding for {image_name}") # Changed print message slightly
    except Exception as e:
        print(f"[Kafka] Error processing image embedding: {e}")


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
    print("Starting Kafka consumer...")
    try:
        # Start consuming in a background thread (assuming start_consuming does this)
        consumer.start_consuming(handler=kafka_image_embedding_handler)

        # Keep the main thread alive while the consumer runs
        while running:
            time.sleep(1) # Check the running flag every second

    except Exception as e:
        print(f"An error occurred during consumer operation: {e}")
    finally:
        # Ensure consumer is stopped and closed on exit
        print("Stopping Kafka consumer...")
        consumer.stop_consuming()
        consumer.close()
        print("Consumer stopped.")