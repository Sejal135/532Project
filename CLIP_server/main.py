from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn
import argparse
from CLIPModel import TextCLIP, ImageCLIP
from fastapi import HTTPException
from preprocessing.kafka import KafkaConfig, SparkRowConsumer

app = FastAPI(title="CLIP Server")
text_model = TextCLIP(batch_size=32, timeout=5.0)
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

        print(row)
        # embedding = image_model.get_embedding(image_name, timeout=10)
        # TODO: Store embedding and metadata in Pinecone or your storage here

        print(f"[Kafka] Successfully created embedding for {image_name}")
    except Exception as e:
        print(f"[Kafka] Error processing image embedding: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Start Kafka consumer
    consumer.start_consuming(handler=kafka_image_embedding_handler)
    yield
    # SHUTDOWN: Stop Kafka consumer
    consumer.stop_consuming()
    consumer.close()


@app.get("/")
def home():
    return {"message": "Welcome to the CLIP Server!"}


@app.get("/query_embedding")
def create_query_embedding(request: Request):
    text = request.query_params.get("text")
    if not text:
        return HTTPException(status_code=400, detail="No text provided")

    # Process the text and return the embedding
    try:
        embedding = text_model.get_embedding(text, timeout=10)
        return {"embedding": embedding.tolist()}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


@app.post("/image_embedding")
def create_image_embedding(request: Request):
    """Deprecated in favor of Kafka kafka_image_embedding_handler"""
    image_name = request.query_params.get("image_name")
    metadata = request.query_params.get("metadata")
    if not image_name:
        return HTTPException(status_code=400, detail="No image name provided")

    # Process the image and return the embedding
    try:
        embedding = image_model.get_embedding(image_name, timeout=10)
        # TODO: store embedding and metadata in pincone dataset here

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

    return HTTPException(status_code=200, detail="Image embedding created successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )

    args = parser.parse_args()

    print(f"Starting CLIP server on http://localhost:{args.port}")
    uvicorn.run(app, port=args.port)
