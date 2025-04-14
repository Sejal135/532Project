import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from confluent_kafka import Consumer, KafkaError, Producer
from PIL import Image

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # File handler - rotates logs when they reach 10MB, keeps 5 backup files
        RotatingFileHandler(
            "logs/kafka_image_service.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=1,
            encoding="utf-8",
        ),
        # Console handler
        logging.StreamHandler(),
    ],
)


class KafkaConfig:
    """Configuration class for Kafka connection settings."""

    def __init__(self, bootstrap_servers, group_id=None):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id or f"image-consumer-{uuid.uuid4()}"
        self.logger = logging.getLogger("KafkaConfig")
        self.logger.info(
            f"Initialized KafkaConfig with bootstrap_servers={bootstrap_servers}"
        )

    def get_producer_config(self):
        """Return configuration dictionary for Kafka producer."""
        client_id = f"image-producer-{uuid.uuid4()}"
        self.logger.debug(f"Created producer config with client_id={client_id}")
        return {"bootstrap.servers": self.bootstrap_servers, "client.id": client_id}

    def get_consumer_config(self):
        """Return configuration dictionary for Kafka consumer."""
        self.logger.debug(f"Created consumer config with group_id={self.group_id}")
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
        }


class ImageSerializer:
    """Handles serialization and deserialization of images for Kafka transport."""

    def __init__(self):
        self.logger = logging.getLogger("ImageSerializer")

    def serialize_image(self, image, image_format="JPEG", **kwargs):
        """
        Serialize a PIL Image to a base64 encoded string within a JSON object.

        Args:
            image: PIL Image object
            image_format: Format to save the image as (JPEG, PNG, etc.)
            **kwargs: Additional metadata to include

        Returns:
            Dictionary with image data and metadata
        """
        self.logger.debug(
            f"Serializing image ({image.width}x{image.height}) to {image_format} format"
        )

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image_format)
        img_byte_arr = img_byte_arr.getvalue()

        # Encode as base64 string
        encoded_img = base64.b64encode(img_byte_arr).decode("utf-8")

        # Create message payload
        payload = {
            "image_data": encoded_img,
            "image_format": image_format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }

        self.logger.debug(
            f"Image serialized. Payload size: {len(json.dumps(payload))} bytes"
        )
        return payload

    def deserialize_image(self, payload):
        """
        Deserialize a JSON object containing a base64 encoded image back to a PIL Image.

        Args:
            payload: Dictionary with image data and metadata

        Returns:
            Tuple of (PIL Image object, metadata dictionary)
        """
        # Extract image data and metadata
        encoded_img = payload.pop("image_data")
        image_format = payload.pop("image_format", "JPEG")

        self.logger.debug(f"Deserializing image from {image_format} format")

        # Decode base64 string
        img_bytes = base64.b64decode(encoded_img)

        # Convert bytes back to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        self.logger.debug(
            f"Image deserialized. Dimensions: {img.width}x{img.height}, Mode: {img.mode}"
        )

        return img, payload


class ImageProducer:
    """Kafka producer for sending images to a topic."""

    def __init__(self, config):
        """
        Initialize the image producer with Kafka configuration.

        Args:
            config: KafkaConfig object with connection settings
        """
        self.producer_config = config.get_producer_config()
        self.producer = Producer(self.producer_config)
        self.serializer = ImageSerializer()
        self.logger = logging.getLogger("ImageProducer")
        self.logger.info(
            f"ImageProducer initialized with config: {self.producer_config}"
        )

        # Add callback for delivery reports
        self.producer_callbacks = 0
        self.producer_errors = 0

    def delivery_callback(self, err, msg):
        """Callback function for producer delivery reports."""
        if err is not None:
            self.producer_errors += 1
            self.logger.error(f"Message delivery failed: {err}")
        else:
            self.producer_callbacks += 1
            self.logger.debug(
                f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )

    def send_image(self, topic, image, image_format="JPEG", metadata=None):
        """
        Send an image to a Kafka topic.

        Args:
            topic: Kafka topic to send image to
            image: PIL Image object to send
            image_format: Format to save the image as (JPEG, PNG, etc.)
            metadata: Additional metadata to include with the image
        """
        metadata = metadata or {}
        self.logger.info(f"Sending image to topic '{topic}' with metadata: {metadata}")

        payload = self.serializer.serialize_image(image, image_format, **metadata)

        # Convert payload to JSON string
        message = json.dumps(payload).encode("utf-8")
        message_size_kb = len(message) / 1024
        self.logger.info(f"Message size: {message_size_kb:.2f} KB")

        # Send message to Kafka topic
        try:
            self.producer.produce(topic, value=message, callback=self.delivery_callback)
            self.logger.debug(f"Message produced to topic '{topic}'")
            self.producer.poll(0)  # Trigger delivery reports
        except BufferError:
            self.logger.warning("Local producer queue is full, waiting for space...")
            self.producer.flush()  # Wait until all messages are sent
            self.producer.produce(topic, value=message, callback=self.delivery_callback)
        except Exception as e:
            self.logger.error(f"Error producing message: {str(e)}")
            raise

    def flush(self, timeout=None):
        """Flush the producer to ensure all messages are sent."""
        messages_pending = self.producer.flush(timeout)
        if messages_pending > 0:
            self.logger.warning(
                f"{messages_pending} messages still pending after flush timeout"
            )
        else:
            self.logger.info("All messages flushed successfully")
        return messages_pending

    def close(self):
        """Close the producer connection."""
        self.logger.info("Closing producer connection")
        self.flush()
        self.logger.info(
            f"Producer stats: {self.producer_callbacks} successful deliveries, {self.producer_errors} errors"
        )


class ImageConsumer:
    """Kafka consumer for receiving images from a topic."""

    def __init__(self, config, topics):
        """
        Initialize the image consumer with Kafka configuration.

        Args:
            config: KafkaConfig object with connection settings
            topics: List of topics to subscribe to
        """
        self.consumer_config = config.get_consumer_config()
        self.consumer = Consumer(self.consumer_config)
        self.topics = topics if isinstance(topics, list) else [topics]
        self.consumer.subscribe(self.topics)
        self.serializer = ImageSerializer()
        self.logger = logging.getLogger("ImageConsumer")
        self.logger.info(f"ImageConsumer initialized for topics: {self.topics}")

        # Track metrics
        self.messages_received = 0
        self.errors_encountered = 0
        self.total_bytes_received = 0

    def receive_image_batch(self, batch_size=10, batch_timeout=5.0):
        """
        Poll for and receive a batch of images.

        Args:
            batch_size: Number of messages to accumulate before returning
            batch_timeout: Max time in seconds to wait for the batch

        Returns:
            List of tuples (PIL Image object, metadata dictionary)
        """
        self.logger.info(f"Receiving a batch of up to {batch_size} images")
        batch = []
        start_time = datetime.now()

        while (
            len(batch) < batch_size
            and (datetime.now() - start_time).total_seconds() < batch_timeout
        ):
            msg = self.consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                self.errors_encountered += 1
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    self.logger.error(f"Error while consuming message: {msg.error()}")
                continue

            self.messages_received += 1
            message_size = len(msg.value()) if msg.value() else 0
            self.total_bytes_received += message_size

            try:
                payload = json.loads(msg.value().decode("utf-8"))
                image, metadata = self.serializer.deserialize_image(payload)
                batch.append((image, metadata))
                self.logger.debug(f"Added image to batch (current size: {len(batch)})")

            except Exception as e:
                self.errors_encountered += 1
                self.logger.error(f"Error deserializing image: {str(e)}")

        if batch:
            self.logger.info(f"Received batch of {len(batch)} images")
        else:
            self.logger.info("No images received in batch")
        return batch

    def receive_image(self, timeout=1.0):
        """
        Poll for and receive an image from subscribed topics.

        Args:
            timeout: Timeout in seconds to wait for a message

        Returns:
            Tuple of (PIL Image object, metadata dictionary) or None if no message
        """
        self.logger.debug(f"Polling for messages with timeout {timeout}s")
        msg = self.consumer.poll(timeout)

        if msg is None:
            self.logger.debug("No message received during poll")
            return None

        if msg.error():
            self.errors_encountered += 1
            error_code = msg.error().code()
            if error_code == KafkaError._PARTITION_EOF:
                # End of partition event
                self.logger.info(
                    f"Reached end of partition {msg.topic()}-{msg.partition()}"
                )
            else:
                self.logger.error(f"Error while consuming message: {msg.error()}")
            return None

        # Message received successfully
        self.messages_received += 1
        message_size = len(msg.value()) if msg.value() else 0
        self.total_bytes_received += message_size

        self.logger.info(
            f"Received message from {msg.topic()} [{msg.partition()}] at offset {msg.offset()}, size: {message_size / 1024:.2f} KB"
        )

        try:
            # Parse and decode the message
            payload = json.loads(msg.value().decode("utf-8"))
            self.logger.debug(
                f"Message decoded. Image format: {payload.get('image_format', 'unknown')}"
            )

            image, metadata = self.serializer.deserialize_image(payload)
            self.logger.info(f"Image deserialized successfully. Metadata: {metadata}")
            return image, metadata

        except json.JSONDecodeError as e:
            self.errors_encountered += 1
            self.logger.error(f"Failed to decode JSON: {str(e)}")
            return None
        except Exception as e:
            self.errors_encountered += 1
            self.logger.error(f"Error deserializing image: {str(e)}")
            return None

    def commit(self):
        """Commit offsets."""
        self.logger.debug("Committing offsets")
        self.consumer.commit()

    def close(self):
        """Close the consumer connection."""
        self.logger.info("Closing consumer connection")
        self.consumer.close()
        self.logger.info(
            f"Consumer stats: {self.messages_received} messages received, "
            f"{self.errors_encountered} errors, "
            f"{self.total_bytes_received / 1024 / 1024:.2f} MB total data"
        )


def process_image(image, metadata, index=None):
    """
    Process a single image in parallel (dummy example: save image).
    Replace this with real processing logic (e.g., inference, transformations).
    """
    logger = logging.getLogger("ParallelProcessor")
    logger.info(f"[Worker {index}] Processing image with metadata: {metadata}")
    try:
        output_path = f"processed_image_{index}.jpg"
        image.save(output_path)
        logger.info(f"[Worker {index}] Image saved to {output_path}")
        return {"status": "success", "index": index, "path": output_path}
    except Exception as e:
        logger.error(f"[Worker {index}] Failed to process image: {e}")
        return {"status": "error", "index": index, "error": str(e)}


# Example usage
if __name__ == "__main__":
    # Configure logging for main
    logger = logging.getLogger("KafkaImageExample")

    # Configuration
    logger.info("Initializing Kafka configuration")
    config = KafkaConfig(bootstrap_servers="localhost:9092")

    # Example 1: Send an image
    logger.info("Example 1: Sending a single image")
    producer = ImageProducer(config)

    try:
        image = Image.open("example.jpg")
        logger.info(f"Loaded image with dimensions: {image.width}x{image.height}")
        producer.send_image("image-input", image, metadata={"source": "camera1"})
    except FileNotFoundError:
        logger.error("Could not find example.jpg")
    except Exception as e:
        logger.error(f"Error sending image: {str(e)}", exc_info=True)
    finally:
        producer.close()

    # Example 2: Receive a message
    consumer = ImageConsumer(config, topics=["image-input"])
    result = consumer.receive_image(timeout=1.0)
    if result is not None:
        image, metadata = result
    # then process the image
    # Could create the EmbeddingRequest here and queue it. Could also use this received instead of the post http request
    # Probably better for performance since no need for TCP connection with HTTP requests at large scale
    consumer.close()
