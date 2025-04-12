import os
import json
import base64
import logging
import requests
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from PIL import Image
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, BinaryType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KafkaConfig:
    """Configuration class for Kafka settings"""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "raw-images",
        output_topic: str = "processed-images",
        consumer_group: str = "image-processor",
        num_partitions: int = 1,
        replication_factor: int = 1,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer_group = consumer_group
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor

    def get_producer_config(self) -> Dict[str, str]:
        """Returns configuration dict for Kafka producer"""
        return {"bootstrap.servers": self.bootstrap_servers}

    def get_consumer_config(self) -> Dict[str, str]:
        """Returns configuration dict for Kafka consumer"""
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.consumer_group,
            "auto.offset.reset": "earliest",
        }

    def get_admin_config(self) -> Dict[str, str]:
        """Returns configuration dict for Kafka admin client"""
        return {"bootstrap.servers": self.bootstrap_servers}


class SparkConfig:
    """Configuration class for Spark settings"""

    def __init__(
        self,
        app_name: str = "Kafka-Spark Image Processing",
        master: str = "local[*]",
        checkpoint_dir: str = "/tmp/checkpoint",
    ):
        self.app_name = app_name
        self.master = master
        self.checkpoint_dir = checkpoint_dir

    def create_spark_session(self) -> SparkSession:
        """Creates and returns a configured SparkSession"""
        return (
            SparkSession.builder.appName(self.app_name)
            .master(self.master)
            .config(
                "spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0",
            )
            .getOrCreate()
        )


class ImageEncoder:
    """Utility class for encoding and decoding images"""

    @staticmethod
    def encode_image_file(image_path: str) -> str:
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def decode_base64_to_bytes(base64_string: str) -> bytes:
        """Decode base64 string to image bytes"""
        return base64.b64decode(base64_string)

    @staticmethod
    def encode_bytes_to_base64(image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def bytes_to_image(image_bytes: bytes) -> Image.Image:
        """Convert bytes to PIL Image"""
        return Image.open(BytesIO(image_bytes))

    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image to bytes"""
        buffer = BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    @staticmethod
    def save_image_bytes(image_bytes: bytes, path: str) -> None:
        """Save image bytes to file"""
        with open(path, "wb") as f:
            f.write(image_bytes)

    @staticmethod
    def save_base64_image(base64_string: str, path: str) -> None:
        """Save base64 encoded image to file"""
        image_bytes = ImageEncoder.decode_base64_to_bytes(base64_string)
        ImageEncoder.save_image_bytes(image_bytes, path)


class ImageProcessor(ABC):
    """Abstract base class for image processors"""

    @abstractmethod
    def process(self, image_bytes: bytes) -> Optional[bytes]:
        """Process image bytes and return processed image bytes"""
        pass


class StandardImageProcessor(ImageProcessor):
    """Standard image processor implementing common preprocessing operations"""

    def __init__(
        self, target_size: Tuple[int, int] = (224, 224), grayscale: bool = True
    ):
        self.target_size = target_size
        self.grayscale = grayscale

    def process(self, image_bytes: bytes) -> Optional[bytes]:
        """Convert image to grayscale, resize, and normalize"""
        try:
            # Use ImageEncoder to convert bytes to PIL Image
            img = ImageEncoder.bytes_to_image(image_bytes)

            # Convert to grayscale if specified
            if self.grayscale:
                img = img.convert("L")

            # Resize to the target size
            img = img.resize(self.target_size)

            # Convert to numpy array
            img_array = np.array(img)

            # Normalize pixel values to [0,1]
            img_array = img_array / 255.0

            # Convert back to PIL Image
            processed_img = Image.fromarray((img_array * 255).astype(np.uint8))

            # Use ImageEncoder to convert PIL Image to bytes
            return ImageEncoder.image_to_bytes(processed_img)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None


class KafkaAdmin:
    """Class for Kafka administration operations"""

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.admin_client = AdminClient(config.get_admin_config())

    def create_topics(self, topics: List[str]) -> None:
        """Create Kafka topics if they don't exist"""
        topic_list = [
            NewTopic(topic, self.config.num_partitions, self.config.replication_factor)
            for topic in topics
        ]

        futures = self.admin_client.create_topics(topic_list)

        for topic, future in futures.items():
            try:
                future.result()
                logger.info(f"Topic {topic} created")
            except Exception as e:
                logger.info(f"Failed to create topic {topic}: {e}")


class KafkaImageProducer:
    """Class for publishing images to Kafka"""

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = Producer(config.get_producer_config())

    def delivery_callback(self, err, msg):
        """Callback function for message delivery reports"""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def publish_image(
        self, filename: str, image_data: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Publish a single image to Kafka"""
        if metadata is None:
            metadata = {}

        # Create a message with image data and metadata
        message = {
            "filename": filename,
            "image_data": image_data,
            "timestamp": metadata.get("timestamp", ""),
            **metadata,
        }

        # Produce message to Kafka
        self.producer.produce(
            self.config.input_topic,
            key=filename,
            value=json.dumps(message).encode("utf-8"),
            callback=self.delivery_callback,
        )
        self.producer.poll(0)

    def publish_images_from_directory(self, directory_path: str) -> None:
        """Publish all images from a directory to Kafka"""
        # Iterate through images in the directory
        for filename in os.listdir(directory_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(directory_path, filename)

                # Use ImageEncoder to encode the image
                image_data = ImageEncoder.encode_image_file(image_path)

                # Get metadata
                metadata = {
                    "timestamp": str(os.path.getmtime(image_path)),
                    "size": os.path.getsize(image_path),
                }

                # Publish to Kafka
                self.publish_image(filename, image_data, metadata)
                logger.info(f"Published {filename} to Kafka")

        # Wait for any outstanding messages to be delivered
        self.producer.flush()


class ServerSender(ABC):
    """Abstract base class for sending processed images to an external server"""

    @abstractmethod
    def send(self, image_data: Dict[str, Any]) -> bool:
        """Send image data to server and return success status"""
        pass


class RestServerSender(ServerSender):
    """Implementation of ServerSender for REST API endpoints"""

    def __init__(
        self, server_url: str, headers: Dict[str, str] = None, timeout: int = 30
    ):
        self.server_url = server_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def send(self, image_data: Dict[str, Any]) -> bool:
        """Send image data to REST server"""
        try:
            response = requests.post(
                self.server_url,
                json=image_data,
                headers=self.headers,
                timeout=self.timeout,
            )

            if response.status_code >= 200 and response.status_code < 300:
                logger.info(
                    f"Successfully sent image {image_data.get('filename')} to server"
                )
                return True
            else:
                logger.error(
                    f"Failed to send image to server: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error sending image to server: {e}")
            return False


class KafkaServerSender(ServerSender):
    """Implementation of ServerSender for sending to another Kafka topic"""

    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": bootstrap_servers})

    def delivery_callback(self, err, msg):
        """Callback function for message delivery reports"""
        if err is not None:
            logger.error(f"Server message delivery failed: {err}")
        else:
            logger.info(
                f"Server message delivered to {msg.topic()} [{msg.partition()}]"
            )

    def send(self, image_data: Dict[str, Any]) -> bool:
        """Send image data to Kafka topic"""
        try:
            # Produce message to Kafka
            self.producer.produce(
                self.topic,
                key=image_data.get("filename", "unknown"),
                value=json.dumps(image_data).encode("utf-8"),
                callback=self.delivery_callback,
            )
            self.producer.poll(0)
            return True
        except Exception as e:
            logger.error(f"Error sending image to Kafka server: {e}")
            return False


class KafkaImageConsumer:
    """Class for consuming processed images from Kafka and forwarding to server"""

    def __init__(
        self,
        config: KafkaConfig,
        server_sender: ServerSender = None,
        output_dir: str = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.server_sender = server_sender
        self.consumer = Consumer(config.get_consumer_config())

        # Create output directory if specified and it doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_consuming(self, process_func=None, timeout: float = 1.0) -> None:
        """Start consuming messages from Kafka"""
        self.consumer.subscribe([self.config.output_topic])

        try:
            while True:
                msg = self.consumer.poll(timeout)

                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                # Process the message
                try:
                    processed_msg = json.loads(msg.value().decode("utf-8"))
                    filename = processed_msg["filename"]
                    status = processed_msg["processing_status"]

                    logger.info(
                        f"Received processed image: {filename}, status: {status}"
                    )

                    # Execute custom processing function if provided
                    if process_func:
                        process_func(processed_msg)

                    # Send to server if server_sender is configured
                    if self.server_sender and status == "success":
                        if self.server_sender.send(processed_msg):
                            logger.info(
                                f"Successfully sent {filename} to server for further processing"
                            )
                        else:
                            logger.warning(f"Failed to send {filename} to server")

                    # Save locally if output_dir is specified
                    if (
                        self.output_dir
                        and status == "success"
                        and "processed_image" in processed_msg
                    ):
                        output_path = os.path.join(
                            self.output_dir, f"processed_{filename}"
                        )

                        # Use ImageEncoder to save the base64 image
                        ImageEncoder.save_base64_image(
                            processed_msg["processed_image"], output_path
                        )
                        logger.info(f"Saved processed image to {output_path}")

                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.consumer.close()


class SparkImageProcessor:
    """Class for processing images with Spark"""

    def __init__(
        self,
        spark_config: SparkConfig,
        kafka_config: KafkaConfig,
        image_processor: ImageProcessor,
    ):
        self.spark_config = spark_config
        self.kafka_config = kafka_config
        self.image_processor = image_processor
        self.spark = None

    def _process_message(self, value: bytes) -> bytes:
        """Process a single Kafka message containing an image"""
        try:
            message = json.loads(value.decode("utf-8"))

            # Use ImageEncoder to decode the base64 image data
            img_data = ImageEncoder.decode_base64_to_bytes(message["image_data"])

            # Preprocess the image using the provided processor
            processed_img = self.image_processor.process(img_data)

            if processed_img:
                # Use ImageEncoder to encode the processed image back to base64
                encoded_processed_img = ImageEncoder.encode_bytes_to_base64(
                    processed_img
                )

                # Create new message with processed image
                result = {
                    "filename": message["filename"],
                    "original_timestamp": message.get("timestamp", ""),
                    "processed_image": encoded_processed_img,
                    "processing_status": "success",
                    # Include original metadata if available
                    "metadata": {
                        k: v
                        for k, v in message.items()
                        if k not in ["filename", "image_data", "timestamp"]
                    },
                }
            else:
                result = {
                    "filename": message["filename"],
                    "original_timestamp": message.get("timestamp", ""),
                    "processing_status": "error",
                    "error": "Image processing failed",
                }

            return json.dumps(result).encode("utf-8")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing message: {error_msg}")
            return json.dumps(
                {
                    "filename": json.loads(value.decode("utf-8")).get(
                        "filename", "unknown"
                    ),
                    "processing_status": "error",
                    "error": error_msg,
                }
            ).encode("utf-8")

    def start_processing(self) -> None:
        """Start Spark Structured Streaming job for image processing"""
        # # Initialize Spark
        # self.spark = self.spark_config.create_spark_session()
        #
        # # Create UDF for processing
        # process_message_udf = udf(self._process_message, BinaryType())
        #
        # # Read from Kafka
        # df = self.spark.readStream \
        #     .format("kafka") \
        #     .option("kafka.bootstrap.servers", self.kafka_config.bootstrap_servers) \
        #     .option("subscribe", self.kafka_config.input_topic) \
        #     .option("startingOffsets", "earliest") \
        #     .load()
        #
        # # Apply processing
        # processed_df = df.select(
        #     df.key,
        #     process_message_udf(df.value).alias("value")
        # )
        #
        # # Write back to Kafka
        # query = processed_df.writeStream \
        #     .format("kafka") \
        #     .option("kafka.bootstrap.servers", self.kafka_config.bootstrap_servers) \
        #     .option("topic", self.kafka_config.output_topic) \
        #     .option("checkpointLocation", self.spark_config.checkpoint_dir) \
        #     .start()
        #
        # # Wait for the streaming query to finish
        # logger.info("Spark Structured Streaming job started. Waiting for termination...")
        # query.awaitTermination()
        pass


class ImageProcessingPipeline:
    """Main class orchestrating the entire image processing pipeline"""

    def __init__(
        self,
        kafka_config: KafkaConfig = None,
        spark_config: SparkConfig = None,
        image_processor: ImageProcessor = None,
        server_sender: ServerSender = None,
        output_dir: str = None,
    ):
        # Use default configs if not provided
        self.kafka_config = kafka_config or KafkaConfig()
        self.spark_config = spark_config or SparkConfig()
        self.image_processor = image_processor or StandardImageProcessor()
        self.server_sender = server_sender
        self.output_dir = output_dir

        # Initialize components
        self.kafka_admin = KafkaAdmin(self.kafka_config)
        self.producer = KafkaImageProducer(self.kafka_config)
        self.consumer = KafkaImageConsumer(
            self.kafka_config, self.server_sender, self.output_dir
        )
        self.spark_processor = SparkImageProcessor(
            self.spark_config, self.kafka_config, self.image_processor
        )

    def setup(self) -> None:
        """Set up the pipeline (create topics, etc.)"""
        # Create Kafka topics
        self.kafka_admin.create_topics(
            [self.kafka_config.input_topic, self.kafka_config.output_topic]
        )

        # Create output directory if specified and it doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def ingest_images(self, image_dir: str) -> None:
        """Ingest images from a directory into Kafka"""
        self.producer.publish_images_from_directory(image_dir)

    def start_processing(self) -> None:
        """Start the Spark processing job"""
        self.spark_processor.start_processing()

    def start_consuming(self) -> None:
        """Start consuming processed images"""
        self.consumer.start_consuming()


# Example usage
if __name__ == "__main__":
    # Create custom configurations (optional)
    kafka_config = KafkaConfig(
        bootstrap_servers="localhost:9092",
        input_topic="raw-images",
        output_topic="processed-images",
    )

    spark_config = SparkConfig(
        app_name="Modular Kafka-Spark Image Processing",
        checkpoint_dir="/tmp/kafka-spark-checkpoint",
    )

    # Create custom image processor (optional)
    image_processor = StandardImageProcessor(target_size=(299, 299), grayscale=True)

    # Create a server sender for forwarding processed images
    # Option 1: REST API server
    rest_sender = RestServerSender(
        server_url="http://image-processing-server.example.com/api/images",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",
        },
    )

    # Option 2: Another Kafka topic
    kafka_sender = KafkaServerSender(
        bootstrap_servers="remote-server:9092", topic="external-image-processing"
    )

    # Choose which sender to use (or None to disable sending to server)
    server_sender = rest_sender  # or kafka_sender or None

    # Initialize the pipeline
    pipeline = ImageProcessingPipeline(
        kafka_config=kafka_config,
        spark_config=spark_config,
        image_processor=image_processor,
        server_sender=server_sender,
        output_dir="./processed_output",
    )

    # Set up the pipeline
    pipeline.setup()

    # Uncomment the following lines to run different parts of the pipeline

    # Option 1: Just ingest images
    # pipeline.ingest_images("/path/to/your/images")

    # Option 2: Just process images with Spark
    # pipeline.start_processing()

    # Option 3: Just consume processed images and send to server
    # pipeline.start_consuming()

    # Example of running the full pipeline in a production environment
    # (You would typically run these components in separate processes or machines)

    # In process 1:
    # pipeline.ingest_images("/path/to/your/images")

    # In process 2:
    # pipeline.start_processing()

    # In process 3:
    # pipeline.start_consuming()
