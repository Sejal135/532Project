import base64
import threading
import io
import json
import logging
import os
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from confluent_kafka import Consumer, KafkaError, Producer
from PIL import Image
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Row
import pickle

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


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class KafkaConfig:
    """Configuration class for Kafka connection settings."""

    def __init__(self, bootstrap_servers, group_id=None):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id or f"spark-consumer-{uuid.uuid4()}"
        self.logger = logging.getLogger("KafkaConfig")
        self.logger.info(
            f"Initialized KafkaConfig with bootstrap_servers={bootstrap_servers}"
        )

    def get_producer_config(self):
        """Return configuration dictionary for Kafka producer."""
        client_id = f"spark-producer-{uuid.uuid4()}"
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


class MessageSerializer:
    """Base class for message serializers."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def serialize_message(self, message):
        """Serialize a message."""
        raise NotImplementedError("Subclasses must implement serialize_message")

    def deserialize_message(self, serialized_message):
        """Deserialize a message."""
        raise NotImplementedError("Subclasses must implement deserialize_message")


class JsonMessageSerializer(MessageSerializer):
    """Handles serialization and deserialization of JSON messages."""

    def serialize_message(self, message):
        """
        Serialize a Python dictionary to a JSON string.
        Args:
            message_dict: Dictionary to serialize
        Returns:
            JSON string
        """
        self.logger.debug(f"Serializing JSON message")
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
        # Convert to JSON string
        json_str = json.dumps(message)
        self.logger.debug(f"Message serialized. Size: {len(json_str)} bytes")
        return json_str

    def deserialize_message(self, serialized_message):
        """
        Deserialize a JSON string to a Python dictionary.
        Args:
            json_str: JSON string to deserialize
        Returns:
            Python dictionary
        """
        self.logger.debug(f"Deserializing JSON message")
        message_dict = json.loads(serialized_message)
        self.logger.debug(f"Message deserialized successfully")
        return message_dict


class SparkRowSerializer(MessageSerializer):
    """Handles serialization and deserialization of individual PySpark Row objects."""

    def __init__(self):
        super().__init__()
        self.logger.info("SparkRowSerializer initialized")

    def serialize_message(self, message, metadata=None):
        """
        Serialize a single PySpark Row to bytes.

        Args:
            row: PySpark Row object to serialize
            metadata: Optional dictionary of metadata to include

        Returns:
            Tuple of (serialized_data, headers)
        """
        self.logger.debug("Serializing PySpark Row")

        # Convert Row to dictionary
        if isinstance(message, Row):
            row_dict = message.asDict()
        else:
            row_dict = message

        # Create metadata
        headers = []
        schema_info = {"type": "spark_row", "timestamp": datetime.now().isoformat()}

        # Add user metadata if provided
        if metadata:
            schema_info["metadata"] = metadata

        # Serialize with pickle and encode with base64 for safety
        pickle_data = pickle.dumps(row_dict)
        serialized_data = base64.b64encode(pickle_data)

        # Add schema info to headers
        headers.append(
            {"key": "schema", "value": json.dumps(schema_info).encode("utf-8")}
        )
        headers.append({"key": "format", "value": b"pickle_base64"})

        self.logger.debug(f"Row serialized. Size: {len(serialized_data)} bytes")
        return serialized_data, headers

    def deserialize_message(self, serialized_message, headers=None):
        """
        Deserialize bytes back to a dictionary representing a PySpark Row.

        Args:
            serialized_message: Serialized Row data
            headers: Message headers with format and schema information

        Returns:
            Dictionary and metadata
        """
        self.logger.debug("Deserializing PySpark Row message")

        # Extract schema from headers
        schema_info = None
        if headers:
            header_dict = {
                h[0]: h[1] for h in headers if h[0] in [b"format", b"schema"]
            }
            if b"schema" in header_dict:
                schema_value = header_dict[b"schema"]
                schema_info = json.loads(
                    schema_value.decode("utf-8")
                    if isinstance(schema_value, bytes)
                    else schema_value
                )

        try:
            # Decode base64 and unpickle
            pickle_data = base64.b64decode(serialized_message)
            row_dict = pickle.loads(pickle_data)

            # Extract metadata
            metadata = {}
            if schema_info:
                if "metadata" in schema_info:
                    metadata = schema_info["metadata"]
                metadata["timestamp"] = schema_info.get("timestamp", None)

            self.logger.debug("Row deserialized successfully")
            return row_dict, metadata
        except Exception as e:
            self.logger.error(f"Error deserializing Row: {e}")
            raise


class DataFrameSerializer(MessageSerializer):
    """Handles serialization and deserialization of DataFrame messages."""

    def __init__(self, format="json"):
        """
        Initialize DataFrame serializer.
        Args:
            format: Serialization format ('json' or 'parquet')
        """
        super().__init__()
        self.format = format.lower()
        if self.format not in ["json", "parquet"]:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'parquet'")
        self.logger.info(f"DataFrameSerializer initialized with format: {self.format}")

    def serialize_message(self, message, metadata=None):
        """
        Serialize a DataFrame to bytes.
        Args:
            df: PySpark or Pandas DataFrame to serialize
            metadata: Optional dictionary of metadata to include
        Returns:
            Tuple of (serialized_data, headers)
        """
        self.logger.debug(f"Serializing DataFrame using {self.format} format")
        # Convert PySpark DataFrame to pandas if needed
        if isinstance(message, DataFrame):
            self.logger.debug("Converting PySpark DataFrame to pandas")
            pandas_df = message.toPandas()
        else:
            pandas_df = message

        # Create metadata including schema information
        headers = []
        schema_info = {
            "columns": pandas_df.columns.tolist(),
            "dtypes": {col: str(pandas_df[col].dtype) for col in pandas_df.columns},
        }

        # Add user metadata if provided
        if metadata:
            schema_info["metadata"] = metadata

        # Add timestamp
        schema_info["timestamp"] = datetime.now().isoformat()

        # Serialize based on format
        if self.format == "json":
            serialized_data = pandas_df.to_json(orient="records")
            headers.append({"key": "format", "value": b"json"})
        else:  # parquet
            buffer = io.BytesIO()
            pandas_df.to_parquet(buffer)
            buffer.seek(0)
            serialized_data = buffer.getvalue()
            headers.append({"key": "format", "value": b"parquet"})

        # Add schema info to headers
        headers.append(
            {"key": "schema", "value": json.dumps(schema_info).encode("utf-8")}
        )

        self.logger.debug(
            f"DataFrame serialized. Size: {len(serialized_data) if isinstance(serialized_data, bytes) else (len(serialized_data.encode('utf-8')) if isinstance(serialized_data, str) else 0)} bytes"
        )
        return serialized_data, headers

    def deserialize_message(self, serialized_message, headers=None):
        """
        Deserialize bytes to a DataFrame.
        Args:
            message_data: Serialized DataFrame data
            headers: Message headers with format and schema information
        Returns:
            Pandas DataFrame and metadata dictionary
        """
        self.logger.debug("Deserializing DataFrame message")
        # Extract format and schema from headers
        format_type = self.format
        schema_info = None
        if headers:
            header_dict = {
                h[0]: h[1] for h in headers if h[0] in [b"format", b"schema"]
            }
            if b"format" in header_dict:
                format_value = header_dict[b"format"]
                format_type = (
                    format_value.decode("utf-8")
                    if isinstance(format_value, bytes)
                    else format_value
                )
            if b"schema" in header_dict:
                schema_value = header_dict[b"schema"]
                schema_info = json.loads(
                    schema_value.decode("utf-8")
                    if isinstance(schema_value, bytes)
                    else schema_value
                )

        # Deserialize based on format
        try:
            if format_type == "json":
                if isinstance(serialized_message, bytes):
                    serialized_message = serialized_message.decode("utf-8")
                pandas_df = pd.read_json(serialized_message, orient="records")
            else:  # parquet
                if isinstance(serialized_message, str):
                    serialized_message = serialized_message.encode("utf-8")
                buffer = io.BytesIO(serialized_message)
                pandas_df = pd.read_parquet(buffer)

            self.logger.debug("DataFrame deserialized successfully")

            # Extract metadata from schema_info
            metadata = {}
            if schema_info:
                if "metadata" in schema_info:
                    metadata = schema_info["metadata"]
                metadata["timestamp"] = schema_info.get("timestamp", None)

            return pandas_df, metadata
        except Exception as e:
            self.logger.error(f"Error deserializing DataFrame: {e}")
            raise


class SparkRowProducer:
    """Kafka producer for sending individual PySpark Row objects."""

    def __init__(self, config):
        """
        Initialize the PySpark Row producer.

        Args:
            config: KafkaConfig object with connection settings
        """
        self.producer_config = config.get_producer_config()
        self.producer = Producer(self.producer_config)
        self.serializer = SparkRowSerializer()
        self.logger = logging.getLogger("SparkRowProducer")
        self.logger.info(
            f"SparkRowProducer initialized with config: {self.producer_config}"
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

    def send_row(self, topic, row, metadata=None, key=None):
        """
        Send a single PySpark Row to a Kafka topic.

        Args:
            topic: Kafka topic to send message to
            row: PySpark Row to be serialized and sent
            metadata: Optional dictionary of metadata to include
            key: Optional message key
        """
        self.logger.info(f"Sending PySpark Row to topic '{topic}'")
        serialized_data, serialized_key = None, None

        try:
            # Serialize the Row
            serialized_data, headers = self.serializer.serialize_message(row, metadata)
            # print(len(serialized_data))
            if serialized_data is None:
                raise Exception("Row serialization failed")

            # Prepare key
            serialized_key = key.encode("utf-8") if key else None

            # Send Row to Kafka topic
            self.producer.produce(
                topic=topic,
                value=serialized_data,
                key=serialized_key,
                callback=self.delivery_callback,
            )
            self.logger.debug(f"Row produced to topic '{topic}'")
            self.producer.poll(0)  # Trigger delivery reports

        except BufferError:
            self.logger.warning("Local producer queue is full, waiting for space...")
            self.producer.flush()
            # Try again after flush
            self.producer.produce(
                topic=topic,
                value=serialized_data,
                key=serialized_key,
                # headers=headers,
                callback=self.delivery_callback,
            )
        except Exception as e:
            self.logger.error(f"Error producing Row message: {str(e)}")
            raise

    def send_dataframe_rows(self, topic, df, key_column=None, metadata=None):
        """
        Send each row of a DataFrame individually to Kafka.

        Args:
            topic: Kafka topic to send messages to
            df: PySpark DataFrame to extract rows from
            key_column: Optional column to use as message key
            metadata: Optional metadata to attach to each row
        """
        self.logger.info(f"Sending DataFrame rows one by one to topic '{topic}'")

        # Collect all rows - this brings data to driver
        rows = df.collect()
        self.logger.info(f"Processing {len(rows)} rows from DataFrame")

        # Send each row individually
        for i, row in enumerate(rows):
            try:
                # Extract key if key_column specified
                key = None
                if key_column and key_column in row.__fields__:
                    key = str(row[key_column])

                # Add row index to metadata
                row_metadata = metadata.copy() if metadata else {}
                row_metadata["row_index"] = i

                # Send single row
                self.send_row(topic, row, metadata=row_metadata, key=key)

                # Log progress for large DataFrames
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Sent {i + 1}/{len(rows)} rows")

            except Exception as e:
                self.logger.error(f"Error sending row {i}: {e}")
                raise

        self.logger.info(f"Completed sending all {len(rows)} rows to topic '{topic}'")

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


class DataFrameProducer:
    """Kafka producer for sending DataFrame messages."""

    def __init__(self, config, format="json"):
        """
        Initialize the DataFrame producer.
        Args:
            config: KafkaConfig object with connection settings
            format: Serialization format ('json' or 'parquet')
        """
        self.producer_config = config.get_producer_config()
        self.producer = Producer(self.producer_config)
        self.serializer = DataFrameSerializer(format)
        self.logger = logging.getLogger("DataFrameProducer")
        self.logger.info(
            f"DataFrameProducer initialized with config: {self.producer_config}"
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

    def send_dataframe(self, topic, df, metadata=None, key=None):
        """
        Send a DataFrame to a Kafka topic.
        Args:
            topic: Kafka topic to send message to
            df: DataFrame to be serialized and sent
            metadata: Optional dictionary of metadata to include
            key: Optional message key
        """
        self.logger.info(f"Sending DataFrame to topic '{topic}'")
        serialized_data, headers, serialized_key = None, None, None
        try:
            # Serialize the DataFrame
            serialized_data, headers = self.serializer.serialize_message(df, metadata)
            if serialized_data is None:
                raise Exception("Dataframe serialized failed")
            # Convert to bytes if needed
            if isinstance(serialized_data, str):
                serialized_data = serialized_data.encode("utf-8")
            message_size_kb = len(serialized_data) / 1024
            self.logger.info(f"DataFrame message size: {message_size_kb:.2f} KB")
            # Prepare key
            serialized_key = key.encode("utf-8") if key else None
            # Send DataFrame to Kafka topic
            self.producer.produce(
                topic=topic,
                value=serialized_data,
                key=serialized_key,
                headers=headers,
                callback=self.delivery_callback,
            )
            self.logger.debug(f"DataFrame produced to topic '{topic}'")
            self.producer.poll(0)  # Trigger delivery reports
        except BufferError:
            self.logger.warning("Local producer queue is full, waiting for space...")
            self.producer.flush()
            # Try again after flush
            self.producer.produce(
                topic=topic,
                value=serialized_data,
                key=serialized_key,
                headers=headers,
                callback=self.delivery_callback,
            )
        except Exception as e:
            self.logger.error(f"Error producing DataFrame message: {str(e)}")
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


class SparkRowConsumer:
    """Kafka consumer for receiving individual PySpark Row objects."""

    def __init__(self, config, topics, spark_session=None):
        """
        Initialize the PySpark Row consumer.

        Args:
            config: KafkaConfig object with connection settings
            topics: List of topics to subscribe to
            spark_session: Optional SparkSession for Row conversions
        """
        self.consumer_config = config.get_consumer_config()
        self.consumer = Consumer(self.consumer_config)
        self.topics = topics if isinstance(topics, list) else [topics]
        self.consumer.subscribe(self.topics)
        self.serializer = SparkRowSerializer()
        self.spark = spark_session
        self.logger = logging.getLogger("SparkRowConsumer")
        self.logger.info(f"SparkRowConsumer initialized for topics: {self.topics}")

        # Track metrics
        self.messages_received = 0
        self.errors_encountered = 0
        self.total_bytes_received = 0
        self.running = False

    def receive_row(self, timeout=1.0):
        """
        Poll for and receive a PySpark Row from subscribed topics.

        Args:
            timeout: Timeout in seconds to wait for a message

        Returns:
            Tuple of (row_dict, metadata) or (None, None) if no message
        """
        self.logger.debug(f"Polling for Row messages with timeout {timeout}s")
        msg = self.consumer.poll(timeout)

        if msg is None:
            self.logger.debug("No message received during poll")
            return None, None

        if msg.error():
            self.errors_encountered += 1
            error_code = msg.error().code()
            if error_code == KafkaError._PARTITION_EOF:
                self.logger.info(
                    f"Reached end of partition {msg.topic()}-{msg.partition()}"
                )
            else:
                self.logger.error(f"Error while consuming message: {msg.error()}")
            return None, None

        # Message received successfully
        self.messages_received += 1
        message_size = len(msg.value()) if msg.value() else 0
        self.total_bytes_received += message_size
        self.logger.info(
            f"Received message from {msg.topic()} [{msg.partition()}] at offset {msg.offset()}, size: {message_size} bytes"
        )

        try:
            # Parse Row message
            row_dict, metadata = self.serializer.deserialize_message(
                msg.value(), msg.headers()
            )

            # Convert to PySpark Row if SparkSession is available
            if self.spark is not None:
                self.logger.debug("Converting dict to PySpark Row")
                spark_row = Row(**row_dict)
                self.logger.info("Row parsed successfully")
                return spark_row, metadata

            self.logger.info("Row dictionary parsed successfully")
            return row_dict, metadata

        except Exception as e:
            self.errors_encountered += 1
            self.logger.error(f"Error deserializing Row message: {str(e)}")
            return None, None

    def start_consuming(self, handler, poll_timeout=1.0):
        """
        Start consuming messages in a loop, processing each with the provided handler.

        Args:
            handler: Callback function to process each Row and metadata
            poll_timeout: Time to wait for a message in each poll cycle (in seconds)
        """
        self.running = True

        def consume_loop():
            while self.running:
                try:
                    row, metadata = self.receive_row(poll_timeout)
                    if row is not None:
                        handler(row, metadata)
                        self.commit()
                except Exception as e:
                    self.logger.error(f"Error during Row consumption: {e}")

        # Start consumption in a separate thread
        self.consumer_thread = threading.Thread(target=consume_loop)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        self.logger.info("Started Row consumer thread")

    def stop_consuming(self):
        """Stop the consumer thread."""
        self.running = False
        if hasattr(self, "consumer_thread"):
            self.consumer_thread.join(timeout=5.0)
        self.logger.info("Stopped consumer thread")

    def commit(self):
        """Commit offsets."""
        self.logger.debug("Committing offsets")
        self.consumer.commit()

    def close(self):
        """Close the consumer connection."""
        self.logger.info("Closing consumer connection")
        self.stop_consuming()
        self.consumer.close()
        self.logger.info(
            f"Consumer stats: {self.messages_received} messages received, "
            f"{self.errors_encountered} errors, "
            f"{self.total_bytes_received} bytes total data"
        )


class DataFrameConsumer:
    """Kafka consumer for receiving DataFrame messages."""

    def __init__(self, config, topics, spark_session=None):
        """
        Initialize the DataFrame consumer.
        Args:
            config: KafkaConfig object with connection settings
            topics: List of topics to subscribe to
            spark_session: Optional SparkSession for DataFrame conversions
        """
        self.consumer_config = config.get_consumer_config()
        self.consumer = Consumer(self.consumer_config)
        self.topics = topics if isinstance(topics, list) else [topics]
        self.consumer.subscribe(self.topics)
        self.serializer = DataFrameSerializer()
        self.spark = spark_session
        self.logger = logging.getLogger("DataFrameConsumer")
        self.logger.info(f"DataFrameConsumer initialized for topics: {self.topics}")
        # Track metrics
        self.messages_received = 0
        self.errors_encountered = 0
        self.total_bytes_received = 0
        self.running = False

    def receive_dataframe(self, timeout=1.0):
        """
        Poll for and receive a DataFrame from subscribed topics.
        Args:
            timeout: Timeout in seconds to wait for a message
        Returns:
            Tuple of (pandas_df, metadata) or (None, None) if no message
        """
        self.logger.debug(f"Polling for DataFrame messages with timeout {timeout}s")
        msg = self.consumer.poll(timeout)

        if msg is None:
            self.logger.debug("No message received during poll")
            return None, None

        if msg.error():
            self.errors_encountered += 1
            error_code = msg.error().code()
            if error_code == KafkaError._PARTITION_EOF:
                self.logger.info(
                    f"Reached end of partition {msg.topic()}-{msg.partition()}"
                )
            else:
                self.logger.error(f"Error while consuming message: {msg.error()}")
            return None, None

        # Message received successfully
        self.messages_received += 1
        message_size = len(msg.value()) if msg.value() else 0
        self.total_bytes_received += message_size
        self.logger.info(
            f"Received message from {msg.topic()} [{msg.partition()}] at offset {msg.offset()}, size: {message_size} bytes"
        )

        try:
            # Parse DataFrame message
            pandas_df, metadata = self.serializer.deserialize_message(
                msg.value(), msg.headers()
            )

            # Convert to PySpark DataFrame if SparkSession is available
            if self.spark is not None:
                self.logger.debug("Converting pandas DataFrame to PySpark DataFrame")
                spark_df = self.spark.createDataFrame(pandas_df)
                self.logger.info("DataFrame parsed successfully")
                return spark_df, metadata

            self.logger.info("DataFrame parsed successfully")
            return pandas_df, metadata

        except Exception as e:
            self.errors_encountered += 1
            self.logger.error(f"Error deserializing DataFrame message: {str(e)}")
            return None, None

    def start_consuming(self, handler, poll_timeout=1.0):
        """
        Start consuming messages in a loop, processing each with the provided handler.
        Args:
            handler: Callback function to process each DataFrame and metadata
            poll_timeout: Time to wait for a message in each poll cycle (in seconds)
        """
        self.running = True

        def consume_loop():
            while self.running:
                try:
                    df, metadata = self.receive_dataframe(poll_timeout)
                    if df is not None:
                        handler(df, metadata)
                        self.commit()
                except Exception as e:
                    self.logger.error(f"Error during DataFrame consumption: {e}")

        # Start consumption in a separate thread
        self.consumer_thread = threading.Thread(target=consume_loop)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        self.logger.info("Started DataFrame consumer thread")

    def stop_consuming(self):
        """Stop the consumer thread."""
        self.running = False
        if hasattr(self, "consumer_thread"):
            self.consumer_thread.join(timeout=5.0)
        self.logger.info("Stopped consumer thread")

    def commit(self):
        """Commit offsets."""
        self.logger.debug("Committing offsets")
        self.consumer.commit()

    def close(self):
        """Close the consumer connection."""
        self.logger.info("Closing consumer connection")
        self.stop_consuming()
        self.consumer.close()
        self.logger.info(
            f"Consumer stats: {self.messages_received} messages received, "
            f"{self.errors_encountered} errors, "
            f"{self.total_bytes_received} bytes total data"
        )
