from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    udf,
    from_json,
    struct,
    to_json,
    current_timestamp,
)
from pyspark.sql.types import StringType, StructType, StructField, BinaryType
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.image import ImageSchema
import numpy as np
import pandas as pd
import os
import json
import base64
from confluent_kafka import Producer, Consumer
from PIL import Image
import io

# Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_INPUT_TOPIC = "image-preprocessing-input"
KAFKA_OUTPUT_TOPIC = "image-preprocessing-output"
CSV_PATH = "image_data.csv"
IMAGE_DIR = "images/"


def ingest_images_to_kafka():
    """
    Read images from a directory based on a CSV file and ingest them into Kafka.
    CSV format: image_id,caption
    """
    # Create Kafka producer
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    # Read CSV file
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} images in CSV file")

    # Process each row in the CSV
    for index, row in df.iterrows():
        image_id = row["image_id"]
        caption = row["caption"]
        image_path = os.path.join(IMAGE_DIR, f"{image_id}")

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found, skipping")
            continue

        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()

            img_base64 = base64.b64encode(img_data).decode("utf-8")

            message = {
                "image_id": image_id,
                "caption": caption,
                "image_data": img_base64,
            }

            producer.produce(
                topic=KAFKA_INPUT_TOPIC, value=json.dumps(message).encode("utf-8")
            )
            print(f"Sent image {image_id} to Kafka input topic")

        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")

    producer.flush()
    print("All images sent to Kafka input topic")


def process_images_with_spark():
    """
    Process images from Kafka using Spark and send results to another Kafka topic
    """
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("KafkaImagePreprocessing")
        .config(
            "spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0"
        )
        .getOrCreate()
    )

    # Define schema for Kafka messages
    schema = StructType(
        [
            StructField("image_id", StringType(), True),
            StructField("caption", StringType(), True),
            StructField("image_data", StringType(), True),
        ]
    )

    # Read from Kafka
    df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_INPUT_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    # Parse JSON from Kafka
    parsed_df = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    # Convert base64 to image
    def base64_to_image(base64_str):
        try:
            # Decode base64 to binary
            img_data = base64.b64decode(base64_str)
            # Create PIL Image
            img = Image.open(io.BytesIO(img_data))
            # Convert to numpy array
            img_array = np.array(img)
            # Create image schema
            height, width, channels = img_array.shape
            return {
                "origin": "base64",
                "height": height,
                "width": width,
                "nChannels": channels,
                "mode": "RGB",
                "data": img_data,
            }
        except Exception as e:
            print(f"Error converting base64 to image: {str(e)}")
            return None

    # Create UDF for base64 to image conversion
    base64_to_image_udf = udf(base64_to_image, ImageSchema.imageSchema)

    # Apply conversion
    df_with_images = parsed_df.withColumn(
        "image", base64_to_image_udf(col("image_data"))
    )

    # Preprocess images (using the provided preprocessing function)
    def preprocess_image(img):
        try:
            arr = ImageSchema.toNDArray(img)
            height, width, channels = arr.shape

            if channels == 4:
                arr = arr[:, :, [2, 1, 0, 3]]
            elif channels == 3:
                arr = arr[:, :, [2, 1, 0]]

            return DenseVector(arr.flatten())
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    # Register UDF
    img2vec = udf(preprocess_image, VectorUDT())

    # Apply preprocessing
    processed_df = df_with_images.withColumn("vecs", img2vec("image")).withColumn(
        "preprocess_time", current_timestamp()
    )

    # Function to send processed data to output Kafka topic

    def send_to_kafka_output(batch_df, batch_id):
        producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
        rows = batch_df.collect()

        for row in rows:
            try:
                vec = row["vecs"]
                img = row["image"]
                height = img.height
                width = img.width
                channels = img.nChannels
                np_img = np.array(vec).reshape((height, width, channels))

                pil_img = Image.fromarray(np_img.astype(np.uint8))
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

                processed_img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

                payload = {
                    "image_id": row["image_id"],
                    "caption": row["caption"],
                    "processed_image": processed_img_base64,
                    "preprocess_time": str(row["preprocess_time"]),
                }

                producer.produce(
                    topic=KAFKA_OUTPUT_TOPIC, value=json.dumps(payload).encode("utf-8")
                )
                print(
                    f"Successfully sent processed image {row['image_id']} to Kafka output topic"
                )

            except Exception as e:
                print(f"Error sending processed image to Kafka output topic: {str(e)}")

        producer.flush()

    # Write results to output Kafka topic
    query = (
        processed_df.writeStream.foreachBatch(send_to_kafka_output)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()


def create_kafka_consumer():
    consumer_conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "image-processing-group",
        "auto.offset.reset": "earliest",
    }

    consumer = Consumer(consumer_conf)
    consumer.subscribe([KAFKA_OUTPUT_TOPIC])

    print(f"Starting to consume messages from {KAFKA_OUTPUT_TOPIC}...")
    try:
        while True:
            msg = consumer.poll(1.0)  # timeout in seconds
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            payload = json.loads(msg.value().decode("utf-8"))
            print(f"Received processed image: {payload['image_id']}")
            # Optionally save image
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    # First, ingest images to Kafka
    ingest_images_to_kafka()

    # Then, process images with Spark
    process_images_with_spark()

    # Optionally, consume processed images to verify
    # Uncomment the line below to run the consumer
    # create_kafka_consumer()
