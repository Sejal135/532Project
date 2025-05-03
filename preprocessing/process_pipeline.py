from pyspark.sql import SparkSession
import time
from kafka import KafkaConfig, SparkRowProducer
from preprocessing import preprocess_image


# Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
PROCESSED_IMAGE_TOPIC = "image_embedding_requests"
CSV_PATH = "image_data.csv"
IMAGE_DIR = "images/"


config = KafkaConfig(KAFKA_BOOTSTRAP_SERVERS)
producer = SparkRowProducer(config)


def process_partition(partition_iterator):
    from pyspark.sql import SparkSession
    from preprocessing import preprocess_image

    config = KafkaConfig(KAFKA_BOOTSTRAP_SERVERS)
    producer = SparkRowProducer(config)

    rows = list(partition_iterator)
    if not rows:
        return

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(rows)

    processed_df = preprocess_image(df)

    # Efficient, streaming-safe send
    producer.send_partition_rows(
        PROCESSED_IMAGE_TOPIC, processed_df.rdd.toLocalIterator()
    )


def main():
    spark = (
        SparkSession.builder.appName("ImagePreprocessing")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .config("spark.executor.memoryOverhead", "1g")
        .getOrCreate()
    )

    image_df = spark.read.format("image").load(IMAGE_DIR)

    image_df = preprocess_image(image_df)

    producer.send_partition_rows(PROCESSED_IMAGE_TOPIC, image_df.rdd.toLocalIterator())


if __name__ == "__main__":
    try:
        main()
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}") # Print the specific error
        import traceback
        traceback.print_exc() # Print the full stack trace
        exit(1) # Exit with a non-zero code to indicate an error
