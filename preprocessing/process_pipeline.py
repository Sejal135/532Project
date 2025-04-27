from pyspark.sql import SparkSession
from preprocessing.preprocessing import preprocess_image

from preprocessing.kafka import KafkaConfig, SparkRowProducer

# Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
PROCESSED_IMAGE_TOPIC = "image_embedding_requests"
CSV_PATH = "image_data.csv"
IMAGE_DIR = "images/"


def main():
    config = KafkaConfig(KAFKA_BOOTSTRAP_SERVERS)
    prouducer = SparkRowProducer(config)

    # Starts the Spark Session
    spark = SparkSession.builder.appName("ImagePreprocessing").getOrCreate()

    # Loads all of the images into a dataframe
    image_df = spark.read.format("image").load(IMAGE_DIR)

    # Preprocessed it with Pyspark and Torch
    preprocessed_df = preprocess_image(image_df)

    if preprocessed_df is None:
        raise Exception("Preprocessing images with Spark error")

    # Stream the processed image data to CLIP
    prouducer.send_dataframe_rows(PROCESSED_IMAGE_TOPIC, preprocessed_df)


if __name__ == "__main__":
    main()
