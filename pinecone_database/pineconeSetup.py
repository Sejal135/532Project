import os
import pinecone
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from preprocessing import preprocess_image

def ingest():
    # 1 Init Pinecone
    api_key = os.getenv("pcsk_3eJ2cq_MYjUEN6HXnQN9X8RTjidvSrKT3AmYsij6oD2URuEccV5r8CyKgz7gNQ6QLf2P7x")
    pinecone_env = os.getenv("us-east1-aws")
    index_name = os.getenv("dense-image-index")

    pinecone.init(
        api_key=api_key,
        environment=pinecone_env
    )

    # 2️ Start Spark & load raw images
    spark = SparkSession.builder.appName("Spark→Pinecone").getOrCreate()
    IMAGE_DIR = "images/"
    image_df = spark.read.format("image").load(IMAGE_DIR)

    # create a simple column for the image path:
    df = image_df.withColumn("image_path", col("image.origin"))

     # 3️ Preprocess to get embeddings with PySpark and Torch
    processed_df = preprocess_image(df)
    # processed_df schema: [ image:struct, image_path:string, tensors:vector, preprocess_time:timestamp ]

    # 4️ Determine your vector dimension
    first_vec = processed_df.select("tensors").head()[0]
    dim = len(first_vec)
    
    # 5 Created dense index
    if index_name not in pinecone.list_indexes():
        # assume all your vectors have the same dim
        pinecone.create_index(name=index_name, dimension=dim, metric="cosine")
    index = pinecone.Index(index_name)

    # Ensure df contains at least:
    #   • "image"→ struct with all image metadata such as height, width, data, mode, nChannels
    #   • "image_path"→ string relative path or filename for retrieval
    #   • "tensors"   → list/ndarray of floats
    #   • "preprocess_time"→ timestamp for retrieval a single image

    # 6 Prepare upsert tuples
    to_upsert = []
    rows = processed_df.select("image_path", "tensors", "preprocess_time").collect()

    for row in rows:
        vid = os.path.basename(row["image_path"])
        vec = row["tensors"].toArray().tolist()
        meta = {
            "image_path": row["image_path"],
            "preprocess_time": row["preprocess_time"].isoformat(),
        }
        to_upsert.append((vid, vec, meta))

    BATCH = 100
    for i in range(0, len(to_upsert), BATCH):
        index.upsert(vectors=to_upsert[i : i + BATCH])

    print(f"Correctly Upserted {len(to_upsert)} items into '{index_name}'")
    spark.stop()

if __name__ == "__main__":
    ingest()