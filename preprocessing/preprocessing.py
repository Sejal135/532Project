from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, current_timestamp
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
from PIL import Image
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT

spark = SparkSession.builder.appName("ImagePreprocessing").getOrCreate()

IMAGE_DIR = "images/"
image_df = spark.read.format("image").load(IMAGE_DIR)

def preprocess_image(dataframe):
    try:
        def image_to_vector(img):
            arr = ImageSchema.toNDArray(img)
            height, width, channels = arr.shape

            if channels == 4:
                arr = arr[:, :, [2, 1, 0, 3]]
            elif channels == 3:
                arr = arr[:, :, [2, 1, 0]]

            return DenseVector(arr.flatten())

        ImageSchema.imageFields
        img2vec = udf(image_to_vector, VectorUDT())
        df = dataframe.withColumn('vecs', img2vec("image")).withColumn('preprocess_time', current_timestamp())

        return df
    except Exception as e:
        print(e)

def vector_to_image(dataframe, index):
    try:
        row = dataframe.collect()[index]
        
        image = row["image"]
        height = image.height
        width = image.width
        n_channels = image.nChannels
        vec = row["vecs"]

        np_img = np.array(vec).reshape((height, width, n_channels))  

        new_img = Image.fromarray(np_img.astype(np.uint8))
        new_img.show()
    except Exception as e:
        print(e)


vectorized_df = preprocess_image(image_df)
vectorized_df.select("image.origin", "vecs", "preprocess_time").show()

vector_to_image(vectorized_df, 0)