from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, current_timestamp
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
from PIL import Image
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT

# Starts the Spark Session
spark = SparkSession.builder.appName("ImagePreprocessing").getOrCreate()

# Image directory from where images would be read
IMAGE_DIR = "images/"

# Loads all of the images into a dataframe
image_df = spark.read.format("image").load(IMAGE_DIR)

def preprocess_image(dataframe):
    try:
        # Converts an image to a vector
        def image_to_vector(img):
            arr = ImageSchema.toNDArray(img)
            height, width, channels = arr.shape

            # Modifies the RGBA channels because the colors would be incorrect otherwise when
            # we convert to an array
            if channels == 4:
                arr = arr[:, :, [2, 1, 0, 3]]
            elif channels == 3:
                arr = arr[:, :, [2, 1, 0]]

            return DenseVector(arr.flatten())

        # The final dataframe would contain the vectorized images with the times at which they were preprocessed
        ImageSchema.imageFields
        img2vec = udf(image_to_vector, VectorUDT())
        df = dataframe.withColumn('vecs', img2vec("image")).withColumn('preprocess_time', current_timestamp())

        return df
    except Exception as e:
        print(e)

def vector_to_image(dataframe, index):
    try:
        # Retrieves the image on the index-th row of the vectorized images dataframe
        row = dataframe.collect()[index]
        
        image = row["image"]
        height = image.height
        width = image.width
        n_channels = image.nChannels
        vec = row["vecs"]

        # Reshapes the vector into the original image size so that the returned image is correct
        np_img = np.array(vec).reshape((height, width, n_channels))  

        new_img = Image.fromarray(np_img.astype(np.uint8))
        new_img.show()
    except Exception as e:
        print(e)

# This dataframe contains all of the original images alongside their corresponding vectors, plus some metadata
vectorized_df = preprocess_image(image_df)
vectorized_df.select("image.origin", "vecs", "preprocess_time").show()

# Debugging to make sure vectorizing images still results in the same images when converted back
vector_to_image(vectorized_df, 0)