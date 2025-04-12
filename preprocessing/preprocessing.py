from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, current_timestamp
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
from PIL import Image
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT

spark = SparkSession.builder.appName("ImagePreprocessing").getOrCreate()

# A different file path will be used for this part later, I just used test3.png to make sure it works.
# Feel free to add an image into this directory and replace the path here with the image file's name to test it out.
image_df = spark.read.format("image").load("./test3.png")

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

df = preprocess_image(image_df)
df.show()
vector_to_image(df, 0)