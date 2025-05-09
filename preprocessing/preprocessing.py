from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, current_timestamp, col, struct
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
from PIL import Image
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT
import torch
from torchvision import transforms

# CLIP preprocess based on 
# https://www.kaggle.com/code/dailysergey/howtodata-interacting-with-clip#Setting-up-input-images-and-texts
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])
# Starts the Spark Session
spark = SparkSession.builder.appName("ImagePreprocessing").getOrCreate()

# Image directory from where images would be read
IMAGE_DIR = "images/"

# Loads all of the images into a dataframe
image_df = spark.read.format("image").load(IMAGE_DIR)

def preprocess_image(dataframe):
    try:
        # Converts an image to a vector
        def image_to_vector(img, channels):
            arr = ImageSchema.toNDArray(img)

            # Modifies the RGBA channels because the colors would be incorrect otherwise when
            # we convert to an array
            if channels == 4:
                arr = arr[:, :, [2, 1, 0, 3]]
            elif channels == 3:
                arr = arr[:, :, [2, 1, 0]]

            return DenseVector(arr.flatten())

        # Due to how Spark reads in images, the images have to be converted into vectors,
        # then the vectors would be converted back into images such as they can then be converted
        # into tensors via Torch
        def vector_to_tensor(image):
            try:
                height = image.height
                width = image.width
                n_channels = image.nChannels

                # Converts the image to a vector
                vec = image_to_vector(image, n_channels)

                np_img = np.array(vec).reshape((height, width, n_channels))  

                # Recreates the vector into an image
                new_img = Image.fromarray(np_img.astype(np.uint8))

                # The tensor is the result of preprocessing the image through Torch,
                # so that it can work with CLIP
                tensor = clip_preprocess(new_img)
                return DenseVector(tensor.flatten().tolist())
            except Exception as e:
                print(e)

        # Remove raw bytes field from image
        def remove_image_data(df):
            new_image_col = struct(
                col("image.origin").alias("origin"),
                col("image.height").alias("height"),
                col("image.width").alias("width"),
                col("image.nChannels").alias("nChannels"),
                col("image.mode").alias("mode")
            )
            
            return df.withColumn("image", new_image_col)

        # The final dataframe would contain the images transformed into tensors for CLIP,
        # along with the times at which they were preprocessed
        ImageSchema.imageFields
        img2vec = udf(vector_to_tensor, VectorUDT())

        df = dataframe.withColumn('tensors', img2vec("image")).withColumn('preprocess_time', current_timestamp())
        df = remove_image_data(df)
        return df
    except Exception as e:
        print(e)


# Converts a tensor back to an image
def tensor_to_img(img_tensor):
    tensor = torch.tensor(img_tensor).reshape(3, 224, 224)

    # Unnormalize the CLIP tensor
    unnormalize = transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
    )
    tensor = unnormalize(tensor).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    image.show()

# for i in range(vectorized_df.count()):
#     tensor_to_img(vectorized_df.collect()[i]["tensors"])
# print(preprocess_image(image_df).first()["image"])