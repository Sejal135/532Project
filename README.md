# 532Project

## Installation

```bash
git clone https://github.com/yourusername/532Project.git
cd 532Project
pip install -r requirements.txt
```
## Setup
Images to be processed and stored in the pinecone database must be put into the images/ directory. We used [Flikr 30k Images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) for testing.

### Preprocessing
[Install Docker](https://www.docker.com/get-started/) onto your computer and run daemon thread.
Run following command once Docker is installed to setup the Kafka server:
```bash
docker compose up
```
To begine pyspark preprocessing of images directory:
```bash
python preprocessing/process_pipeline.py
```
Run the following to start the image consumer. The image consumer will consume messages from the kafka topic, create embeddings from incoming image vectors, and send image embeddings to the pinecone database.
```bash
python CLIP_server/image_consumer.py
```

### Image Search
First, start the query server so user queries can be embedded using the Text CLIP model:
```bash
python CLIP_server/query_server.py --port <port>
```
Next, start the query pipeline to retrieve an image based on a natural language query. For an image to actually open on your computer, you must have it downloaded on your local computer with the same name as the one stored on pinecone. The query "otter swimming" should work with the test image in images/.
```bash
python main_client.py
```

## Tests / Experiments

### Preprocessing pipeline
By running the preprocessing pipeline with image_consumer.py and process_pipeline.py as explained above, it will automatically complete the pipleline with test images in the images/directory with no other setup.

### Query Server
Python script is set up to test the query server with 10,000 randomized queries with 1000 threads at a time. Test results will be printed out once all queries are processed by the server.
```bash
python CLIP_server/test_query_server.py
```

### End-to-End Pipeline
To test the full user-side pipeline from user query to returned image, set the custom_query variable in main_client.py to False. This will run a test which sends 50 requests randomly in 5 second intervals, in order to simulate realistic user loads. Tests results are printed out after completion.
```bash
python main_client.py
```

### Image Model Throughput
The image and text CLIP models will log their throughput automatically every 5 seconds, so the query server and image consumers will both display throughput over time. To visualize the image mdoel's throughput over time after doing image preprocessing, run the following script which will analyze the latest Kafka logs and plot throughput over time.
```bash
python plot_throughput.py
```