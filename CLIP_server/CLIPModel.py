from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer
from queue import Queue, Empty
from threading import Thread, Event
import time
import torch
import logging
from ImageDirectory import ImageDirectory
# Probably will have to account for possibility that you can only have one clip model in memory at a time



class EmbeddingRequest:
    """Tracks a pending embedding request and its eventual result."""
    def __init__(self, content):
        self.content = content
        self.result = None
        self.done = Event()
    
    def set_result(self, result):
        self.result = result
        self.done.set()
        
    def wait_for_result(self, timeout=None):
        self.done.wait(timeout)
        return self.result

class CLIP:
    def __init__(self, batch_size: int, timeout: float, model_id: str = "openai/clip-vit-base-patch32"):
        print(f"Loading model {model_id}...")
        self.model, self.tokenizer = self._get_tokenizer_and_model(model_id)
        print("Model loaded successfully.")
        # Batch size for processing requests
        self.batch_size = batch_size
        # Timeout for processing requests
        self.timeout = timeout
        # Stores embedding requests to be processed in embedding thread
        self.queue = Queue(1_000)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Tracking
        self.images_processed_last_5_seconds = 0
        self.start_time = time.time()

        # Start the embedding thread
        self.running = True
        print("Starting embedding thread...")
        self.thread = Thread(target=self._start_processing)
        self.thread.daemon = True
        self.thread.start()
        print("Embedding thread started.")

    def _get_tokenizer_and_model(self, model_id):
        return None, None
    
    def get_embedding(self, item, timeout=5):
        """Item will be a string for query embeddings, and an image path for image embeddings."""
        # blocking call that waits on the embedding thread
        request = EmbeddingRequest(item)
        self.queue.put(request)
        result = request.wait_for_result(timeout)
        if result is None:
            raise TimeoutError("Embedding request timed out")
        return result
    
    def _start_processing(self):
        curr_batch = []
        curr_requests = []
        last_arrival = time.time()
        
        while self.running:
            try:
                # Try to get a new request with a small timeout
                request = self.queue.get(timeout=0.1)
                curr_batch.append(request.content)
                curr_requests.append(request)
                last_arrival = time.time()
                
                # Process batch if we've reached batch size
                if len(curr_batch) >= self.batch_size:
                    self.images_processed_last_5_seconds += len(curr_batch)
                    self._process_batch(curr_batch, curr_requests)
                    curr_batch = []
                    curr_requests = []
                    
            except Empty:
                # If we have items and timed out waiting for more, process what we have
                if curr_batch and time.time() - last_arrival >= self.timeout:
                    self.images_processed_last_5_seconds += len(curr_batch)
                    self._process_batch(curr_batch, curr_requests)
                    curr_batch = []
                    curr_requests = []

            if time.time() - self.start_time > 5:
                self.logger.info(f"Throughput: {self.images_processed_last_5_seconds / (time.time() - self.start_time): .2f} requests/sec")
                self.images_processed_last_5_seconds = 0
                self.start_time = time.time()
    
    
    def _process_batch(self, batch, requests):
        raise NotImplementedError("Batch processing not implemented yet")
    
    def shutdown(self):
        print("Shutting down embedding thread...")
        self.running = False
        self.thread.join()



class TextCLIP(CLIP):
    def __init__(self, batch_size: int, timeout: float, model_id: str = "openai/clip-vit-base-patch32"):
        super().__init__(batch_size, timeout, model_id)

    def _get_tokenizer_and_model(self, model_id):
        return CLIPTextModelWithProjection.from_pretrained(model_id), CLIPTokenizer.from_pretrained(model_id)
    
    def _process_batch(self, texts, requests):
        """Process batch of texts and set results in requests."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert to numpy for easier handling
        embeddings = outputs.text_embeds.detach().cpu().numpy()
        
        # Distribute results back to requesters
        for i, request in enumerate(requests):
            request.set_result(embeddings[i])


class ImageCLIP(CLIP):
    def __init__(self, batch_size: int, timeout: float, image_dir: str, model_id: str = "openai/clip-vit-base-patch32"):
        super().__init__(batch_size, timeout, model_id)
        self.images = ImageDirectory(image_dir)

    def _get_tokenizer_and_model(self, model_id):
        return CLIPVisionModelWithProjection.from_pretrained(model_id), None
    
    def _process_batch(self, vectors, requests):
        """Process batch of images and set results in requests."""
        vectors = torch.stack([torch.tensor(v.toArray().copy().reshape(3, 224, 224)) for v in vectors])
        # [1, 150528])
        # batch_size, _, height, width = pixel_values.shape
        print("Shape:", vectors.shape)
        inputs = {"pixel_values": vectors}
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert to numpy for easier handling
        embeddings = outputs.image_embeds.detach().cpu().numpy()
        
        # Distribute results back to requesters
        for i, request in enumerate(requests):
            request.set_result(embeddings[i])
