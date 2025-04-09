from fastapi import FastAPI, Request
import uvicorn
import argparse
from CLIPModel import TextCLIP, ImageCLIP

app = FastAPI(title="CLIP Server")
text_model = TextCLIP(batch_size=32, timeout=2.0)
image_model = ImageCLIP(batch_size=32, timeout=2.0, image_dir="images/")

@app.get("/")
def home():
    return {"message": "Welcome to the CLIP Server!"}

@app.get("/query_embedding")
def create_query_embedding(request: Request):
    text = request.query_params.get("text")
    if not text:
        return {"error": "No text provided"}
    
    # Process the text and return the embedding
    embedding = text_model.get_embedding(text, timeout=10)
    return {"embedding": embedding.tolist()}

@app.post("/image_embedding")
def create_image_embedding(request: Request):
    image_name = request.query_params.get("image_name")
    metadata = request.query_params.get("metadata")
    if not image_name:
        return {"error": "No image name provided"}
    
    # Process the image and return the embedding
    embedding = image_model.get_embedding(image_name, timeout=10)
    # TODO: store embedding and metadata in pincone dataset here

    return {"status": "success"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")

    args = parser.parse_args()
    
    print(f"Starting CLIP server on http://localhost:{args.port}")
    uvicorn.run(app, port=args.port)