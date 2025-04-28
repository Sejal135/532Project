import sys, os
sys.path.append(os.getcwd())

from fastapi import FastAPI, Request
import uvicorn
import argparse
from CLIPModel import TextCLIP
from fastapi import HTTPException

app = FastAPI(title="CLIP Server")
text_model = TextCLIP(batch_size=32, timeout=5.0)

@app.get("/")
def home():
    return {"message": "Welcome to the CLIP Server!"}


@app.get("/query_embedding")
def create_query_embedding(request: Request):
    text = request.query_params.get("text")
    if not text:
        return HTTPException(status_code=400, detail="No text provided")

    # Process the text and return the embedding
    try:
        embedding = text_model.get_embedding(text, timeout=10)
        return {"embedding": embedding.tolist()}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )

    args = parser.parse_args()

    print(f"Starting CLIP server on http://localhost:{args.port}")
    uvicorn.run(app, port=args.port)
