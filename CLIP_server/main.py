from fastapi import FastAPI, HTTPException
import uvicorn
import argparse

app = FastAPI(title="CLIP Server")


@app.get("/")
def home():
    return {"message": "Welcome to the CLIP Server!"}

@app.get("/query_embedding")
def create_query_embedding():
    pass

@app.post("/image_embedding")
def create_image_embedding():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")

    args = parser.parse_args()
    
    print(f"Starting CLIP server on http://localhost:{args.port}")
    uvicorn.run(app, port=args.port)