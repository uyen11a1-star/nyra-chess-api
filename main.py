import os
import subprocess
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

SERVICE_NAME = "Nyra Chess API (C++ Core)"
VERSION = "1.0.0"

app = FastAPI(title=SERVICE_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://uyen11a1-star.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Cấu hình tải mô hình (nếu chưa có sẵn trên Render)
MODEL_PATH = Path("nyra_net.onnx")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_URL = "https://huggingface.co/huynek231/nyra-chess-weights/resolve/main/nyra_net.onnx"

@app.on_event("startup")
def startup_event():
    # Tự động tải mô hình ONNX nếu chưa có
    if not MODEL_PATH.is_file():
        print("Downloading ONNX model from HuggingFace...")
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        response = requests.get(MODEL_URL, headers=headers, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully!")
        else:
            print(f"Failed to download model: HTTP {response.status_code}")

class MoveRequest(BaseModel):
    fen: str
    legal_moves: list[Any] = Field(default_factory=list)

@app.get("/")
def root():
    return {"service": SERVICE_NAME, "version": VERSION, "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ready", "engine": "C++"}

@app.post("/move")
def choose_move(request: MoveRequest):
    try:
        # Gọi file C++ engine và truyền FEN vào
        result = subprocess.run(
            ["./nyra_engine", request.fen],
            capture_output=True,
            text=True,
            timeout=10 # Timeout 10 giây đề phòng kẹt
        )
        
        best_move = result.stdout.strip()
        if not best_move or best_move == "NONE":
            raise HTTPException(status_code=400, detail="No move found")
            
        return {
            "uci": best_move,
            "search": {
                "depth": 4,
                "method": "C++ Alpha-Beta + ONNX"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
